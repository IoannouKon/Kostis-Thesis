package linear_tranform

import chisel3._
import chisel3.util._
import chisel3.iotesters.{PeekPokeTester, Driver}
import scala.util.Random

/**
 * Class: MultiLinearFilter_fsm
 * 
 * This Chisel module implements a multi-filter linear operation with a finite state machine (FSM) architecture.
 * It processes multiple rows of input data and outputs the result through multiple filters, supporting vectorized 
 * processing of weights, biases, and inputs.
 * 
 * @param inputRows    Number of input rows to be processed.
 * @param inputCols    Number of input columns in each row.
 * @param outputCols   Number of output columns (length of the output vectors per filter).
 * @param dataWidth    Bit width of the input/output data.
 * @param numFilters   Number of filters (i.e., the number of independent output channels).
 * @param VectorSize_w Vector size for load weights (must divide inputCols * outputCols).
 * @param VectorSize_b Vector size for loaf biases (must divide outputCols).
 * @param VectorSize_in Vector size for input elements load per cycle (must divide inputCols).
 * @param VectorSize_out Vector size for output elements exit(outs) per cycle (must divide outputCols).
 * @param num_rows     Number of input rows to be processed-store in registers at a time (must be <= inputRows).
 *
 * The module enforces these constraints using the require() statements to ensure the vector sizes 
 * are compatible with the input and output dimensions.
 *
 * ### Inputs:
 * - **input_x_elem**: A vector of `VectorSize_in` elements, each of which is a Decoupled interface 
 *   carrying input data (width of each element is `dataWidth` bits). Each clock cycle processes
 *   a chunk of input data. The interface allows the module to consume input data asynchronously.
 *
 * - **weight_elems**: A 2D vector that stores `VectorSize_w` elements for each of the `numFilters`.
 *   Each element is a Decoupled interface carrying weight data (width of each element is `dataWidth` bits).
 *   Weights are provided separately for each filter and can be processed asynchronously.
 *
 * - **bias_elems**: A 2D vector storing bias values, where each filter has `VectorSize_b` bias elements.
 *   Biases are also provided using the Decoupled interface, allowing the module to process them asynchronously.
 *
 * - **ISbias**: A control signal input (width is `dataWidth` bits), which indicates whether the current 
 *   operation should include the bias in the computation or not.
 *
 * ### Outputs:
 * - **output_elems**: A 2D vector that stores the processed output data for each filter. Each filter produces
 *   `VectorSize_out` output elements at a time (with each element being `dataWidth` bits wide) and is also a 
 *   Decoupled interface, allowing the output to be consumed asynchronously.
 *
 * ### Functionality:
 * The module performs matrix multiplication between the input data and weights, applies biases (if `ISbias` is enabled),
 * and produces the results for multiple filters. It processes the inputs in batches of `num_rows` rows and
 * produces the corresponding outputs in the same batched manner.
 */

class MultiLinearFilter_fsm( val inputRows: Int, val inputCols: Int, val outputCols: Int, val dataWidth: Int,val numFilters: Int,val VectorSize_w:Int,val VectorSize_b :Int, val VectorSize_in :Int , val VectorSize_out :Int,val num_rows :Int) extends Module {
  
  require(VectorSize_w <= inputCols * outputCols, "VectorSize_w must be less than or equal to inputCols * outputCols")
  require(VectorSize_b <= outputCols, "VectorSize_b must be less than or equal to  outputCols")
  require((inputCols * outputCols) % VectorSize_w == 0, "VectorSize_w must divide inputCols * outputCols perfectly")
  require(( outputCols) % VectorSize_b == 0, "VectorSize_b must divide  outputCols perfectly") 
  require(VectorSize_in <= inputCols, "VectorSize_in must be less than or equal to  inputCols")
  require(( inputCols) % VectorSize_in == 0, "VectorSize_in must divide  outputCols perfectly") 
  require(VectorSize_out <= inputCols, "VectorSize_out must be less than or equal to  inputCols")
  require(( inputCols) % VectorSize_out == 0, "VectorSize_out must divide  outputCols perfectly")
  require(( num_rows <= inputRows), "num_rows must be less or equal to inputRows")  
  

  val io = IO(new Bundle {
    val input_x_elem  =  Vec(VectorSize_in, Flipped(Decoupled(UInt(dataWidth.W))))
    val weight_elems = Vec(numFilters, Vec(VectorSize_w, Flipped(Decoupled(UInt(dataWidth.W)))))
    val bias_elems =   Vec(numFilters, Vec(VectorSize_b, Flipped(Decoupled(UInt(dataWidth.W)))))
    val output_elems = Vec(numFilters,Vec(VectorSize_out, Decoupled(UInt(dataWidth.W))))
    val ISbias = Input(UInt(dataWidth.W))
      
  })
   
  val DEBUG = false
  // Define states
  val sIdle :: sLoadWeights :: sLoadBias :: sLoadInput :: sCompute :: sOutput :: sDone :: Nil = Enum(7)
  val state = RegInit(sIdle)

  //buffers
  val inputBuffer = Reg(Vec((inputCols)*num_rows , UInt(dataWidth.W))) //new 
    
  val weightRegs = Reg(Vec(numFilters, Vec(inputCols * outputCols, UInt(dataWidth.W))))
  val biasRegs = Reg(Vec(numFilters, Vec(outputCols, UInt(dataWidth.W))))
  val outputBuffers = Reg(Vec(numFilters, Vec(outputCols*num_rows, UInt(dataWidth.W)))) //new
   
  //counters  
  val inputColCount = RegInit(0.U(log2Ceil((1 + inputCols)*num_rows).W))
  val inputRowCount = RegInit(0.U(log2Ceil(1 + inputRows).W))
    
  val weightCounts = Reg(Vec(numFilters, UInt(log2Ceil(1 + inputCols * outputCols).W))) 
  val biasCounts = Reg(Vec(numFilters, UInt(log2Ceil(1 + outputCols).W)))
  val outputColCounts = Reg(Vec(numFilters, UInt(log2Ceil(1 + outputCols*num_rows).W)))  //new
  val VectorCounter = RegInit(0.U(log2Ceil(1 + VectorSize_w).W))
    
  val loadComplete = RegInit(VecInit(Seq.fill(numFilters)(0.U(1.W)))) 
  val StoreComplete = RegInit(VecInit(Seq.fill(numFilters)(0.U(1.W))))
  
  // Initialize counts
  weightCounts.foreach(_ := 0.U)
  biasCounts.foreach(_ := 0.U)
  outputColCounts.foreach(_ := 0.U)

  io.input_x_elem.foreach(_.ready := false.B)

  io.weight_elems.foreach { weightElem =>
    weightElem.foreach(_.ready := false.B)
  }

  io.bias_elems.foreach { biasElem =>
    biasElem.foreach(_.ready := false.B)
  }
    
  io.output_elems.foreach { outputElem =>
    outputElem.foreach(_.valid := false.B)
    outputElem.foreach(_.bits := 0.U)
  }
   

  switch(state) {
    is(sIdle) { 
      printf("State: sIdle\n")
      state := sLoadWeights
    }

is(sLoadWeights) {
  io.weight_elems.zip(weightCounts).foreach { case (weightElems, weightCount) =>
    for (i <- 0 until VectorSize_w) {
      weightElems(i).ready := (weightCount < (inputCols * outputCols).U)

      when(weightElems(i).valid && weightElems(i).ready  && (loadComplete(io.weight_elems.indexOf(weightElems)) === 0.U)) {
        val filterIdx = io.weight_elems.indexOf(weightElems)
        val weightPos = weightCount + i.U
        val weightValue = weightElems(i).bits

        // Store the weight value in the appropriate position in weightRegs
        weightRegs(filterIdx)(weightPos) := weightValue

        // Print debug information
        if(DEBUG) {
        printf(p"Filter $filterIdx: Storing weight at position $weightPos, value = $weightValue\n")
        }
        weightCounts(io.weight_elems.indexOf(weightElems)) := weightCounts(io.weight_elems.indexOf(weightElems)) + VectorSize_w.U
      }
    }

    // Mark completion when all weights are loaded
    when(weightCounts(io.weight_elems.indexOf(weightElems)) === (inputCols * outputCols).U -VectorSize_w.U) {
      loadComplete(io.weight_elems.indexOf(weightElems)) := 1.U
      if(DEBUG) {
      printf(p"Filter ${io.weight_elems.indexOf(weightElems)}: All weights loaded. Marking loadComplete.\n")
      }
    }
  }

  // Check if all weights are loaded
  when(loadComplete.forall(_ === 1.U)) {
    when(io.ISbias === 1.U) {
      if (DEBUG) {
        printf("All weights loaded for all filters. Transitioning to sLoadBias\n")
      }
      // Reset loadComplete and transition to bias loading state
      loadComplete := VecInit(Seq.fill(numFilters)(0.U(1.W)))
      state := sLoadBias
    }.otherwise {
      if (DEBUG) {
        printf("All weights loaded for all filters. Transitioning to sLoadInput\n")
      }
      // Transition to input loading state
      state := sLoadInput
    }
  }
}
    is(sLoadBias) { 
      // Set the ready signals for each bias element
      io.bias_elems.zip(biasCounts).foreach { case (biasElem, biasCount) =>
         for (i <- 0 until VectorSize_b) {
              biasElem(i).ready := (biasCount < outputCols.U)

            // Update biasRegs and biasCounts for each valid biasElem
            when(biasElem(i).valid && biasElem(i).ready && (loadComplete(io.bias_elems.indexOf(biasElem)) === 0.U)) {
              val filterIdx = io.bias_elems.indexOf(biasElem)
              val biastPos = biasCount+ i.U
              val biasValue = biasElem(i).bits

              biasRegs(filterIdx )(biastPos) := biasValue
              biasCounts(io.bias_elems.indexOf(biasElem)) := biasCount + VectorSize_b.U
            }
          }   
        when(biasCounts(io.bias_elems.indexOf(biasElem)) === (outputCols).U -VectorSize_b.U) {
           loadComplete(io.bias_elems.indexOf(biasElem)) := 1.U
            if(DEBUG) {
          printf(p"Filter ${io.bias_elems.indexOf(biasElem)}: bias loaded. Marking loadComplete.\n")
      }
        } 
    }
      // Check if all biases are loaded
      when(loadComplete.forall(_ === 1.U)) {
        if(DEBUG) {
        printf("All biases loaded. Transitioning to sLoadInput\n")}
        state := sLoadInput
      }
    }


    is(sLoadInput) { 

      for(i <-0 until VectorSize_in) {
      io.input_x_elem(i).ready := true.B 

        when(io.input_x_elem(i).valid && io.input_x_elem(i).ready) {
              val filterIdx = io.input_x_elem.indexOf(io.input_x_elem)
              val tPos = inputColCount + i.U
              val Value = io.input_x_elem(i).bits
        inputBuffer(tPos) :=  Value
        inputColCount := inputColCount + VectorSize_in.U
        if(DEBUG){ 
        printf("Loading input[%d]: %d\n", tPos, io.input_x_elem(i).bits)
          }
        }

      when(inputColCount >= (inputCols*num_rows).U -VectorSize_in.U ) {
        inputColCount := 0.U
        if(DEBUG){
        printf("%d Input rows loaded. Transitioning to sCompute\n",num_rows.U)}
        state := sCompute
      }
      }
    }

    is(sCompute) {
      // Compute the output for the current row
      for(r <- 0 until num_rows){
        for (j <- 0 until outputCols) {
          for (f <- 0 until numFilters) {
           
            val sum = (0 until inputCols).map { i =>
              inputBuffer(i +r*inputCols) * weightRegs(f)(i * outputCols + j) 
            }.reduce(_ + _)

            // Apply bias if ISbias is enabled
            outputBuffers(f)(j +r*outputCols) := Mux(io.ISbias === 1.U, sum + biasRegs(f)(j), sum)
          }
        } 
      }
     if(DEBUG){
      printf("Computing outputs for row %d\n", inputRowCount)}
      
      outputColCounts.foreach(_ := 0.U) 
      loadComplete.foreach(_ := 0.U) 
      state := sOutput 
     
     if(DEBUG){ 
      printf("Moving to Output \n")}
    }


is(sOutput) {
  // Valid signals for all output elements
 for(r <-0 until num_rows){ 
  for (f <- 0 until numFilters) { 
    
      // Mark StoreComplete when all columns are processed
    when(outputColCounts(f) >= (outputCols*num_rows).U -VectorSize_out.U) {
      StoreComplete(f) := 1.U 
      io.output_elems(f).foreach(_.valid := false.B)
    //  printf(p"Filter $f: Marking StoreComplete as 1\n") 
    }

    when(StoreComplete(f) === 0.U) { 
        outputColCounts(f) := outputColCounts(f) + VectorSize_out.U

        // Set the valid signal for output
        io.output_elems(f).foreach(_.valid := true.B)
        
        for (i <- 0 until VectorSize_out) { 
            val filterIdx = f 
            val Pos = outputColCounts(f) + i.U
            val Value = outputBuffers(f)(i) 
            io.output_elems(f)(i).bits := outputBuffers(f)(Pos)
       }
        
      }
    // Print the status of StoreComplete for each filter
   // printf(p"Filter $f: StoreComplete = ${StoreComplete(f)}, outputColCounts = ${outputColCounts(f)}\n")

  }
 }
  // Check if all filters have completed storing output
  when(StoreComplete.forall(_ === 1.U)) {
    // Reset output column counts and loadComplete
    outputColCounts.foreach(_ := 0.U)
    StoreComplete := VecInit(Seq.fill(numFilters)(0.U(1.W)))

    // Increment the input row count
    inputRowCount := inputRowCount + num_rows.U

    // Print the row completion status
    if(DEBUG){
    printf(" %d Output rows completed.\n", num_rows.U) }

    // Transition to next state based on input row count
    when(inputRowCount === inputRows.U) {
      state := sDone
      printf("All input rows processed. Transitioning to sDone\n")
    } .otherwise { 
      if(DEBUG){
      printf("Transitioning to sLoadInput for next %d rows\n",num_rows.U)}
      state := sLoadInput
    }
  }
}



    is(sDone) {
      io.output_elems.foreach { outputElem =>
    outputElem.foreach(_.valid := false.B)
  }
      printf("State: sDone. FSM completed processing.\n")
    }

  }
}


// Software reference model for the linear filter
object SoftwareLinearFilter {
  def apply(inputs: Seq[Seq[Int]], weights: Seq[Int], biases: Seq[Int], inputCols: Int, outputCols: Int): Seq[Seq[Int]] = {
    inputs.map { input =>
      val output = Array.fill(outputCols)(0)
      for (j <- 0 until outputCols) {
        output(j) = biases(j)
        for (i <- 0 until inputCols) {
          output(j) += input(i) * weights(i * outputCols + j)
        }
      }
      output.toSeq
    }
  }
}

import chisel3._
import chisel3.iotesters.{PeekPokeTester, Driver}
import scala.util.Random

// Define the testbench as a PeekPokeTester
class LinearFilter_fsmTester(c: MultiLinearFilter_fsm) extends PeekPokeTester(c) {
  val inputCols = c.inputCols
  val outputCols = c.outputCols
  val inputRows = c.inputRows
  val numFilters = c.numFilters
  val dataWidth = c.dataWidth
  val VectorSize_w = c.VectorSize_w
  val VectorSize_b = c.VectorSize_b
  val VectorSize_in = c.VectorSize_in 
  val VectorSize_out = c.VectorSize_out
  val num_rows = c.num_rows

  val DEBUG = false// Set to false to disable debug prints
  val ISBIAS = true // Have bias or not

  def initBiases(isBias: Boolean, numFilters: Int, inputCols: Int, outputCols: Int): Seq[Seq[Int]] = {
    if (isBias) {
      Seq.fill(numFilters)(Seq.fill(outputCols)(Random.nextInt(256)))
    } else {
      Seq.fill(numFilters)(Seq.fill(outputCols)(0))
    }
  }

  // Generate random inputs, weights, and biases for each filter
  val inputs = Seq.fill(inputRows, inputCols)(Random.nextInt(256))
  val weights = Seq.fill(numFilters)(Seq.fill(inputCols * outputCols)(Random.nextInt(256)))
  val biases = initBiases(ISBIAS, numFilters, inputCols, outputCols)

  // Print inputs and weights for debugging
  if (DEBUG) {
    println("Inputs:")
    inputs.zipWithIndex.foreach { case (row, idx) =>
      println(s"Row $idx: ${row.mkString(", ")}")
    }

    for (filterIdx <- 0 until numFilters) {
      println(s"Weights for Filter $filterIdx:")
      weights(filterIdx).grouped(outputCols).foreach { row =>
        println(row.mkString(", "))
      }
    }

    println("Biases:")
    biases.foreach(b => println(b.mkString(", ")))
  }

  // Software reference output
  val expectedOutput = weights.zip(biases).map { case (w, b) =>
    SoftwareLinearFilter(inputs, w, b, inputCols, outputCols)
  }

  if (DEBUG) {
    // Print the expected values for each filter
    expectedOutput.zipWithIndex.foreach { case (filterOutput, filterIdx) =>
      println(s"Expected result for Filter $filterIdx:")
      filterOutput.zipWithIndex.foreach { case (row, rowIdx) =>
        println(s"Row $rowIdx: ${row.mkString(", ")}")
      }
      println("\n") // New line for clarity between filters
    }
  }

  // Load ISbias flag
  poke(c.io.ISbias, ISBIAS)

def pokeWeightElems(filterIdx: Int, weights: Seq[Int],VectorSize_w: Int): Unit = {
  // Number of chunks needed to process all weights
  val numChunks = (weights.length + VectorSize_w - 1) / VectorSize_w
  
  for (chunkIdx <- 0 until numChunks) {
    val startIdx = chunkIdx * VectorSize_w
    val endIdx = Math.min(startIdx + VectorSize_w, weights.length)
    val weightChunk = weights.slice(startIdx, endIdx)

    // Set valid signal for the weight elements
    for ((weight, i) <- weightChunk.zipWithIndex) {
      poke(c.io.weight_elems(filterIdx)(i).valid, true.B)
      poke(c.io.weight_elems(filterIdx)(i).bits, weight)
    }

    // Ensure that all weight elements are ready before proceeding
    // Check readiness of each weight element
    while (!weightChunk.indices.forall(i => peek(c.io.weight_elems(filterIdx)(i).ready) == 1)) {
      step(1)
    }

    // Clear valid signal after sending a chunk
    // for (i <- weightChunk.indices) {
    //   poke(c.io.weight_elems(filterIdx)(i).valid, false.B)
    // }
    
    step(1) // Move to the next clock cycle
  }
}


  // Load weights in chunks
  for (filterIdx <- 0 until numFilters) {
    pokeWeightElems(filterIdx, weights(filterIdx),VectorSize_w)
  }


    // Load biases if ISBIAS is true
if (ISBIAS) {
  for (filterIdx <- 0 until numFilters) {
    // Calculate how many chunks are needed to send all biases
    val numBiasChunks = (biases(filterIdx).length + VectorSize_b - 1) / VectorSize_b

    for (chunkIdx <- 0 until numBiasChunks) {
      val startIdx = chunkIdx * VectorSize_b
      val endIdx = Math.min(startIdx + VectorSize_b, biases(filterIdx).length)
      val biasChunk = biases(filterIdx).slice(startIdx, endIdx)

      // Set the valid signal for the chunk
      for ((bias, i) <- biasChunk.zipWithIndex) {
        poke(c.io.bias_elems(filterIdx)(i).valid, true.B)
        poke(c.io.bias_elems(filterIdx)(i).bits, bias)
      }

      // Wait until all bias elements are ready before proceeding
      while (!biasChunk.indices.forall(i => peek(c.io.bias_elems(filterIdx)(i).ready) == 1)) {
        step(1)
      }

      // Step to the next clock cycle after sending a chunk
      step(1)

      // Clear the valid signal after sending the chunk
      // for (i <- biasChunk.indices) {
      //   poke(c.io.bias_elems(filterIdx)(i).valid, false.B)
      // }
    }
  }
}

 // Process inputs row by row
  for (rowIdx <- 0 until (inputRows/num_rows) by num_rows ) {
    // Send input row in chunks of VectorSize_in elements
   for(r<- 0 until num_rows) {
    for (inputIdx <- 0 until inputCols by VectorSize_in) {
      for (i <- 0 until VectorSize_in) {
        if (inputIdx + i < inputCols) {
          poke(c.io.input_x_elem(i).bits, inputs(rowIdx + r)(inputIdx + i))
          poke(c.io.input_x_elem(i).valid, true.B)
        } else {
          poke(c.io.input_x_elem(i).bits, 0.U) // Padding if necessary
          poke(c.io.input_x_elem(i).valid, false.B)
        }
      }
       

    // Wait for all the ready signals to be asserted before moving to the next clock cycle
    while (!c.io.input_x_elem.map((elem: DecoupledIO[UInt]) => peek(elem.ready) == 1).reduce(_ && _)) {
      step(1)
    }


      step(1) // Move to next clock cycle
    }
  
    // After processing the inputs, set valid to false for all elements
    for (i <- 0 until VectorSize_in) {
      poke(c.io.input_x_elem(i).valid, false.B)
    }
} 

// Initialize an array of sequences to store the outputs for each filter
var outputs = Array.fill(numFilters)(Seq[Int]())

// Run the loop until all outputs are collected for each filter
while (outputs(0).length < outputCols) {
  // Check if the first filter has valid output (all filters should have valid outputs simultaneously)
  if (peek(c.io.output_elems(0)(0).valid) == 1) {
    // Iterate over each filter
    for (filterIdx <- 0 until numFilters) {
      // Iterate over the number of outputs per cycle (VectorSize_out)
      for (vecIdx <- 0 until VectorSize_out) {
        // Only collect if we're still within the bounds of the expected output size
        if (outputs(filterIdx).length < outputCols) {
          // Get the output value for the current filter and vector index
          val out = peek(c.io.output_elems(filterIdx)(vecIdx).bits).toInt
          // Append the output to the respective filter's sequence
          outputs(filterIdx) = outputs(filterIdx) :+ out

          if (DEBUG) {
            println(s"Output for Filter $filterIdx (Row $rowIdx, Col ${outputs(filterIdx).length - 1}): $out")
          }
        }
      }
    }
    // Step forward in the simulation
    step(1)
  } else {
    // If not valid, step forward and wait
    step(1)
  }
}

// Compare actual outputs with expected outputs for all filters
for (filterIdx <- 0 until numFilters) {
  for (i <- 0 until outputCols) {
    // Check if the collected output matches the expected output
    expect(
      outputs(filterIdx)(i) == expectedOutput(filterIdx)(rowIdx)(i),
      s"Output mismatch at index $i for row $rowIdx (Filter $filterIdx): " +
      s"expected ${expectedOutput(filterIdx)(rowIdx)(i)}, got ${outputs(filterIdx)(i)}"
    )
  }
}


/////////////
    if (DEBUG) {
      // Print actual and expected outputs for debugging
      for (filterIdx <- 0 until numFilters) {
        println(s"Actual Output for Filter $filterIdx:")
        outputs(filterIdx).zipWithIndex.foreach { case (out, i) =>
          println(s"Col $i: $out")
        }
        println(s"Expected Output for Filter $filterIdx:")
        expectedOutput(filterIdx)(rowIdx).zipWithIndex.foreach { case (exp, i) =>
          println(s"Col $i: $exp")
        }
        println("\n")
      }
    }
  } 

}

object LinearFilter_fsmTest {
  def main(args: Array[String]): Unit = {
    val inputRows = 10
    val inputCols = 10
    val outputCols = 10
    val numFilters = 3

    val dataWidth = 32

    val VectorSize_w = 10 //4->16003 cycles, 10->1333 cycles
    val VectorSize_b = 5
    val VectorSize_in = 10
    val VectorSize_out = 10

    val num_rows = 1 // 10->194 cycles  1->257 cycles 

    // Run the test for MultiLinearFilter_fsm
    val result = Driver.execute(Array("--generate-vcd-output", "on"), () =>
      new MultiLinearFilter_fsm(inputRows, inputCols, outputCols, dataWidth, numFilters, VectorSize_w,VectorSize_b,VectorSize_in,VectorSize_out, num_rows)
    ) {
      c => new LinearFilter_fsmTester(c)
    }

    // Exit with an error code if the test fails
    if (!result) System.exit(1)
  }
}
