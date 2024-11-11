package linear_tranform

import chisel3._
import chisel3.util._
import chisel3.iotesters.{PeekPokeTester, Driver}
import scala.util.Random

/* 
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
 */

 /*
class MultiLinearFilter_fsm( val inputRows: Int, val inputCols: Int, val outputCols: Int, val dataWidth: Int,val numFilters: Int,val VectorSize_w:Int,
                                                                     val VectorSize_b :Int, val VectorSize_in :Int , val VectorSize_out :Int,val num_rows :Int) extends Module {
  
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
    //data 
    val input_x_elem  =  Vec(VectorSize_in, Flipped(Decoupled(UInt(dataWidth.W))))
    val weight_elems = Vec(numFilters, Vec(VectorSize_w, Flipped(Decoupled(UInt(dataWidth.W)))))
    val bias_elems =   Vec(numFilters, Vec(VectorSize_b, Flipped(Decoupled(UInt(dataWidth.W)))))
    val output_elems = Vec(numFilters,Vec(VectorSize_out, Decoupled(UInt(dataWidth.W))))
    
    //control signals
    val ISbias = Input(Bool())
    val stateInput = Input(UInt(4.W)) 

    //busy signal interface 
    val busy = Output(Bool())

    //Status signals for complete 
    val loadWeightsDone  = Output(Bool())
    val loadBiasesDone   = Output(Bool())
    val loadInputsDone   = Output(Bool())
    val computeDone      = Output(Bool())
    val storeOutputsDone = Output(Bool())
    val Done =  Output(Bool())
  })
   
  //status register intialize
  val loadWeightsDoneReg  = RegInit(false.B)
  val loadBiasesDoneReg   = RegInit(false.B)
  val loadInputsDoneReg   = RegInit(false.B)
  val computeDoneReg      = RegInit(false.B)
  val storeOutputsDoneReg = RegInit(false.B)
  val doneReg             = RegInit(false.B)
  val stateInputReg       = RegInit(0.U) 

  //connect status registers with status output/input signals 
  io.loadWeightsDone  := loadWeightsDoneReg
  io.loadBiasesDone   := loadBiasesDoneReg
  io.loadInputsDone   := loadInputsDoneReg
  io.computeDone      := computeDoneReg
  io.storeOutputsDone := storeOutputsDoneReg
  io.Done             := doneReg
  stateInputReg       := io.stateInput  

  //busy signal 
  val busyInternal = Wire(Bool())
  busyInternal := false.B  

  //buffers
  val inputBuffer = Reg(Vec((inputCols)*num_rows , UInt(dataWidth.W))) 
  val weightRegs = Reg(Vec(numFilters, Vec(inputCols * outputCols, UInt(dataWidth.W))))
  val biasRegs = Reg(Vec(numFilters, Vec(outputCols, UInt(dataWidth.W))))
  val outputBuffers = Reg(Vec(numFilters, Vec(outputCols*num_rows, UInt(dataWidth.W)))) 
  val ISbiasStored = RegInit(false.B) 
   
  //counters  
  val inputColCount = RegInit(0.U(log2Ceil((1 + inputCols)*num_rows).W))
  val inputRowCount = RegInit(0.U(log2Ceil(1 + inputRows).W)) 
  val weightCounts = Reg(Vec(numFilters, UInt(log2Ceil(1 + inputCols * outputCols).W))) 
  val biasCounts = Reg(Vec(numFilters, UInt(log2Ceil(1 + outputCols).W)))
  val outputColCounts = Reg(Vec(numFilters, UInt(log2Ceil(1 + outputCols*num_rows).W))) 
  val VectorCounter = RegInit(0.U(log2Ceil(1 + VectorSize_w).W))

  //Control signals   
  val loadComplete = RegInit(VecInit(Seq.fill(numFilters)(0.U(1.W)))) 
  val StoreComplete = RegInit(VecInit(Seq.fill(numFilters)(0.U(1.W))))
  
  // Initialize counts
  weightCounts.foreach(_ := 0.U)
  biasCounts.foreach(_ := 0.U)
  outputColCounts.foreach(_ := 0.U)
  
  //initialize input signals
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
   

    when(stateInputReg  === 0.U) {
      ISbiasStored := io.ISbias // Store the input ISbias
    }

  when(stateInputReg === 1.U && ! loadWeightsDoneReg)  { //load weights
      busyInternal := true.B 
      io.weight_elems.zip(weightCounts).foreach { case (weightElems, weightCount) =>
        for (i <- 0 until VectorSize_w) {
            weightElems(i).ready := (weightCount < (inputCols * outputCols).U)

            when(weightElems(i).valid && weightElems(i).ready  && (loadComplete(io.weight_elems.indexOf(weightElems)) === 0.U)) {
              val filterIdx = io.weight_elems.indexOf(weightElems)
              val weightPos = weightCount + i.U
              val weightValue = weightElems(i).bits
              weightRegs(filterIdx)(weightPos) := weightValue
              weightCounts(io.weight_elems.indexOf(weightElems)) := weightCounts(io.weight_elems.indexOf(weightElems)) + VectorSize_w.U
            }
        }

        when(weightCounts(io.weight_elems.indexOf(weightElems)) === (inputCols * outputCols).U -VectorSize_w.U) { // Mark completion when all weights are loaded
          loadComplete(io.weight_elems.indexOf(weightElems)) := 1.U
        }
    }

    when(loadComplete.forall(_ === 1.U)) {  // Check if all weights are loaded
        loadComplete := VecInit(Seq.fill(numFilters)(0.U(1.W)))
        loadWeightsDoneReg := true.B
        busyInternal := false.B 
    } 
  }

    when(stateInputReg === 1.U && loadWeightsDoneReg && !loadBiasesDoneReg && ISbiasStored)  { //load Bias 
      busyInternal := true.B 
      io.bias_elems.zip(biasCounts).foreach { case (biasElem, biasCount) =>
         for (i <- 0 until VectorSize_b) {
              biasElem(i).ready := (biasCount < outputCols.U)

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
        } 
    }
      when(loadComplete.forall(_ === 1.U)) {  // Check if all biases are loaded
        loadBiasesDoneReg := true.B
        busyInternal := false.B 
      }
    
  } 

    when(stateInputReg === 2.U &&  !loadInputsDoneReg) { //load input 
      busyInternal := true.B 
      for(i <-0 until VectorSize_in) {
        io.input_x_elem(i).ready := true.B 

          when(io.input_x_elem(i).valid && io.input_x_elem(i).ready) {
                val filterIdx = io.input_x_elem.indexOf(io.input_x_elem)
                val tPos = inputColCount + i.U
                val Value = io.input_x_elem(i).bits
          inputBuffer(tPos) :=  Value
          inputColCount := inputColCount + VectorSize_in.U
          }

        when(inputColCount >= (inputCols*num_rows).U -VectorSize_in.U ) {
          inputColCount := 0.U
          loadInputsDoneReg := true.B
        }
      }
  } 

    when(stateInputReg === 2.U && loadInputsDoneReg) { //compute output 
      busyInternal := true.B 
      for(r <- 0 until num_rows){
        for (j <- 0 until outputCols) {
          for (f <- 0 until numFilters) {
           
            val sum = (0 until inputCols).map { i =>
              inputBuffer(i +r*inputCols) * weightRegs(f)(i * outputCols + j) 
            }.reduce(_ + _)

            outputBuffers(f)(j +r*outputCols) := Mux(ISbiasStored, sum + biasRegs(f)(j), sum)
          }
        } 
      }

      outputColCounts.foreach(_ := 0.U) 
      loadComplete.foreach(_ := 0.U)
      computeDoneReg := true.B
      busyInternal := false.B 
  } 


    when(stateInputReg === 3.U && !storeOutputsDoneReg) { //store output 
      busyInternal := true.B 
      for(r <-0 until num_rows){ 
        for (f <- 0 until numFilters) { 
          when(outputColCounts(f) >= (outputCols*num_rows).U -VectorSize_out.U) {
            StoreComplete(f) := 1.U 
            io.output_elems(f).foreach(_.valid := false.B)
          }

          when(StoreComplete(f) === 0.U) { 
              outputColCounts(f) := outputColCounts(f) + VectorSize_out.U
              io.output_elems(f).foreach(_.valid := true.B)
              
              for (i <- 0 until VectorSize_out) { 
                  val filterIdx = f 
                  val Pos = outputColCounts(f) + i.U
                  val Value = outputBuffers(f)(i) 
                  io.output_elems(f)(i).bits := outputBuffers(f)(Pos)
              }
            }
        }
      }
   
      when(StoreComplete.forall(_ === 1.U)) { // Check if all filters have completed storing output 
          outputColCounts.foreach(_ := 0.U)
          StoreComplete := VecInit(Seq.fill(numFilters)(0.U(1.W)))
          inputRowCount := inputRowCount + num_rows.U 
          busyInternal := false.B 

          when(inputRowCount === inputRows.U) {
             doneReg := true.B
          } .otherwise { 
              loadInputsDoneReg := false.B
              computeDoneReg := false.B
          }
        }
  }

  io.busy := busyInternal
}

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

  //register 
  val ISbiasStored = RegInit(0.U) 
  //ISbiasStored := io.ISbias // Store the input ISbias

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
    when(ISbiasStored  === 1.U) {
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
            outputBuffers(f)(j +r*outputCols) := Mux( ISbiasStored === 1.U, sum + biasRegs(f)(j), sum)
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
   // outputElem.foreach(_.valid := false.B)
  }
      printf("State: sDone. FSM completed processing.\n")
    }

  }
}
