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
   
  // Initialize completion signals
  io.loadWeightsDone  := false.B
  io.loadBiasesDone   := false.B
  io.loadInputsDone   := false.B
  io.computeDone      := false.B
  io.storeOutputsDone := false.B
  io.Done := false.B

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
   

    when(io.currentState  === 0.U) {
      ISbiasStored := io.ISbias // Store the input ISbias
    }

  when(io.currentState === 1.U && !io.loadWeightsDone)  { //load weights
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
        io.loadWeightsDone := true.B
        busyInternal := false.B 
    } 
  }

    when(io.currentState === 1.U && io.loadWeightsDone &&  !io.loadBiasesDone && io.ISbias)  { //load Bias 
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
      io.loadWeightsDone = true.B
      busyInternal := false.B 
      }
    
  } 

    when(io.currentState === 2.U &&  !io.loadInputsDone) { //load input 
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
          io.loadInputsDone := true.B
        }
      }
  } 

    when(io.currentState === 2.U &&  io.loadInputsDone) { //compute output 
      busyInternal := true.B 
      for(r <- 0 until num_rows){
        for (j <- 0 until outputCols) {
          for (f <- 0 until numFilters) {
           
            val sum = (0 until inputCols).map { i =>
              inputBuffer(i +r*inputCols) * weightRegs(f)(i * outputCols + j) 
            }.reduce(_ + _)

            outputBuffers(f)(j +r*outputCols) := Mux(io.ISbias, sum + biasRegs(f)(j), sum)
          }
        } 
      }

      outputColCounts.foreach(_ := 0.U) 
      loadComplete.foreach(_ := 0.U)
      io.computeDone := true.B
      busyInternal := false.B 
  } 


    when(io.currentState === 3.U &&  !storeOutputsDone) { //store output 
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
             io.Done = true.B
          } .otherwise { 
              io.loadInputsDone := false.B
              io.computeDone := false.B
          }
        }
  }

  io.busy := busyInternal
}
