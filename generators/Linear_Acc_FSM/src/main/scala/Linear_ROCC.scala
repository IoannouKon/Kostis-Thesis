package linear_tranform

import chisel3._
import chisel3.util._

import freechips.rocketchip.tile._
import org.chipsalliance.cde.config._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tile.{BuildRoCC, OpcodeSet}
import freechips.rocketchip.diplomacy.LazyModule
import os.truncate
import firrtl.PrimOps.Add

class LinearFilterExample(  
    opcodes: OpcodeSet,
    val params: LinearFilterParams  
)(implicit p: Parameters) extends LazyRoCC(
    opcodes = opcodes,
   // nPTWPorts = if (p(LinearFilterTLB).isDefined) 1 else 0  // Adjust based on your configuration
) {
  override lazy val module = new LinearFilterExampleModuleImpl(this)
  
}

// class LinearFilterExampleModuleImpl(outer: LinearFilterExample)(implicit p: Parameters) extends LazyRoCCModuleImp(outer) 
// with HasCoreParameters {

//     val params: LinearFilterParams = outer.params
//     val linear = Module(new MultiLinearFilter_fsm(
//         params.inputRows,
//         params.inputCols,
//         params.outputCols,
//         params.dataWidth,
//         params.numFilters,
//         params.VectorSize_w,
//         params.VectorSize_b,
//         params.VectorSize_in,
//         params.VectorSize_out,
//         params.num_rows
//     ))

//     //intialize input signals 
//     linear.io.input_x_elem.foreach(_.valid := false.B)
//     linear.io.input_x_elem.foreach(_.bits := 0.U)
//     linear.io.weight_elems.foreach { weightElem =>
//         weightElem.foreach(_.valid := false.B)
//         weightElem.foreach(_.bits := 0.U)
//     }
//     linear.io.bias_elems.foreach { biasElem =>
//         biasElem.foreach(_.valid := false.B)
//         biasElem.foreach(_.bits := 0.U)
//     }
//     linear.io.output_elems.foreach { outputElem =>
//         outputElem.foreach(_.ready := false.B)
//     }
//     linear.io.ISbias := false.B
//     linear.io.stateInput := 0.U 

//     //     //RISCV --> RoCC  
//         val cmd = Queue(io.cmd) 
//         val funct = cmd.bits.inst.funct
//         val rs1 = cmd.bits.rs1
//      // val ISBIAS = cmd.bits.rs2 
        
//         //bias 
//         val ISBIAS = RegInit(false.B)
        
//         //states 
//         val set_bias           =    (funct === 0.U)   
//         val load_weights       =    (funct === 1.U && !linear.io.loadWeightsDone) 
//         val load_bias          =    (funct === 1.U &&  linear.io.loadWeightsDone && !linear.io.loadBiasesDone && ISBIAS)
//         val load_input_rows    =    (funct === 2.U && !linear.io.loadInputsDone)
//         val compute_ouput_rows =    (funct === 2.U &&  linear.io.loadInputsDone)
//         val store_output_rows  =    (funct === 3.U && !linear.io.storeOutputsDone)
//         val read_busy          =    (funct === 4.U)  

//         // Using to use load Interleaved Sequential as load Sequential  
//         //load weight_filter_0 chunk[0] .... weight_filter_0 chunk[n] and after for next filter          
//         var filterId_weights = 0
//         var counter_weights  = 0
//         var filterId_bias    = 0
//         var counter_bias     = 0

//          when(cmd.valid && cmd.ready) {  
//             linear.io.stateInput := funct 

//             when(set_bias) { 
//                 linear.io.ISbias := rs1 
//                 ISBIAS := true.B
//             }.elsewhen(load_weights) { 
                
//                 when(counter_weights.U < (params.inputCols * params.outputCols).U) { 
//                     // Load weights for the current filter
//                     for (j <- 0 until params.VectorSize_w) {
//                     linear.io.weight_elems(filterId_weights)(j).bits := rs1 + (filterId_weights.U * params.VectorSize_w.U + j.U)
//                     linear.io.weight_elems(filterId_weights)(j).valid := true.B
//                     }
//                     counter_weights += params.VectorSize_w
//                 } 
                
//                 when(counter_weights.U === (params.inputCols * params.outputCols).U ) {  //- params.VectorSize_w.
//                     counter_weights = 0 
//                     filterId_weights = filterId_weights + 1 
//                 }

//             }.elsewhen(load_bias) {
                
//                 when(counter_bias.U < (params.outputCols).U) { 
//                     // Load weights for the current filter
//                     for (j <- 0 until params.VectorSize_b) {
//                     linear.io.bias_elems(filterId_bias)(j).bits := rs1 + (filterId_bias.U * params.VectorSize_b.U + j.U)
//                     linear.io.bias_elems(filterId_bias)(j).valid := true.B
//                     }
//                     counter_bias += params.VectorSize_b
//                 }
//                 when(counter_bias.U === (params.outputCols).U) {  //- params.VectorSize_b.
//                     counter_bias = 0 
//                     filterId_bias = filterId_bias + 1 
//                 }

//             }.elsewhen(load_input_rows) {
//                 for (j <- 0 until params.VectorSize_in) {
//                     linear.io.input_x_elem(j).bits := rs1 + j.U 
//                     linear.io.input_x_elem(j).valid := true.B  
//                 }
//             }.elsewhen(store_output_rows) {
//                 printf("HELLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLO")
//                 for (i <- 0 until params.numFilters) {
//                     for (j <- 0 until params.VectorSize_out) { 
//                         io.resp.bits.data :=  34.U //linear.io.output_elems(i)(j).bits // Data to store at memory address
//                         io.resp.valid := linear.io.output_elems(i)(j).valid 
//                         io.resp.bits.rd := cmd.bits.inst.rd  + (i.U * params.VectorSize_out.U + j.U)// Send data to memory at rd + offset
//                     }  
//                 }
//             }
//     } 
           

//         // Control logic
//         val doResp = cmd.bits.inst.xd
//         val stallReg = 0.U
//         val stallResp = doResp && !io.resp.ready
//         cmd.ready := !stallReg && !stallResp

//         // Processor response interface
//         //io.resp.valid := cmd.valid && doResp && !stallReg
//         //io.resp.bits.rd := cmd.bits.inst.rd
//         //io.resp.bits.data := result
//         io.busy := cmd.valid
//         io.interrupt := false.B

// }

import freechips.rocketchip.rocket._

// class LinearFilterExampleModuleImpl(outer: LinearFilterExample)(implicit p: Parameters) extends LazyRoCCModuleImp(outer) 
// with HasCoreParameters {

//         val params: LinearFilterParams = outer.params
//         val linear = Module(new MultiLinearFilter_fsm(
//             params.inputRows,
//             params.inputCols,
//             params.outputCols,
//             params.dataWidth,
//             params.numFilters,
//             params.VectorSize_w,
//             params.VectorSize_b,
//             params.VectorSize_in,
//             params.VectorSize_out,
//             params.num_rows
//         ))

//         val dataSizeBytes = (params.dataWidth / 8).U 

//         //intialize input signals 
//         linear.io.input_x_elem.foreach(_.valid := false.B)
//         linear.io.input_x_elem.foreach(_.bits := 0.U)
//         linear.io.weight_elems.foreach { weightElem =>
//             weightElem.foreach(_.valid := false.B)
//             weightElem.foreach(_.bits := 0.U)
//         }
//         linear.io.bias_elems.foreach { biasElem =>
//             biasElem.foreach(_.valid := false.B)
//             biasElem.foreach(_.bits := 0.U)
//         }
//         linear.io.output_elems.foreach { outputElem =>
//             outputElem.foreach(_.ready := false.B) 
//         }
//         linear.io.ISbias := 0.U

//         //RISCV --> RoCC  
//         val cmd   = Queue(io.cmd) 
//         val funct = cmd.bits.inst.funct
//         val rs1   = cmd.bits.rs1
//         val rs2   = cmd.bits.rs2 
//         // val memRespTag = io.mem.resp.bits.tag(log2Up(outer_n)-1,0)

//         //Registers
//         val ISBIAS = RegInit(false.B) 
//         val state  = RegInit(0.U) // 1-> load weghts 2->load bias 3->load row inputs 4->store row outputs 
//         val Addr   = RegInit(0.U)   
//         val busy   = RegInit(false.B)
        
//         //states 
//         val set_bias           =    (state === 0.U)   
//         val load_weights       =    (state === 1.U)
//         val load_bias          =    (state === 2.U)
//         val load_input_rows    =    (state === 3.U)
//         val store_output_rows  =    (state === 4.U) 

//         // Using to use load Interleaved Sequential as load Sequential  
//         //load weight_filter_0 chunk[0] .... weight_filter_0 chunk[n] and after for next filter          

//         //counters 
//         val count_w          =  RegInit(0.U)
//         val filterId_weights =  RegInit(0.U)
//         val count_b          =  RegInit(0.U)
//         val filterId_bias    =  RegInit(0.U)
//         val count_in         =  RegInit(0.U)
//         val count_out        =  RegInit(0.U)
//         val filterId_out     =  RegInit(0.U) 


//         when(filterId_weights === params.numFilters.U ) { 
//             state := 0.U
//         }
//         when(filterId_bias === params.numFilters.U) { 
//             state := 0.U
//         }
//         when(filterId_out === params.numFilters.U ) { 
//             state := 0.U
//         }

//         when(cmd.valid && cmd.ready) {   //cmd.fire() 
//             switch(funct) {
//                 // Case for setting bias
//                 is(0.U) { 
//                    linear.io.ISbias := rs1 
//                    ISBIAS := !ISBIAS
//                 }
//                 // Case for loading weights
//                 is(1.U) { 
//                     state := 1.U 
//                     Addr := rs1  //store base addr 
//                 }
//                 // Case for loading bias
//                 is(2.U) { 
//                     state := 2.U
//                     Addr := rs1  //store base addr         
//                 }
//                 // Case for loading input rows
//                 is(3.U) { 
//                     state := 3.U  
//                     Addr := rs1  //store base addr    
//                 }
//                 // Case for storing output rows
//                 is(4.U) {
//                     state := 4.U
//                     Addr := rs1  //store base addr 
//                     linear.io.output_elems(filterId_out)(count_out).ready := true.B 
//                 }
//             }
//         } 



//         when (io.mem.resp.valid) { // Memory to Accelerator 

//             //TODO out of order 
//             //regfile(memRespTag) := io.mem.resp.bits.data
//             //busy(memRespTag) := false.B

//             when(load_weights) { 
//                 linear.io.weight_elems(filterId_weights)(count_w).bits := io.mem.resp.bits.data 
//                 linear.io.weight_elems(filterId_weights)(count_w).valid := true.B
                
//                 count_w := count_w + 1.U 
//                 Addr := Addr + dataSizeBytes
//                 busy := false.B
//                 when( count_w >= (params.inputCols * params.outputCols).U -1.U ) { 
//                     count_w := 0.U 
//                     filterId_weights := filterId_weights + 1.U
//                 }
//             }    

//             when(load_bias) { 
//                 linear.io.bias_elems(filterId_bias)(count_b).bits := io.mem.resp.bits.data 
//                 linear.io.bias_elems(filterId_bias)(count_b).valid := true.B
                        
//                 count_b := count_b + 1.U
//                 Addr := Addr + dataSizeBytes
//                 busy := false.B
//                 when( count_b >= (params.outputCols).U -1.U ) { 
//                     count_b := 0.U 
//                     filterId_bias := filterId_bias + 1.U
//                 }
//             }

//             when(load_input_rows) { 
//                 linear.io.input_x_elem(count_in).bits := io.mem.resp.bits.data 
//                 linear.io.input_x_elem(count_in).valid := true.B   

//                 count_in:= count_in + 1.U
//                 Addr := Addr + dataSizeBytes
//                 busy := false.B
//                 when(count_in >= (params.inputCols*params.num_rows).U -1.U ) { 
//                      count_in := 0.U 
//                 }
//             }

//             when(store_output_rows){ 
                 
//                 linear.io.output_elems(filterId_out)(count_out).ready := false.B 

//                 count_out := count_out + 1.U
//                 Addr := Addr + dataSizeBytes
//                 busy := false.B
//                 when( (count_out >= (params.outputCols*params.num_rows).U -1.U) && io.resp.valid  ) { 
//                     count_out := 0.U 
//                     filterId_out := filterId_out + 1.U
//                 }

//                 linear.io.output_elems(filterId_out)(count_out).ready := true.B 
//             }
//         }

//         // control (Accelreator to Memory)
//         when (io.mem.req.valid && io.mem.req.ready) { //TODO
//           //  busy(addr) := true.B
//            busy := true.B
//         }
       
//         //memory help val 
//         val doLoad      = load_bias || load_weights || load_input_rows
//         val doStore     = store_output_rows && linear.io.output_elems(filterId_out)(count_out).valid 
//         val perform     = Mux(doLoad,M_XRD,M_XWR) 
//         val store_val   = Mux(doStore,linear.io.output_elems(filterId_out)(count_out).bits,17.U) 

//         //stall help val 
//         val doResp     = cmd.bits.inst.xd
//         val stallLoad  = doLoad  && !io.mem.req.ready
//         val stallStore = doStore && !io.mem.req.ready
//         val stallResp  = doResp  && !io.resp.ready
//         val stallReg   = busy //busy(addr)

//         cmd.ready := !stallReg && !stallLoad && !stallStore && !stallResp

//         //PROC responce interface 
//         io.resp.valid      := cmd.valid && doResp && !stallReg && !stallLoad && !stallStore 
//         io.resp.bits.rd    := cmd.bits.inst.rd
//         io.resp.bits.data  := state //??? debug value  (return  currrent the state)
//         io.busy            := cmd.valid || busy //busy.reduce(_||_)
//         io.interrupt       := false.B

//         //Memory Requeset Interface (Accelreator to Memory)
//         io.mem.req.valid       := true.B //cmd.valid && (doLoad || doStore) && !stallReg && !stallResp
//         io.mem.req.bits.addr   := Addr  
//         io.mem.req.bits.tag    := 0.U //cmd.bits.rs2(log2Up(outer_n)-1,0)  TODO 
//         io.mem.req.bits.cmd    := perform 
//         io.mem.req.bits.size   := params.dataWidth.U //log2Ceil(8).U // ???
//         io.mem.req.bits.signed := false.B
//         io.mem.req.bits.data   := 7.U //store_val 
//         io.mem.req.bits.phys   := false.B
//         io.mem.req.bits.dprv   := cmd.bits.status.dprv
//         io.mem.req.bits.dv     := cmd.bits.status.dv
//  }

class LinearFilterExampleModuleImpl(outer: LinearFilterExample)(implicit p: Parameters) extends LazyRoCCModuleImp(outer) 
  with HasCoreParameters {
  
  val outer_n     = 4  //here actually a parameter  (n+1) for array with n elemts!
  val regfile     = Mem(outer_n, UInt(xLen.W))
  val busy        = RegInit(VecInit(Seq.fill(outer_n) { false.B }))
  val loadCount   = RegInit(0.U(log2Ceil(outer_n).W)) // Track how many loads to perform based on outer_n
  val start_load  = RegInit(false.B) 

  val cmd   = Queue(io.cmd)
  val funct = cmd.bits.inst.funct
  val addr  = cmd.bits.rs2(log2Up(outer_n)-1, 0)
  val doWrite = funct === 0.U
  val doRead  = funct === 1.U 
  val doLoad  = funct === 2.U  //extend to load multiple multiple values with one command via Cache  
  val doAccum = funct === 3.U  //extend to add a offset to all elements of a vector 
  val doStore = funct === 4.U  //extend to store multiple values with one command via Cache
  val memRespTag = io.mem.resp.bits.tag(log2Up(outer_n)-1, 0)

  // Datapath 
  val addend = cmd.bits.rs1
  val accum = regfile(addr)
  val wdata = Mux(doWrite, addend, accum + addend)

  when(cmd.fire() && (doWrite || doAccum)) { //here later add compute flow TODO
    for (i <- 0 until (outer_n -1)  ) { 
      val temp = regfile(addr + i.U)
      regfile(addr + i.U) := temp + addend
    }
  }

  when(io.mem.resp.valid ) { //&& start_load //TODO here need only when load no on store  //&& doLoad ->freeze the load why?? TODO
    when(doLoad){ // new 
    regfile(memRespTag) := io.mem.resp.bits.data
    }
    busy(memRespTag) := false.B
  }

  // Control Logic for Multiple Loads
  when(cmd.fire() && (doLoad || doStore) ) {
    loadCount := 0.U // Reset load counter at the start of a load command
    start_load := true.B
  }
  
  when(io.mem.req.fire() && start_load) {  
    busy(addr + loadCount) := true.B
    loadCount := loadCount + 1.U // Increment load counter for each memory request
  }

  when(loadCount === outer_n.U -1.U) { 
    start_load := false.B
    loadCount := 0.U
  }

  // Response handling
  val doResp    = cmd.bits.inst.xd
  val stallReg  = busy(addr + loadCount)   
  val stallLoad = doLoad && !io.mem.req.ready 
  val stallResp = doResp && !io.resp.ready

  cmd.ready := !stallReg && !stallLoad && !stallResp

  // PROC RESPONSE INTERFACE
  io.resp.valid := cmd.valid && doResp && !stallReg && !stallLoad //&& loadCount === outer_n.U
  io.resp.bits.rd := cmd.bits.inst.rd
  io.resp.bits.data := accum // Adjust based on your requirement (you might need to aggregate results if necessary)

  io.busy := cmd.valid || busy.reduce(_||_) // Be busy when have pending memory requests or committed possibility of pending requests
  io.interrupt := false.B

  // Generate memory requests for multiple loads
  io.mem.req.valid        := (cmd.valid || start_load) && (doLoad || doStore) && !stallReg && ! stallResp //here maybe one by one request CHNAGE IT ! TODO  
  io.mem.req.bits.addr    := addend + (loadCount * 8.U)   // Increment address for each load request (assuming each value is 8 bytes)
  io.mem.req.bits.tag     := Mux(doLoad,addr + loadCount,addr + outer_n.U + loadCount)
  io.mem.req.bits.cmd     := Mux(doLoad,M_XRD,M_XWR) 
  io.mem.req.bits.size    := log2Ceil(8).U
  io.mem.req.bits.signed  := false.B
  io.mem.req.bits.data    := Mux(doStore,regfile(addr + loadCount),0.U) 
  io.mem.req.bits.phys    := false.B
  io.mem.req.bits.dprv    := cmd.bits.status.dprv
  io.mem.req.bits.dv      := cmd.bits.status.dv 
}

class WithLinearAccelerator extends Config((site, here, up) => {
  case BuildRoCC => Seq(
    (p: Parameters) => {
      implicit val implicitParams: Parameters = p // Ensure Parameters are passed implicitly
      implicit val valName: ValName = ValName("linear_filter_example") // Assign a name for the module

      LazyModule(
        new LinearFilterExample(
          opcodes = OpcodeSet.all,            // Opcode used for the accelerator
          params  = LinearFilterConfig.linearFilterConfig  // Use the config from the LinearFilterConfig object
        )
      )
    }
  )
})

