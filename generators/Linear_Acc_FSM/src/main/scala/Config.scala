package linear_tranform

case class LinearFilterParams(
  inputRows:      Int,
  inputCols:      Int,
  outputCols:     Int,
  dataWidth:      Int,
  numFilters:     Int,
  VectorSize_w:   Int,
  VectorSize_b:   Int,
  VectorSize_in:  Int,
  VectorSize_out: Int,
  num_rows:       Int
)

// Create an object to hold your configuration
object LinearFilterConfig {
  val linearFilterConfig = LinearFilterParams(
    inputRows      = 1,             
    inputCols      = 1,
    outputCols     = 1,
    dataWidth      = 32,           
    numFilters     = 1,           
    VectorSize_w   = 1,        
    VectorSize_b   = 1,        
    VectorSize_in  = 1,       
    VectorSize_out = 1,      
    num_rows       = 1              
  )
}

// class LinearFilterExampleModuleImpl(outer: LinearFilterExample)(implicit p: Parameters) extends LazyRoCCModuleImp(outer) 
//   with HasCoreParameters {
  
//   val outer_n = 4  //here actually a parameter  (n+1) for array with n elemts!
//   val regfile = Mem(outer_n, UInt(xLen.W))
//   val busy = RegInit(VecInit(Seq.fill(outer_n) { false.B }))
//   val loadCount = RegInit(0.U(log2Ceil(outer_n).W)) // Track how many loads to perform based on outer_n
//   val start_load = RegInit(false.B)
//   val store_data = Wire(UInt()) 
//   store_data := 0.U 


//   val cmd = Queue(io.cmd)
//   val funct = cmd.bits.inst.funct
//   val addr = cmd.bits.rs2(log2Up(outer_n)-1, 0)
//   val doWrite = funct === 0.U
//   val doRead  = false.B  // funct === 1.U 
//   val doStore = funct === 1.U 
//   val doLoad  = funct === 2.U  //extend to load multiple multiple values with one command 
//   val doAccum = funct === 3.U  //extend to add a offset to all elements of a vector 
//   val memRespTag = io.mem.resp.bits.tag(log2Up(outer_n)-1, 0)

//   // Datapath 
//   val addend = cmd.bits.rs1
//   val accum = regfile(addr)
//   val wdata = Mux(doWrite, addend, accum + addend)

//   when(cmd.fire() && (doWrite || doAccum)) {
//     for (i <- 0 until (outer_n -1)  ) { 
//       val temp = regfile(addr + i.U)
//       regfile(addr + i.U) := temp + addend
//     }
//   }
//   when(io.mem.resp.valid && doLoad) { //maybe add a responce for store
//     regfile(memRespTag) := io.mem.resp.bits.data
//     busy(memRespTag) := false.B
//   }

//   // Control Logic for Multiple Loads
//   when(cmd.fire() && (doLoad || doStore)) {
//     loadCount  := 0.U // Reset load counter at the start of a load command
//     start_load := true.B
//   }
  
//   when(io.mem.req.fire() ) { //load and store  
//     busy(addr + loadCount) := true.B
//     loadCount := loadCount + 1.U // Increment load counter for each memory request 
//     store_data := Mux(doStore,regfile(addr + loadCount),0.U) // maybe before increaze loadCount
//   }

//   when(loadCount === outer_n.U -1.U) { 
//     start_load := false.B
//   }

//   // Response handling
//   val doResp    = cmd.bits.inst.xd
//   val stallReg  = busy(addr + loadCount)   
//   val stallLoad = doLoad && !io.mem.req.ready 
//   val stallResp = doResp && !io.resp.ready

//   cmd.ready := !stallReg && !stallLoad && !stallResp

//   // PROC RESPONSE INTERFACE
//   io.resp.valid := cmd.valid && doResp && !stallReg && !stallLoad //&& loadCount === outer_n.U
//   io.resp.bits.rd := cmd.bits.inst.rd
//   io.resp.bits.data := accum // Adjust based on your requirement (you might need to aggregate results if necessary)

//   io.busy := cmd.valid || busy.reduce( || ) // Be busy when have pending memory requests or committed possibility of pending requests
//   io.interrupt := false.B

//   val perform = Mux(doLoad,M_XRD,M_XWR)  //perform  Load or Store 

//   // Generate memory requests for multiple loads
//   io.mem.req.valid := (cmd.valid || start_load) && (doLoad || doStore) && !stallReg && !stallResp //here maybe one by one request CHNAGE IT ! TODO  
//   io.mem.req.bits.addr := addend + (loadCount * 8.U) // Increment address for each load request (assuming each value is 8 bytes)
//   io.mem.req.bits.tag :=  addr + loadCount // Use loadCount for tagging TODO ensuer load and store tags are differnt 
//   io.mem.req.bits.cmd :=  M_XRD // perform --new
//   io.mem.req.bits.size := log2Ceil(8).U
//   io.mem.req.bits.signed := false.B
//   io.mem.req.bits.data := 0.U //store_data // No data for loads --new
//   io.mem.req.bits.phys := false.B
//   io.mem.req.bits.dprv := cmd.bits.status.dprv
//   io.mem.req.bits.dv := cmd.bits.status.dv
  
// }

// class LinearFilterExampleModuleImpl(outer: LinearFilterExample)(implicit p: Parameters) extends LazyRoCCModuleImp(outer) 
//   with HasCoreParameters {
  
//   val outer_n = 4  //here actually a parameter  (n+1) for array with n elemts!
//   val regfile = Mem(outer_n, UInt(xLen.W))
//   val busy = RegInit(VecInit(Seq.fill(outer_n) { false.B }))
//   val loadCount = RegInit(0.U(log2Ceil(outer_n).W)) // Track how many loads to perform based on outer_n
//   val start_load = RegInit(false.B)


//   val cmd = Queue(io.cmd)
//   val funct = cmd.bits.inst.funct
//   val addr = cmd.bits.rs2(log2Up(outer_n)-1, 0)
//   val doWrite = funct === 0.U
//   val doRead  = funct === 1.U 
//   //val doStore = funct === 1.U 
//   val doLoad  = funct === 2.U  //extend to load multiple multiple values with one command 
//   val doAccum = funct === 3.U  //extend to add a offset to all elements of a vector 
//   val memRespTag = io.mem.resp.bits.tag(log2Up(outer_n)-1, 0)

//   // Datapath 
//   val addend = cmd.bits.rs1
//   val accum = regfile(addr)
//   val wdata = Mux(doWrite, addend, accum + addend)

//   when(cmd.fire() && (doWrite || doAccum)) {
//     for (i <- 0 until (outer_n -1)  ) { 
//       val temp = regfile(addr + i.U)
//       regfile(addr + i.U) := temp + addend
//     }
//   }

//   when(io.mem.resp.valid) {
//     regfile(memRespTag) := io.mem.resp.bits.data
//     busy(memRespTag) := false.B
//   }

//   // Control Logic for Multiple Loads
//   when(cmd.fire() && doLoad) {
//     loadCount := 0.U // Reset load counter at the start of a load command
//     start_load := true.B
//   }
  
//   when(io.mem.req.fire() ) { // && doLoad
//     busy(addr + loadCount) := true.B
//     loadCount := loadCount + 1.U // Increment load counter for each memory request
//   }

//   when(loadCount === outer_n.U -1.U) { 
//     start_load := false.B
//   }

//   // Response handling
//   val doResp    = cmd.bits.inst.xd
//   val stallReg  = busy(addr + loadCount)   
//   val stallLoad = doLoad && !io.mem.req.ready 
//   val stallResp = doResp && !io.resp.ready

//   cmd.ready := !stallReg && !stallLoad && !stallResp

//   // PROC RESPONSE INTERFACE
//   io.resp.valid := cmd.valid && doResp && !stallReg && !stallLoad //&& loadCount === outer_n.U
//   io.resp.bits.rd := cmd.bits.inst.rd
//   io.resp.bits.data := accum // Adjust based on your requirement (you might need to aggregate results if necessary)

//   io.busy := cmd.valid || busy.reduce( || 
// ) // Be busy when have pending memory requests or committed possibility of pending requests
//   io.interrupt := false.B

//    val perform  = Mux(doLoad,M_XRD,M_XWR)  //perform  Load or Store 

//   // Generate memory requests for multiple loads
//   io.mem.req.valid := (cmd.valid || start_load) && doLoad && !stallReg && !stallResp //here maybe one by one request CHNAGE IT ! TODO  
//   io.mem.req.bits.addr := addend + (loadCount * 8.U) // Increment address for each load request (assuming each value is 8 bytes)
//   io.mem.req.bits.tag :=  addr + loadCount // Use loadCount for tagging
//   io.mem.req.bits.cmd := M_XRD // Perform a load
//   io.mem.req.bits.size := log2Ceil(8).U
//   io.mem.req.bits.signed := false.B
//   io.mem.req.bits.data := 7.U  // No data for loads
//   io.mem.req.bits.phys := false.B
//   io.mem.req.bits.dprv := cmd.bits.status.dprv
//   io.mem.req.bits.dv := cmd.bits.status.dv
  
// }