package linear_tranform

import chisel3._
import chisel3.util._
import chisel3.util.HasBlackBoxResource
import chisel3.experimental.IntParam
import org.chipsalliance.cde.config._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.rocket._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.util.InOrderArbiter

case class LinearFilterParams(
  inputRows:     Int,
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


class LinearFilterExample(
    opcodes: OpcodeSet,
    params: LinearFilterParams 
)(implicit p: Parameters) extends LazyRoCC(opcodes) {
  override lazy val module = new LinearFilterExampleModuleImp(this)
}

class LinearFilterExampleModuleImpl(outer: LinearFilterExample)(implicit p: Parameters) extends LazyRoCCModuleImp(outer) 
with HasCoreParameters {

    val linear = Module(new MultiLinearFilter_fsm(
        params.inputRows,
        params.inputCols,
        params.outputCols,
        params.dataWidth,
        params.numFilters,
        params.VectorSize_w,
        params.VectorSize_b,
        params.VectorSize_in,
        params.VectorSize_out,
        params.num_rows
    ))
        //RISCV --> RoCC  
        val cmd = Queue(io.cmd) 
        val funct = cmd.bits.inst.funct
        val rs1 = cmd.bits.rs1
        //val ISBIAS = cmd.bits.rs2 
        
        //states 
        val set_bias           =    (funct === 0.U)   
        val load_weights       =    (funct === 1.U && !io.loadWeightsDone) 
        val load_bias          =    (funct === 1.U && io.loadWeightsDone && !io.loadBiasesDone && ISbias)
        val load_input_rows    =    (funct === 2.U && !io.loadInputsDone)
        val compute_ouput_rows =    (funct === 2.U && io.loadInputsDone)
        val store_output_rows  =    (funct === 3.U && !storeOutputsDone)
        val read_busy          =    (funct === 4.U)

        //RISCV <-- RoCC  
        io.interrupt := false.B          // TODO update pooling to interupt 
        io.busy      := linear.io.busy   
        cmd.ready    := !linear.io.busy   // TODO pipeline phases 

        //bias 
        ISBIAS = RegInit(false.B)

        when(read_busy) { //ready busy state 
                io.resp.bits.data :=linear.io.busy 
                io.resp.valid := true.B
                io.resp.bits.rd := cmd.bits.inst.rd 
            }

        when(cmd.valid && cmd.ready) {  
            linear.io.stateInput := funct 

            when(set_bias) { 
                linear.io.ISbias := rs1 
                ISBIAS := true.B
            }.elsewhen(load_weights) { 
                for (i <- 0 until params.numFilters) {
                    for (j <- 0 until params.VectorSize_w) {
                        linear.io.weight_elems(i)(j).bits := rs1 + (i * params.VectorSize_w + j) 
                        linear.io.weight_elems(i)(j).valid := true.B  
                    }
                } 
            }.elsewhen(load_bias) {
                for (i <- 0 until params.numFilters) {
                    for (j <- 0 until params.VectorSize_b) {
                        linear.io.bias_elems(i)(j).bits := rs1 + (i * params.VectorSize_b + j) 
                        linear.io.bias_elems(i)(j).valid := true.B 
                    }
                }
            }.elsewhen(load_input_rows) {
                for (j <- 0 until params.VectorSize_in) {
                    linear.io.input_x_elem(j).bits := rs1 + j 
                    linear.io.input_x_elem(j).valid := true.B  
                }
            }.elsewhen(store_output_rows) {
                for (i <- 0 until params.numFilters) {
                    for (j <- 0 until params.VectorSize_out) { 
                        io.resp.bits.data := linear.io.output_elems(i)(j).bits // Data to store at memory address
                        io.resp.valid := linear.io.output_elems(i)(j).valid 
                        io.resp.bits.rd := cmd.bits.inst.rd  + (i * params.VectorSize_out.U + j.U)// Send data to memory at rd + offset
                    }  
                }
            }
    } 

}

class WithLineaAccelerator extends Config((site, here, up) => {
  case BuildRoCC => Seq((p: Parameters) => LazyModule(
    new LinearFilterExample(OpcodeSet.custom0,linearFilterConfig)(p)))
})


