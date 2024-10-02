#include "rocc_custom.h"
#include "load_store_values.h"
#include "rocc.h"
#include <stdint.h>
#include <stdbool.h>

 /** helper functions for custom RoCC instructions **/
static inline unsigned long busy_read()
{
	unsigned long value;
	ROCC_INSTRUCTION_D(0, value, CUSTOM_READ_BUSY_FUNCT);
	return value;
}

static inline unsigned long read_vectors(float *buffer,int start_index)
{    
    float *target_address = buffer + start_index;
	 asm volatile ("fence");
	 ROCC_INSTRUCTION_S(0, &target_address ,CUSTOM_READ_OUTPUT_FUNCT); 
     //read fix num elemts in output  
}

static inline unsigned long set_bias(bool ISBIAS)
{   
	 asm volatile ("fence");
	 ROCC_INSTRUCTION_S(0, ISBIAS, CUSTOM_SET_ISBIAS);
}

static inline unsigned long write_vectors(float *buffer,int start_index)
{   
	 float *target_address = buffer + start_index;
	 asm volatile ("fence");
	 ROCC_INSTRUCTION_S(0, target_address, CUSTOM_LOAD_WEIGHTS_FUNCT);
     //write fix num elemts in accelerator buffers 
}


/** API Functions **/

static inline void linear_filter_load_all_weights(unsigned long *inputs, size_t numFilters, size_t vectorSize_in, size_t totalSize) {
    size_t offset = 0;
    while (offset < totalSize) {
          while (busy_read() == 0) {
            // Busy wait (polling) until the hardware is ready
            }
        write_vectors(inputs,offset);
        offset += vectorSize_in * numFilters;
    }
}

static inline void linear_filter_load_num_rows_inputs(unsigned long *inputs, size_t num_rows, size_t row_elems, size_t vectorSize_in) {
    size_t offset = 0;
    size_t totalSize = num_rows * row_elems;
    while (offset < totalSize) {
            while (busy_read() == 0)  {
            // Busy wait (polling) until the hardware is ready
            }
            
         write_vectors(inputs,offset);
         offset += vectorSize_in;
    }
}

static inline void linear_filter_num_rows_outputs(unsigned long *outputs, size_t numFilters, size_t num_rows, size_t row_elems, size_t vectorSize_out) {
    size_t offset = 0;
    size_t totalSize = num_rows*row_elems*numFilters;
    while (offset < totalSize) {
          while (busy_read() == 0)  {
            // Busy wait (polling) until the hardware is ready
            }
        linear_filter_read_output_chunk(&outputs[offset], offset);
        offset += vectorSize_out * numFilters;
    }
}


