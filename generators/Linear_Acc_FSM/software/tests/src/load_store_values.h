#ifndef LOAD_VALUES_H
#define LOAD_VALUES_H

#include <stddef.h>
#include <stdbool.h>

static inline unsigned long set_bias(bool ISBIAS);

/** API Functions **/

/**
 * @brief Load num_rows input values into the hardware in chunks.
 * 
 * This function iteratively loads chunks of input data from the input array
 * into the hardware. It breaks the input array into smaller pieces (chunks) 
 * and sends them one at a time using the `linear_filter_load_inputs_chunk` function.
 * 
 * Note: The structure of the input data is assumed to be in the form:
 * | row_0 | row_1 | ... | row_n |
 * 
 * @param [in] inputs         Pointer to the array of input values (base address) 
 * @param [in] num_rows       Number of input rows to be loaded at once into the hardware
 * @param [in] row_elems      Number of elements per row       
 * @param [in] vectorSize_in  Maximum number of elements to load in each chunk
 * 
 * @return void               No direct output, sends data to hardware in chunks
 */
static inline void linear_filter_load_num_rows_inputs(unsigned long *inputs, size_t num_rows, size_t row_elems, size_t vectorSize_in);

/**
 * @brief Load all weight values for multiple filters into the hardware in chunks.
 * 
 * This function breaks the weight data into chunks and sends them to the hardware.
 * Each chunk includes the weight values for all filters. The size of each chunk is
 * determined by the provided vector size.
 * 
 * Note: The structure of the input data is assumed to be in the form:
 *   message[0] = weights_filter_0 chunk[0] | weights_filter_1 chunk[0] | ... | weights_filter_n chunk[0] | 
 * 
 * @param [in] inputs         Pointer to the array of weight values (base address)
 * @param [in] numFilters     Number of filters whose weights need to be loaded
 * @param [in] totalSize      Total number of weight elements to be loaded (numFilters * outputCols * InputCols)
 * @param [in] vectorSize_w   Maximum number of elements to load in each chunk
 * 
 * @return void               No direct output, sends weight data to hardware in chunks
 */
static inline void linear_filter_load_all_weights(unsigned long *inputs, size_t numFilters, size_t totalSize, size_t vectorSize_w);

/**
 * @brief Read  bum_row data for every filter from the hardware in chunks.
 * 
 * This function reads chunks of output data from the hardware. Each chunk 
 * contains output values from all filters. The size of each chunk is 
 * determined by the provided vector size.
 * 
 * Note: The structure of the output data is assumed to be in the form:
 * store[0]: | out_filter_0 chunk[0] | out_filter_1 chunk[0] | ... | out_filter_n chunk[0] |
 * store[0] | store[1] | ... | store[n]
 * 
 * @param [out] outputs        Pointer to the array where output values will be stored (base address)
 * @param [in]  numFilters     Number of filters whose outputs need to be read
 * @param [in]  num_rows       Number of input rows to be loaded at once into the hardware
 * @param [in]  row_elems      Number of elements per row  
 * @param [in]  vectorSize_out Maximum number of elements to read in each chunk
 * 
 * @return void               No direct output, reads output data from hardware in chunks
 */
static inline void linear_filter_num_rows_outputs(unsigned long *outputs, size_t numFilters, size_t num_rows, size_t row_elems, size_t vectorSize_out);

#endif // LOAD_VALUES_H
