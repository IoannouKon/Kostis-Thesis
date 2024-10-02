//see LICENSE for license
// The following is a RISC-V program to test the functionality of the
// Linaer RoCC accelerator.
// Compile with riscv-gcc sha3-rocc.c 
// Run with spike --extension=sha3 pk a.out

#include <stdio.h>
#include <stdint.h>
#include "compiler.h"
#include <stdbool.h>
#include "load_store_values.h"

#define RAND_MAX 2147483647

typedef float elem_t;

void generate_random_matrix(elem_t *matrix, int size) {
    for (int i = 0; i < size; ++i) { 
        matrix[i] = (elem_t)rand()/ RAND_MAX; ;
    }
}

void linear_transform(elem_t* input_matrix, int batch_size, int input_rows, int input_cols,
                     elem_t* weight_matrix, int weight_rows, int weight_cols,
                     elem_t* bias_vector, elem_t* output_matrix) {
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < input_rows; ++i) {
            for (int j = 0; j < weight_cols; ++j) {
                elem_t sum = 0.0f;
                for (int k = 0; k < input_cols; ++k) {
                    sum += input_matrix[b * input_rows * input_cols + i * input_cols + k] * weight_matrix[k * weight_cols + j];
                }
                if (bias_vector != NULL) {
                    sum += bias_vector[j];
                }
                output_matrix[b * input_rows * weight_cols + i * weight_cols + j] = sum;
            }
        }
    }
}


typedef struct {
    int inputRows;      // Number of input rows
    int inputCols;      // Number of input columns
    int outputCols;     // Number of output columns
    int dataWidth;      // Width of data (in bits)
    int numFilters;     // Number of filters
    int VectorSize_w;   // Size of weight vectors
    int VectorSize_b;   // Size of bias vectors
    int VectorSize_in;  // Size of input vectors
    int VectorSize_out; // Size of output vectors
    int num_rows;       // Number of rows in the filter
} LinearFilterParams;


int main() {

    // Initialize the linear filter parameters
    LinearFilterParams p = {
        .inputRows = 8, 
        .inputCols = 8,
        .outputCols = 8,
        .dataWidth = 32,
        .numFilters = 1,
        .VectorSize_w = 1,
        .VectorSize_b = 1,
        .VectorSize_in = 1,
        .VectorSize_out = 1,
        .num_rows = 1
    };
  bool ISBIAS =true;

  unsigned long start, end;
  elem_t Weights[p.numFilters][p.inputCols * p.outputCols];
  elem_t biases[p.numFilters][p.outputCols];
  elem_t input_tensor[p.inputCols * p.inputRows];
  elem_t out_matrix[p.numFilters][1* p.inputRows*p.outputCols];


  generate_random_matrix(Weights, p.inputCols* p.outputCols);
  generate_random_matrix(biases, p.outputCols);
  generate_random_matrix(input_tensor, 1 * p.inputCols * p.inputRows);

  do {
    printf("Start Linear Transform  in SW .\n");
    start = rdcycle();
    for(int i =0; i< p.numFilters; i++){ 
     linear_transform(input_tensor, 1, p.inputRows, p.inputCols, Weights[i],p.inputRows, p.outputCols, biases[i], out_matrix[i]);
    }
    end = rdcycle();
    printf("Success!\n");
    printf("Linear execution took %lu cycles in SW \n", end - start);

    printf("Start Linear Transform  in HW .\n");
    start = rdcycle();
    
    //preload phase // 
    set_bias(ISBIAS);
    size_t totalSize = p.numFilters *p.inputCols * p.outputCols;
    linear_filter_load_all_weights(Weights,p.numFilters,p.VectorSize_w,totalSize);
    if(ISBIAS){ 
        size_t totalSize = p.numFilters * p.outputCols;
       linear_filter_load_all_weights(biases,p.numFilters,p.VectorSize_b,totalSize);
    }
    //buffer for HW must have onother structure 
    //load+compute+store phase 
    for(int step =0 ;step < p.inputRows; step += p.num_rows){ 
        linear_filter_load_num_rows_inputs(input_tensor + step,p.num_rows,p.inputCols,p.VectorSize_in);
        linear_filter_num_rows_outputs(out_matrix +step*p.numFilters,p.numFilters, p.num_rows,p.inputCols,p.VectorSize_out);
    }

    end = rdcycle();
    printf("Success!\n");
    printf("Linear execution took %lu cycles in HW \n", end - start); 

  } while(0);

  return 0;
}