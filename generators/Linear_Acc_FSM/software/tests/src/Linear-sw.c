/* SW TESTING tempalte 
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h> 
#include "compiler.h"
#include <stdbool.h>
//#include "load_store_values.h"
//#include "rocc_custom.h"
#include "rocc.h"

#include <stdio.h>

// Define custom function codes
#define CUSTOM_SET_ISBIAS 0
#define CUSTOM_LOAD_WEIGHTS_FUNCT 1
#define CUSTOM_LOAD_INPUTS_FUNCT 3
#define CUSTOM_READ_OUTPUT_FUNCT 4

typedef unsigned long elem_t;

void generate_random_matrix(elem_t *matrix, int size) {
    for (int i = 0; i < size; ++i) { 
        matrix[i] = (elem_t)rand()/ RAND_MAX; ;
    }
}

static inline long rdcycle(void)
{
	long cycle;
	asm volatile ("csrr %[cycle], cycle" : [cycle] "=r" (cycle));
	return cycle;
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

 static inline unsigned long read_value()
{
	unsigned long value;
    ROCC_INSTRUCTION_D(0,value,6); //CUSTOM_READ_OUTPUT_FUNCT
	return value; 
}

 void read_vector(unsigned long *buffer,int start_index)
{    
     unsigned long *target_address = buffer + start_index;
	 asm volatile ("fence");
	 ROCC_INSTRUCTION_D(0,buffer[1] ,CUSTOM_READ_OUTPUT_FUNCT);  
}

// Memory Accelerator Interface 
 static inline void set_bias(unsigned long value)
{   
    ROCC_INSTRUCTION_S(0,value,CUSTOM_SET_ISBIAS);
}

static inline void load_weights(void *ptr)
{
	asm volatile ("fence");
    ROCC_INSTRUCTION_S(0,(uintptr_t) ptr,CUSTOM_LOAD_WEIGHTS_FUNCT);
}

static inline void load_bias(void *ptr)
{
	asm volatile ("fence");
    ROCC_INSTRUCTION_S(0,(uintptr_t) ptr,CUSTOM_LOAD_ΒΙΑΣ_FUNCT);
}

static inline void load_input_rows(void *ptr)
{
	asm volatile ("fence");
    ROCC_INSTRUCTION_S(0,(uintptr_t) ptr,CUSTOM_LOAD_INPUTS_FUNCT);
}

static inline void store_output_rows(void *ptr)
{
	asm volatile ("fence");
    ROCC_INSTRUCTION_S(0,(uintptr_t) ptr,CUSTOM_READ_OUTPUT_FUNCT);
}

 static inline unsigned long read_status()
{
	unsigned long value;
    ROCC_INSTRUCTION_D(0,value,6); 
	return value; 
}

int main() {

    // Initialize the linear filter parameters
    LinearFilterParams p = {
        .inputRows = 1, 
        .inputCols = 1,
        .outputCols = 1,
        .dataWidth = 32,
        .numFilters = 1,
        .VectorSize_w = 1,
        .VectorSize_b = 1,
        .VectorSize_in = 1,
        .VectorSize_out = 1,
        .num_rows = 1
    };
  bool ISBIAS = true;

unsigned long start, end;

elem_t Weights[p.numFilters * p.inputCols * p.outputCols];            // Weights[p.numFilters][p.inputCols * p.outputCols]
elem_t biases[p.numFilters * p.outputCols];                           // biases[p.numFilters][p.outputCols]
elem_t input_tensor[p.inputCols * p.inputRows];                       // input_tensor[p.inputCols * p.inputRows]
elem_t out_matrix_sw[p.numFilters * 1 * p.inputRows * p.outputCols];  // out_matrix[p.numFilters][1 * p.inputRows * p.outputCols]
elem_t out_matrix_hw[p.numFilters * 1 * p.inputRows * p.outputCols]; 


// calculation for values --example
printf("Hello World!\n");

  generate_random_matrix(Weights,p.numFilters*p.inputCols* p.outputCols);
  generate_random_matrix(biases, p.numFilters*p.outputCols);
  generate_random_matrix(input_tensor, 1 * p.inputCols * p.inputRows);

    // Printing Weights as multiple 2D arrays (one for each filter)
    printf("Weights (as multiple 2D arrays):\n");
    for (int f = 0; f < p.numFilters; f++) {
        printf("Filter %d:\n", f);
        for (int i = 0; i < p.inputCols; i++) {
            for (int j = 0; j < p.outputCols; j++) {
                // Compute the correct index in the 1D array
                printf("%f ", Weights[f * (p.inputCols * p.outputCols) + i * p.outputCols + j]);
            }
            printf("\n");  // New line after each row
        }
        printf("\n");  // Extra line between filters
    }

    // Printing biases as 2D arrays (one for each filter)
    printf("Biases (as 1D arrays per filter):\n");
    for (int f = 0; f < p.numFilters; f++) {
        printf("Filter %d:\n", f);
        for (int j = 0; j < p.outputCols; j++) {
            printf("%f ", biases[f * p.outputCols + j]);
        }
        printf("\n");  // New line after each filter
    }

    // Printing input_tensor as a 2D array [p.inputRows][p.inputCols]
    printf("Input Tensor (as a 2D array):\n");
    for (int i = 0; i < p.inputRows; i++) {
        for (int j = 0; j < p.inputCols; j++) {
            printf("%f ", input_tensor[i * p.inputCols + j]);
        }
        printf("\n");  // New line after each row
    }


  do {
    printf("\n");
    printf("########################################################\n");
    printf("Start Linear Transform in SW\n");
    start = rdcycle();
    for(int i =0; i< p.numFilters; i++){ 
     linear_transform(input_tensor, 1, p.inputRows, p.inputCols,
                      Weights + i *p.inputCols * p.outputCols,p.inputRows, p.outputCols, 
                      biases + i * p.outputCols, 
                      out_matrix_sw + i * (p.inputRows * p.outputCols));
    }
    end = rdcycle();
    printf("Success!\n");
    printf("Linear execution took %lu cycles in SW \n", end - start);
    
    printf("########################################################\n");
    printf("\n");

    printf("########################################################\n");
    printf("Start Linear Transform in HW\n");
    start = rdcycle();
    //preload phase // 
        set_bias(ISBIAS);
        if (busy_read())  {
                printf("HW is busy \n");
        }else {
                printf("HW is not busy \n");
        }

     size_t totalSize = p.numFilters *p.inputCols * p.outputCols;
     printf("Preload weights for all filters\n");
     linear_filter_load_all_weights(Weights,p.numFilters,p.VectorSize_w,totalSize);
    if(ISBIAS){ 
        printf("Preload bias for all filters\n");
        size_t totalSize = p.numFilters * p.outputCols;
        linear_filter_load_all_weights(biases,p.numFilters,p.VectorSize_b,totalSize);
    }


     //load+compute+store phase 
    for(int row = 0; row < p.inputRows; row += p.num_rows){ 
        int step = row * p.inputCols;

        // Step 0: Load phase
        // printf("Loading %d rows of input starting from row %d\n", p.num_rows, row);
        linear_filter_load_num_rows_inputs(input_tensor + step, p.num_rows, p.inputCols, p.VectorSize_in);

        // Step 2: Store phase
        //printf("Storing %d rows of output for each filter\n", p.num_rows);
        linear_filter_num_rows_outputs(out_matrix_hw + step * p.numFilters, p.numFilters, p.num_rows, p.inputCols, p.VectorSize_out);
    }


    end = rdcycle();
    printf("Linear execution took %lu cycles in HW \n", end - start); 
    printf("########################################################\n");
    printf("\n");

    // Printing out_matrix_sw as multiple 2D arrays (one for each filter)
printf("Output Matrix SW (as multiple 2D arrays):\n");
for (int f = 0; f < p.numFilters; f++) {
    printf("Filter %d:\n", f);
    for (int i = 0; i < p.inputRows; i++) {
        for (int j = 0; j < p.outputCols; j++) {
            // Compute the correct index in the 1D array
            printf("%f ", out_matrix_sw[f * (p.inputRows * p.outputCols) + i * p.outputCols + j]);
        }
        printf("\n");  // New line after each row
    }
    printf("\n");  // Extra line between filters
}

    Printing out_matrix_sw as multiple 2D arrays (one for each filter)
printf("Output Matrix HW (as multiple 2D arrays):\n");
for (int f = 0; f < p.numFilters; f++) {
    printf("Filter %d:\n", f);
    for (int i = 0; i < p.inputRows; i++) {
        for (int j = 0; j < p.outputCols; j++) {
            // Compute the correct index in the 1D array
            printf("%f ", out_matrix_hw[f * (p.inputRows * p.outputCols) + i * p.outputCols + j]);
        }
        printf("\n");  // New line after each row
    }
    printf("\n");  // Extra line between filters
}


    //compare SW and HW 
    int count  = 0;
    int step   = 0;
    bool match = true;

     for(int row = 0; row < p.inputRows; row++) { 
        int seq_rows = row*p.inputCols;
         for(int elemts_row =0; elemts_row < p.inputCols; elemts_row++) { 
            if(out_matrix_sw[seq_rows + elemts_row] != out_matrix_hw[step + count*p.inputCols + elemts_row]) match = false;
         }

         count++;

        if(count == p.num_rows-1) { 
            count = 0 ;
            step += p.num_rows * p.inputCols * p.numFilters;
            }  

     }

     if(match) {
        printf("HW AND SW outputs are the same\n");
     } else { 
        printf("HW AND SW outputs are different\n");
     }
     
  } while(0);

  return 0;
}
*/

#include "rocc.h"
#include <stdio.h>

static inline void accum_write(int idx, unsigned long data)
{
	ROCC_INSTRUCTION_SS(0, data, idx, 0);
}

static inline unsigned long accum_read(int idx)
{
	unsigned long value;
	ROCC_INSTRUCTION_DSS(0, value, 0, idx, 1);
	return value;
}

static inline void accum_load(int idx, void *ptr)
{
	asm volatile ("fence");
	ROCC_INSTRUCTION_SS(0, (uintptr_t) ptr, idx, 2);
}

static inline void accum_add(int idx, unsigned long addend)
{
	ROCC_INSTRUCTION_SS(0, addend, idx, 3);
}

static inline void accum_store(int idx, void *ptr)
{
	asm volatile ("fence");
	ROCC_INSTRUCTION_SS(0, (uintptr_t) ptr, idx, 4);

    asm volatile ("fence");
	ROCC_INSTRUCTION_SS(0, (uintptr_t) ptr, idx, 4)
}

/*//////////////////////////////////  EXAMPLE for LOAD MULTIPLE VALUES one per instruction /////////////////////////////////
#include <stdio.h>
unsigned long data_matrice[10] = {0, 1, 2, 3};
int main(void) {
    unsigned long result;

    // Manually load all elements from data_matrice into the accumulator
    for (int i = 0; i < 4; i++) {
        accum_load(i, &data_matrice[i]);  // Load the value at the current index into the accumulator
        printf("Loaded data_matrice[%d]: %lu into accumulator at index %d\n", i, data_matrice[i], i);
    }

    // Test reading back from the accumulator
    for (int i = 0; i < 4; i++) {
        result = accum_read(i); // Read the value back from the accumulator
        printf("Reading accumulator at index %d: %lu\n", i, result);
    }

    return 0;
}
*/

/*//////////////////////////////////  EXAMPLE for WRITE AND LOAD OPERATIONS ////////////////////////////////////////
#include <stdio.h>
int main(void) {

    printf("WRITE OPERATION EXAMPLE\n");
    printf("Writing data 3 to the accumulator...\n");
    accum_write(0, 3);
    printf("Adding +1 to the accumulator...\n");
    accum_add(0, 1);
    unsigned long result = accum_read(0); // Read the new value
    printf("New data is: %lu\n", result);

    unsigned long data_matrice[4] = {0, 1, 2, 3}; // Initialize with some values
    unsigned long data; 

    printf("LOAD OPERATION EXAMPLE\n");
    printf("Loading data from matrice[0]...\n");
    data = data_matrice[0]; 
    accum_load(0, &data); 
    printf("Reading data ... \n");
    result = accum_read(0);   //here freeze
    printf("Loaded data: %lu\n", result);
    printf("Adding +2 to the accumulator...\n");
    accum_add(0, 2);
    result = accum_read(0); 
    printf("New data is: %lu\n", result);
    return 0;
}
*/

//////////////////////////////////  EXAMPLE for  load and store via caches ////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
int main(void) {
    // Correctly declare data_matrice as a pointer to unsigned long
    unsigned long *data_in = malloc(10 * sizeof(unsigned long)); // Dynamically allocate memory
    if (data_in == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1; // Exit if allocation fails
    }

   unsigned long *data_out = malloc(10 * sizeof(unsigned long)); // Dynamically allocate memory
    if (data_out == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1; // Exit if allocation fails
    }

    // Initialize the pointer with values
    for (int i = 0; i < 10; i++) {
        *(data_in + i) = i; // Assign values to the first 10 elements
    }
    // Initialize the pointer with values
    for (int i = 0; i < 10; i++) {
        *(data_out + i) = 1; // Assign values to the first 10 elements
    }

for ( int i=0; i<2;i++) {
    unsigned long result;
    printf("Loading matrice ...\n");
    accum_load(0, data_in);
    printf("Loading done\n");

   // Test reading back from the accumulator
    for (int i = 0; i < 3; i++) {
        result = accum_read(i); // Read the value back from the accumulator
        printf("Reading accumulator at index %d: %lu\n", i, result);
    }
     
     printf("Add + %d \n",i+1);
     accum_add(0,i+1);
     
    for (int i = 0; i < 3; i++) {
        result = accum_read(i); // Read the value back from the accumulator
        printf("Reading accumulator at index %d: %lu\n", i, result);
    }

    printf("Start storing results...\n");

    accum_store(0, &data_out[0]);

    printf("Store Finished\n");

   printf("Contents of data_matrice:\n");
    for (int i = 0; i < 3; i++) {
        printf("data_matrice[%d]: %lu\n", i, data_out[i]);
    }

}
    free(data_in); // Free the allocated memory
    free(data_out);

    return 0;
}

