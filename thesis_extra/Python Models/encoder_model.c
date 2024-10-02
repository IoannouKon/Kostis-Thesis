#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>
#include <float.h>  // For FLT_MIN
#include <assert.h>
#include <time.h>
#include <cblas.h>
#include <sys/resource.h>
#include <unistd.h>


/////////////////////// FOR DEBUG FUNCTIONS //////////////////////////// 

// Function to convert float to string
void floatToString(float value, char* buffer, int precision) {
    // Handle negative numbers
    if (value < 0) {
        *buffer++ = '-';
        value = -value;
    }

    // Extract integer part
    int intPart = (int)value;
    value -= intPart;

    // Convert integer part to string
      // itoa(intPart, buffer, 10);
      // sprintf(buffer, "%d", value); // Converts integer to string
    while (*buffer != '\0') {
        buffer++;
    }

    // Add decimal point
    *buffer++ = '.';

    // Extract fractional part
    for (int i = 0; i < precision; i++) {
        value *= 10;
        int digit = (int)value;
        *buffer++ = '0' + digit;
        value -= digit;
    }

    // Null-terminate the string
    *buffer = '\0';
}

// Function to print a float number
void printFloat(float number ) {
    char buffer[50]; // Buffer to hold string representation of float
    floatToString(number, buffer, 6); // Convert float to string with 6 decimal places
    printf("%s ", buffer);
}

// Function to print a matrix
void printMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printFloat(matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void print3DArrayAs2D(float *head_out, int batch_size, int num_tokens, int value_size) {
    for (int b = 0; b < batch_size; b++) {
        printf("Batch %d:\n", b);
        printMatrix(head_out + b * num_tokens * value_size, num_tokens, value_size);
        printf("\n");
    }
}

// Function to generate random values for a matrix
void generate_random_matrix(float *matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

void generate_random_mask(int size, float *mask) {
    // Seed the random number generator
    srand((unsigned int)time(NULL));

    // Calculate the total number of elements in the mask
    int total_elements = size;

    for (int i = 0; i < total_elements; i++) {
        // Generate a random value (0 or 1)
        mask[i] = 1;// rand() % 2;
    }
}

// Function to print a 3D array as a 2D matrix
void print3DArrayAs2D_FLOAT(float *array, int batch_size, int num_tokens, int d_out) {
    // Loop through each 2D matrix in the 3D array
    for (int b = 0; b < batch_size; b++) {
        printf("BATCH %d:\n", b);
        for (int i = 0; i < num_tokens; i++) {
            for (int j = 0; j < d_out; j++) {
                // Calculate the index for the 3D array
                int index = b * num_tokens * d_out + i * d_out + j;
                printf("%f ", array[index]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
float print_memory_usage() {
    struct rusage usage;
   getrusage(RUSAGE_SELF, &usage);
    //printf("Memory usage: %ld KB\n", usage.ru_maxrss);
    return usage.ru_maxrss;
}

// Function to measure CPU utilization
void print_cpu_usage(clock_t start, clock_t end) {
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("CPU time used: %f seconds\n", cpu_time_used);
}

// Function to save an array to a file
void save_array_to_file(float *array, int size, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++) {
        fprintf(file, "%f\n", array[i]);
    }
    fclose(file);
}


////////////////////////////// Matrice computations FUNCTIONS //////////////////////////////

// Function to perform matrix multiplication
void matrixMultiply_manual(float *A, float *B, float *C, int rowA, int colA, int colB) {
    for (int i = 0; i < rowA; ++i) {
        for (int j = 0; j < colB; ++j) {
            C[i * colB + j] = 0;
            for (int k = 0; k < colA; ++k) {
                C[i * colB + j] += A[i * colA + k] * B[k * colB + j];
            }
        }
    }
}

void matrixMultiply(float *A, float *B, float *C, int rowA, int colA, int colB) {
    // Perform matrix multiplication using CBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rowA, colB, colA,
                1.0f, A, colA,
                B, colB,
                0.0f, C, colB);
}

// Function to transpose a matrix
void transpose(float *matrix, float *transposed, int rows, int columns) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            transposed[j * rows + i] = matrix[i * columns + j];
        }
    }
}

void transpose_in_place(float *matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = i + 1; j < size; ++j) {
            // Swap the elements at positions (i, j) and (j, i)
            float temp = matrix[i * size + j];
            matrix[i * size + j] = matrix[j * size + i];
            matrix[j * size + i] = temp;
        }
    }
}



// Function to scale a 2D matrix
void Scale_2D(float *A, int n, int dk) {
    float sqrt_dk = sqrt(dk);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i * n + j] /= sqrt_dk;
        }
    }
}

void apply_mask_old(float* dot, float* mask, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dot[i * n + j] = dot[i * n + j] - 1e6 * (1 - mask[i * n + j]);
        }
    }
}

void apply_mask(float* dot, float* mask,int num_tokens) {
    for(int i =0;i<num_tokens;i++){ 
        for (int j = 0; j < num_tokens; j++) {
            dot[i*num_tokens+j] = dot[i*num_tokens+j] - 1e6 * (1 - mask[j]);
        }
    }        

}

// Function to apply dropout to a matrix
void apply_dropout(float *matrix, int rows, int cols, float p) {
    float scale = 1.0 / (1.0 - p); // Scaling factor
    srand((unsigned int)time(NULL)); // Seed the random number generator

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Apply dropout with probability p
            if ((rand() / (float)RAND_MAX) < p) {
                matrix[i * cols + j] = 0.0; // Set to zero
            } else {
                matrix[i * cols + j] *= scale; // Scale the remaining values
            }
        }
    }
}

void softmax_new(float *matrix, int rows, int cols, float *result) {
    for (int i = 0; i < rows; ++i) {
        float max = matrix[i * cols];
        // Find the max value in the row
        for (int j = 1; j < cols; ++j) {
            if (matrix[i * cols + j] > max) {
                max = matrix[i * cols + j];
            }
        }

        // Calculate the exponentials and sum
        float sum_exp = 0.0f;
        for (int j = 0; j < cols; ++j) {
            result[i * cols + j] = exp(matrix[i * cols + j] - max);
            sum_exp += result[i * cols + j];
        }

        // Normalize
        for (int j = 0; j < cols; ++j) {
            result[i * cols + j] /= sum_exp;
        }
    }
}

void split_QKV_for_heads(float *Wk, float *Wk_splits, int batch_size, int embedding_size, int dk, int num_heads) {
    int head_dim = dk / num_heads;

    // Iterate over the batch size
    for (int b = 0; b < batch_size; ++b) {
        // Iterate over the number of heads
        for (int h = 0; h < num_heads; ++h) {
            // Iterate over the embedding size (rows)
            for (int i = 0; i < embedding_size; ++i) {
                // Copy the relevant portion of the weight matrix to the split matrices
                for (int j = 0; j < head_dim; ++j) {
                    // Calculate the position in the output array
                    int out_idx = b * num_heads * embedding_size * head_dim + h * embedding_size * head_dim + i * head_dim + j;
                    // Calculate the position in the input array
                    int in_idx = b * embedding_size * dk + i * dk + h * head_dim + j;
                    Wk_splits[out_idx] = Wk[in_idx];
                }
            }
        }
    }
}

void transpose_full(float *X, float *X_t, int batch_size, int num_heads, int num_tokens, int head_dim) {
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < num_tokens; ++t) {
            for (int h = 0; h < num_heads; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    X_t[((b * num_tokens + t) * num_heads + h) * head_dim + d] = 
                        X[((b * num_heads + h) * num_tokens + t) * head_dim + d];
                }
            }
        }
    }
}

void combine_heads(float *X, float *X_out, int batch_size, int num_heads, int num_tokens, int head_dim) {
    int embed_dim = num_heads * head_dim;
    
    
    float *X_t = (float *)malloc(batch_size * num_tokens * num_heads * head_dim * sizeof(float));
    transpose_full(X, X_t, batch_size, num_heads, num_tokens, head_dim);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < num_tokens; ++t) {
            for (int e = 0; e < embed_dim; ++e) {
                X_out[b * num_tokens * embed_dim + t * embed_dim + e] = 
                    X_t[(b * num_tokens + t) * embed_dim + e];
            }
        }
    }
    free(X_t);
}

void relu(float* input, int size) {
    for (int i = 0; i < size; i++) {
        input[i] = (input[i] > 0) ? input[i] : 0;
    }
}

////////////////////////////////////////////COMPUTE SELF-ATTENTION //////////////////////////////

// Define the AttentionInputs struct
typedef struct {
    float *Q;    //Q[n][dq]
    float *K;    //K[n][dk] where dk=dq
    float *V;    //V[n][dv]
    float *mask;   
    int batch_size;
    int n;       // number of input tokens   
    int dk;      
    int dv;
    float p;     //propabily for bernulli 
    float *Z;    //output matrice 
} AttentionInputs;

// this not copy the data just give the pointers as initiallize 
void initAttentionInputs(AttentionInputs *ai, float *Q, float *K, float *V, float *mask, int batch_size,int n, int dk, int dv, float p) {
    ai->Q = Q;
    ai->K = K;
    ai->V = V;
    ai->mask = mask;
    ai->batch_size =batch_size;
    ai->n = n;
    ai->dk = dk;
    ai->dv = dv;
    ai->p = p;
    ai->Z = (float *)malloc(batch_size * n * dv * sizeof(float)); // Allocate memory for result matrix
    if (!ai->Z) {
        perror("Failed to allocate memory for Z");
        exit(EXIT_FAILURE);
    }
}

// Function to free the memory allocated for AttentionInputs
void freeAttentionInputs(AttentionInputs *ai) {
       free(ai->Z);
}

void ScaleAttention(AttentionInputs *ai, float *SA) {


      for (int b = 0; b < ai->batch_size; ++b) {
            
            float *K_T =  (float *)malloc( ai->dk * ai->n * sizeof(float));
            float *dot =  (float *)malloc( ai->n * ai->n *sizeof(float)); 
            float *dot1 = (float *)malloc( ai->n * ai->n * sizeof(float)); 
         transpose(&ai->K[b * ai->n * ai->dk], K_T, ai->n, ai->dk);

          // Perform matrix multiplication Q * K_T for the current batch
          matrixMultiply(&ai->Q[b * ai->n * ai->dk], K_T, dot, ai->n, ai->dk, ai->n);

         //Scale the dot product for the current batch
          Scale_2D(dot, ai->n, ai->dk); 
         //printf("Score is\n");
        // print3DArrayAs2D(dot,ai->batch_size,ai->n,ai->n);
         
        //  printf("the mask is\n");
        //  print3DArrayAs2D(&ai->mask[b * ai->n],1,ai->n,1); 
  
        // printf("dot before mask is\n");
        // print3DArrayAs2D(&dot[b * ai->n * ai->n],1,ai->n,ai->n);
              
        // Apply mask if available (assume mask is broadcastable to the shape)
        apply_mask(dot, &ai->mask[b * ai->n],ai->n);

        // printf("dot after mask is\n");
        // print3DArrayAs2D(&dot[b * ai->n * ai->n],1,ai->n,ai->n);

        // Apply softmax on the dot product for the current batch
         softmax_new(dot, ai->n, ai->n, &SA[b * ai->n * ai->n]);

            free(K_T);
            free(dot);
            free(dot1);
     }


 
}

void Self_Attention(AttentionInputs *ai) {
    float *SA = (float *)malloc(ai->batch_size * ai->n * ai->n * sizeof(float));
    ScaleAttention(ai, SA);

        // printf("Sa is\n");
        // print3DArrayAs2D_FLOAT(SA,ai->batch_size,ai->n,ai->n);

    for (int b = 0; b < ai->batch_size; ++b) {
        // Pointers to the current batch
        float *SA_batch = SA + b * ai->n * ai->n;
        float *V_batch = ai->V + b * ai->n * ai->dv;
        float *Z_batch = ai->Z + b * ai->n * ai->dv;

        // Perform matrix multiplication for the current batch
        matrixMultiply(SA_batch, V_batch, Z_batch, ai->n, ai->n, ai->dv);
    }
        //  printf("Attention is\n");
        //  print3DArrayAs2D(ai->Z,ai->batch_size,ai->n,ai->dv);
        free(SA);
}

////////////////////////////// COMPUTE Q K V /////////////////////////////////////

void linear_transform_manual(float* input_matrix, int batch_size, int input_rows, int input_cols,
                     float* weight_matrix, int weight_rows, int weight_cols,
                     float* bias_vector, float* output_matrix) {
    // Ensure dimensions match
    if (input_cols != weight_rows) {
        fprintf(stderr, "Error: Incompatible dimensions for matrix multiplication\n");
        exit(1);
    }

    // Perform matrix multiplication and add bias for each sample in the batch
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < input_rows; ++i) {
            for (int j = 0; j < weight_cols; ++j) {
                float sum = 0.0f;
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

void linear_transform(float* input_matrix, int batch_size, int input_rows, int input_cols,
                      float* weight_matrix, int weight_rows, int weight_cols,
                      float* bias_vector, float* output_matrix) {
    // Ensure dimensions match
    if (input_cols != weight_rows) {
        fprintf(stderr, "Error: Incompatible dimensions for matrix multiplication\n");
        exit(1);
    }
    
     
    // Loop over each sample in the batch
    for (int b = 0; b < batch_size; ++b) {
        // Perform matrix multiplication for this batch
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    input_rows, weight_cols, input_cols,
                    1.0f, input_matrix + b * input_rows * input_cols, input_cols,
                    weight_matrix, weight_cols,
                    0.0f, output_matrix + b * input_rows * weight_cols, weight_cols);
        
        // Add bias if provided
        if (bias_vector != NULL) {
            for (int i = 0; i < input_rows; ++i) {
                for (int j = 0; j < weight_cols; ++j) {
                    output_matrix[b * input_rows * weight_cols + i * weight_cols + j] += bias_vector[j];
                }
            }
        }
    }
}


typedef struct {
    //inputs//
    float *input_x;
    float *W_k;
    float *W_v;
    float *W_q;
    float *K_biases;
    float *V_biases;
    float *Q_biases;
  
    //parameters//
    int batch_size;
    int num_hidden;
    int num_sequence;
    int dq; // dq = dk
    int dv;

    //outputs//
    float *K;
    float *Q;
    float *V;
} Linear_Inputs;

// Function to initialize Linear_Inputs structure without copying data
void initLinearInputs(Linear_Inputs *li, float *input_x, float *W_k, float *W_v, float *W_q, 
                      float *K_biases, float *V_biases, float *Q_biases, 
                     int batch_size, int num_hidden, int num_sequence, int dq, int dv) {
    li->input_x = input_x;
    li->W_k = W_k;
    li->W_v = W_v;
    li->W_q = W_q;
    li->K_biases = K_biases;
    li->V_biases = V_biases;
    li->Q_biases = Q_biases;

    li->batch_size = batch_size;
    li->num_hidden = num_hidden;
    li->num_sequence = num_sequence;
    li->dq = dq;
    li->dv = dv;

    // Allocate memory for result matrix
    li->K = (float *)malloc(batch_size * num_sequence * dq * sizeof(float));
    li->Q = (float *)malloc(batch_size * num_sequence * dq * sizeof(float)); 
    li->V = (float *)malloc(batch_size * num_sequence * dv * sizeof(float));
}

// Function to free the memory allocated for AttentionInputs
void freeLinearnputs(Linear_Inputs *LI ) {
       free(LI->Q);
       free(LI->K);
       free(LI->V);
}

void L_Trasforms( Linear_Inputs *LI  ) {  //ptoduce Q,K,V
     linear_transform(LI->input_x,LI->batch_size,LI->num_sequence,LI->num_hidden,LI->W_k,LI->num_hidden,LI->dq,LI->K_biases,LI->K);
     linear_transform(LI->input_x,LI->batch_size,LI->num_sequence,LI->num_hidden,LI->W_q,LI->num_hidden,LI->dq,LI->Q_biases,LI->Q);
     linear_transform(LI->input_x,LI->batch_size,LI->num_sequence,LI->num_hidden,LI->W_v,LI->num_hidden,LI->dv,LI->V_biases,LI->V);

 }

 ////////////////////// One-Head////////////////////////

 void head(
    //inputs/
    float *input_x,
    float *mask, 
    float *Q,
    float *K,
    float *V,

    //inputs dimesnions//
    int batch_size,
    int num_sequence,
    int num_hidden,
    int dq,
    int dv,
    int p,

    //output//
    float *output // or just flot *output and copy the Z to ouput with for loop (trade-off)
     ) { 

  
    //self-attention layer,
    AttentionInputs ai;
    initAttentionInputs(&ai,Q, K, V, mask,batch_size, num_sequence, dq, dv, p);
    Self_Attention(&ai);
    //  *output = ai.Z ; //copy the pointer  

    // // Copy the contents of ai.Z to the output buffer (float *output) considering batch size
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < num_sequence; ++i) {
            for (int j = 0; j < dv; ++j) {
                output[b * num_sequence * dv + i * dv + j] = ai.Z[b * num_sequence * dv + i * dv + j];
            }
        }
    }
    
        // printf("Z output\n");
    // print3DArrayAs2D(ai.Z,batch_size,num_sequence,dv);
}

/////////////////////Multi-Head Attention///////////////////////////////////////////////////////////////////////////////


/// @brief 
/// @param input_x input_x[batch_size][num_tokens][embedding_size]
/// @param W_k W_k[embedding_size][dk]
/// @param W_q W_q[embedding_size][dq]
/// @param W_v W_v[embedding_size][dv]
/// @param W_out W_out[dv][d_out]
/// @param K_biases K_biases[dk] 
/// @param V_biases V_biases[dv]
/// @param Q_biases Q_biases[dq]
/// @param Out_biases Out_biases[d_out]
/// @param mask [batch_size][num_heads][num_tokens][num_tokens] 
/// @param num_heads 
/// @param num_tokens The number of tokens in each sequence.
/// @param embedding_size The embedding dimension of each token.
/// @param dk 
/// @param dv 
/// @param d_out same as embedding_size
/// @param batch_size The number of sequences in a batch. 
/// @param p 
/// @param Out  batch_size][num_tokens][embedding_size] 

/// @brief 
/// @param input_x input_x[batch_size][num_tokens][embedding_size]
/// @param W_k W_k[embedding_size][dk]
/// @param W_q W_q[embedding_size][dq]
/// @param W_v W_v[embedding_size][dv]
/// @param W_out W_out[dv][d_out]
/// @param K_biases K_biases[dk] 
/// @param V_biases V_biases[dv]
/// @param Q_biases Q_biases[dq]
/// @param Out_biases Out_biases[d_out]
/// @param mask [batch_size][num_heads][num_tokens][num_tokens] 
/// @param num_heads 
/// @param num_tokens The number of tokens in each sequence.
/// @param embedding_size The embedding dimension of each token.
/// @param dk 
/// @param dv 
/// @param d_out same as embedding_size
/// @param batch_size The number of sequences in a batch. 
/// @param p 
/// @param Out  batch_size][num_tokens][embedding_size] 
void  Multi_head_Attention( 
    //inputs//
    float *input_x,  
    float *W_k,      
    float *W_q,      
    float *W_v,      
    float *W_out,   

    float *K_biases, 
    float *V_biases, 
    float *Q_biases, 
    float *Out_biases, 
    float *mask,  
    
    //input parameters//
    int num_heads,
    int num_tokens,
    int embedding_size,
    int dk,
    int dv,
    int d_out, 
    int batch_size,
    int p, 

    //output//
    float *Out 
     ) 
    { 
    
    // Define dimensions - Assuming embedding_size is divisible by num_heads (head_dim)
    int Query_Size= dk/ num_heads; 
    int Value_Size= dv/num_heads; 

    if (dk % num_heads != 0) {
        fprintf(stderr, "Error: dk is not divisible by num_heads\n");
        exit(EXIT_FAILURE);
    }

    if (dv % num_heads != 0) {
        fprintf(stderr, "Error: dv is not divisible by num_heads\n");
         exit(EXIT_FAILURE);
    }

     if (d_out != embedding_size) {
        fprintf(stderr, "Error: d_out must be same as embedding_Size\n");
         exit(EXIT_FAILURE);
    }

    float *Wq_T = (float *)malloc(dk * embedding_size*sizeof(float));
    float *Wk_T = (float *)malloc(dk * embedding_size *sizeof(float));
    float *Wv_T=  (float *)malloc(dv * embedding_size*sizeof(float));

    transpose(W_q, Wq_T, embedding_size, dk);
    transpose(W_k, Wk_T, embedding_size, dk);
    transpose(W_v, Wv_T, embedding_size, dv);

    //Compute Q ,K,V  Linear Projections 
     Linear_Inputs li;
     initLinearInputs(&li, input_x, Wk_T,Wv_T,Wq_T, K_biases, V_biases, Q_biases,batch_size,embedding_size,num_tokens,dk,dv);
     L_Trasforms(&li);   

     free(Wk_T);
     free(Wv_T);
     free(Wq_T);

    float *Q_split = (float *)malloc(dk * batch_size * num_tokens*sizeof(float));
    float *K_split = (float *)malloc(dk * batch_size * num_tokens*sizeof(float));
    float *V_split = (float *)malloc(dv * batch_size * num_tokens*sizeof(float));

    //Splitting Q, K, and V Into Their Heads
    split_QKV_for_heads(li.K,K_split,batch_size,num_tokens,dk, num_heads);
    split_QKV_for_heads(li.Q,Q_split,batch_size,num_tokens,dk, num_heads);
    split_QKV_for_heads(li.V,V_split,batch_size,num_tokens,dv, num_heads);
 
    //allocate memory for every output head  
    float *Concat_out = (float *)malloc(batch_size*num_tokens*d_out*sizeof(float));

    // Split for Q, K, V and call the head function
    float *head_outs = (float *)malloc( num_tokens*batch_size*Value_Size* sizeof(float));
    float *Outs = (float *)malloc( num_tokens*batch_size *dv* sizeof(float));

    for(int i = 0; i < num_heads; i++){   
       // printf("Head %d \n",i);
        head(input_x, mask +i*batch_size*num_tokens*num_tokens,
               Q_split + i * batch_size* num_tokens*Query_Size,
               K_split + i * batch_size* num_tokens*Query_Size,
               V_split + i *  batch_size*num_tokens*Value_Size,
             batch_size, num_tokens, embedding_size, Query_Size, Value_Size, p,
             head_outs); 
//  //Concatenate the outputs for all batches --smart way
    // for (int b = 0; b < batch_size; b++) {
    //     for (int t = 0; t < num_tokens; t++) {
    //         for (int v = 0; v < Value_Size; v++) {  
    //             int ind1=b * num_tokens * dv + t * dv + i * Value_Size + v;
    //             int ind2= b * num_tokens * Value_Size + t * Value_Size + v;
    //             printf("%d\n",ind2);
    //             Concat_out[ind1] = head_outs[ind2];
    //         }
    //     }
    // }
    
        for(int j =0;j<num_tokens*Value_Size*batch_size;j++) { 
            Outs[j + i*num_tokens*Value_Size*batch_size] = head_outs[j];
        }   

        free(head_outs);
    } 
        // printf("Outs:\n"); 
        // print3DArrayAs2D(Outs,batch_size,num_tokens,dv);
    combine_heads(Outs,Concat_out,batch_size,num_heads,num_tokens,Value_Size);
        // printf("Cocnat Out:\n");
        // print3DArrayAs2D(Concat_out,batch_size,num_tokens,dv);
            
        //Apply linear transformation
          float *Wo_T=  (float *)malloc(d_out*dv*sizeof(float));        
          transpose(W_out, Wo_T, d_out, dv);
          linear_transform(Concat_out,batch_size,num_tokens,dv,Wo_T,dv,d_out,Out_biases,Out);

    free(Q_split);
    free(K_split);
    free(V_split);
    free(Concat_out);    
    free(Wo_T);  
    freeLinearnputs(&li);
}

/// @brief 
/// @param input_X [batch_size][num_tokens][embedding_size]
/// @param W1 [embedding_size][hidden_size1]
/// @param b1  [hidden_size1]
/// @param W2  [hidden_size1][hidden_size2]
/// @param b2  [hidden_size2]
/// @param Out  [batch_size][num_tokens][hidden_size2] ,hidden_size2 --> potentially back to embedding_size
/// @param p 
void FFN(float* input_X, 
        int batch_size,int num_tokens,int embedding_size,
        float *W1,      
        int hidden_size1, 
        float* b1,    
        float*W2,      
        int hidden_size2,
        float* b2, 
        float *Out 
        ,int p){ 
         
        float *W1_T = (float *)malloc(hidden_size1 * embedding_size*sizeof(float));
        float *W2_T = (float *)malloc(hidden_size2 * hidden_size1*sizeof(float));
        float *out1 = (float *)malloc(batch_size*num_tokens*hidden_size1*sizeof(float)); 

        transpose(W1, W1_T, embedding_size, hidden_size1);
        transpose(W2, W2_T, hidden_size2, embedding_size);

        linear_transform(input_X,batch_size,num_tokens,embedding_size,W1_T,embedding_size,hidden_size1,b1,out1);
        // printf("First linear\n");
        // print3DArrayAs2D(out1,batch_size,num_tokens,hidden_size1);
        relu(out1,batch_size*num_tokens*hidden_size1);
        // printf("Relu l\n");
        // print3DArrayAs2D(out1,batch_size,num_tokens,hidden_size1);
        // apply_dropout(out1,);
        linear_transform(out1,batch_size,num_tokens,hidden_size1,W2_T,hidden_size1,hidden_size2,b2,Out); 
        // printf("Second linear\n");
        // print3DArrayAs2D(Out,batch_size,num_tokens,hidden_size2);

        free(W1_T);
        free(W2_T);
        free(out1);  
} 

// Function to calculate the mean of an array
float calculate_mean(float* array, int length) {
    float sum = 0.0;
    for (int i = 0; i < length; i++) {
        sum += array[i];
    }
    return sum / length;
}

// Function to calculate the standard deviation of an array
float calculate_std(float* array, int length, float mean) {
    float sum = 0.0;
    for (int i = 0; i < length; i++) {
        sum += (array[i] - mean) * (array[i] - mean);
    }
    // Use Bessel's correction for sample standard deviation
    return sqrt(sum / (length - 1));  // Note: length - 1 for sample std
}

void Layer_Norm(
    float* x ,int batch_size,int num_tokens,int embedding_size,
    float* gamma, //[embedding_size]
    float* beta,  //[embedding_size]
    float epsilon,
    float* output ) { 

    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < num_tokens; t++) {

                float* token_embedding = &x[(b * num_tokens + t) * embedding_size];
                float mean = calculate_mean(token_embedding, embedding_size);
                float std = calculate_std(token_embedding, embedding_size, mean);

                for (int i = 0; i < embedding_size; i++) {
                    output[(b * num_tokens + t) * embedding_size + i] =
                    gamma[i] * ((token_embedding[i] - mean) / (std + epsilon)) + beta[i];
            }
        }
    }

    }


/// @param x output of the previus layer / output after Add&Norm layer too 
/// @param residual input to the layer
void Add_and_Norm(float* x,float* residual,int d1,int d2,int d3,float* gamma,float*beta,float epsilon ){ 
   
   
    //Add operation 
    for(int i=0;i<d1*d2*d3;i++) residual[i] = x[i] +residual[i];
    
    //Normalazation operation  
    Layer_Norm(residual,d1,d2,d3,gamma,beta,epsilon,x);
}


void encoder_block(
    float* input ,int batch_size,int num_tokens ,int embedding_size,
    float* mask,
    //weights for multi-head//
    float* Wq ,float* Q_bias,int dq, 
    float* Wk,float* K_Bias,
    float* Wv,float* V_bias,int dv,
    float* W_out,float* Out_bias,   
    //weights for FFN//
    float* W1,float* b1,int hidden_size_1,
    float* W2,float* b2,int hidden_size_2,
    //ADD and Norm//
    float* gamma_1,float* beta_1,float epsilon_1,
    float* gamma_2,float* beta_2,float epsilon_2,
    //parameters//
    int num_heads,
    int p,
    float* Out) {
     
   //  float *Out1 = (float *)malloc(batch_size * num_tokens * embedding_size*sizeof(float));
       clock_t start, end;
    double cpu_time_used1;

            start = clock();
     Multi_head_Attention(input,Wk,Wq,Wv,W_out,K_Bias,V_bias,Q_bias,Out_bias,mask,num_heads,num_tokens,embedding_size,dq,dv,embedding_size,batch_size,p,Out);
             end = clock();                  
            cpu_time_used1 = ((double)(end - start)) / CLOCKS_PER_SEC;
            printf("Time taken for MHA: %f seconds\n", cpu_time_used1);    
            save_array_to_file(Out, batch_size*num_tokens*embedding_size, "tensor/mha_out.txt");  
           // Print the output after Multi-Head Attention
            // printf("MHA out is:\n");
            // print3DArrayAs2D_FLOAT(Out1, batch_size, num_tokens, embedding_size);
              start = clock();

    Add_and_Norm(Out,input,batch_size,num_tokens,embedding_size,gamma_1,beta_1,epsilon_1);
             end = clock();                  
            cpu_time_used1 = ((double)(end - start)) / CLOCKS_PER_SEC;
            printf("Time taken for fist Add&Norm: %f seconds\n", cpu_time_used1);  
            //  printf("MHA out after Add&Norm is:\n");
            // print3DArrayAs2D_FLOAT(Out1, batch_size, num_tokens, embedding_size);
            
            start = clock();
     FFN(Out,batch_size,num_tokens,embedding_size,W1,hidden_size_1,b1,W2,hidden_size_2,b2,input,p);
            end = clock();                  
            cpu_time_used1 = ((double)(end - start)) / CLOCKS_PER_SEC;
            printf("Time taken for FFN : %f seconds\n", cpu_time_used1);  
            //Print the output after the Feed Forward Network
            //printf("FFN out is:\n");
           //print3DArrayAs2D_FLOAT(Out, batch_size, num_tokens, hidden_size_2);

            start = clock();
    Add_and_Norm(input,Out,batch_size,num_tokens,embedding_size,gamma_2,beta_2,epsilon_2);
            end = clock();                  
            cpu_time_used1 = ((double)(end - start)) / CLOCKS_PER_SEC;
            printf("Time taken for second Add&Norm : %f seconds\n", cpu_time_used1);  
            // printf("After FFN and add&Norm\n");
            // print3DArrayAs2D_FLOAT(Out,batch_size,num_tokens,embedding_size);

    // free(Out1);

    }

    void encoder_stack(
    float* input, int batch_size, int num_tokens, int embedding_size,
    float* mask,
    // Weights for all encoder blocks
    float** Wq_list, float** Q_bias_list, int dq, 
    float** Wk_list, float** K_Bias_list,
    float** Wv_list, float** V_bias_list, int dv,
    float** W_out_list, float** Out_bias_list,   
    float** W1_list, float** b1_list, int hidden_size_1,
    float** W2_list, float** b2_list, int hidden_size_2,
    float** gamma_1_list, float** beta_1_list, float epsilon_1,
    float** gamma_2_list, float** beta_2_list, float epsilon_2,
    int num_heads, int p,
    int num_layers, // Number of encoder blocks
    float* final_output) 
    {
    
    float* current_input = input;
    float* current_output = (float*)malloc(batch_size * num_tokens * embedding_size * sizeof(float));

    for (int i = 0; i < num_layers; i++) {
        // Call encoder_block for each layer
        encoder_block(
            current_input, batch_size, num_tokens, embedding_size, mask,
            Wq_list[i], Q_bias_list[i], dq,
            Wk_list[i], K_Bias_list[i],
            Wv_list[i], V_bias_list[i], dv,
            W_out_list[i], Out_bias_list[i],
            W1_list[i], b1_list[i], hidden_size_1,
            W2_list[i], b2_list[i], hidden_size_2,
            gamma_1_list[i], beta_1_list[i], epsilon_1,
            gamma_2_list[i], beta_2_list[i], epsilon_2,
            num_heads, p, current_output
        );

        // Prepare for the next layer
        if (i < num_layers - 1) {
            // The output of the current layer becomes the input for the next
            current_input = current_output;
            // Allocate new space for the next output
            current_output = (float*)malloc(batch_size * num_tokens * embedding_size * sizeof(float));
        }
    }

    // Copy the final output
    memcpy(final_output, current_output, batch_size * num_tokens * embedding_size * sizeof(float));

    // Free the last allocated memory
    free(current_output);
}


int main() {


     #define BATCH_SIZE 1
    #define NUM_TOKENS 128

    #define EMBEDDING_SIZE 512
    #define D_OUT 512 // must be same as EMBEDDING_SIZE
    #define DK 512
    #define DV 512
    #define HIDDEN_SIZE_1 512
    #define HIDDEN_SIZE_2 512

    #define NUM_HEADS 1

     
      // Save epsilon values separately
    FILE *file_e = fopen("tensor/Dimensions.txt", "w");
    if (file_e != NULL) {
        fprintf(file_e, "Batch Size: %d\n", BATCH_SIZE);
        fprintf(file_e, "num tokens: %d\n", NUM_TOKENS);
        fprintf(file_e, "Embedding Size: %d\n", EMBEDDING_SIZE);
        fprintf(file_e, "DK: %d\n", DK); 
        fprintf(file_e, "DV: %d\n", DV); 
        fprintf(file_e, "Hidden size 1: %d\n", HIDDEN_SIZE_1); 
        fprintf(file_e, "num of heads: %d\n", NUM_HEADS); 
        fclose(file_e);
    } else {
        fprintf(stderr, "Error: Could not open file for writing epsilon values\n");
    }

    // Allocate memory for the arrays
    static float input_tensor[BATCH_SIZE * NUM_TOKENS * EMBEDDING_SIZE];
    static float Out[BATCH_SIZE * NUM_TOKENS * EMBEDDING_SIZE];
    static float mask[BATCH_SIZE * NUM_TOKENS];
    static float W_q[EMBEDDING_SIZE * DK];
    static float W_k[EMBEDDING_SIZE * DK];
    static float W_v[EMBEDDING_SIZE * DV];
    static float W_out[DV * D_OUT];
    static float W_q_bias[DK];
    static float W_k_bias[DK];
    static float W_v_bias[DV];
    static float W_out_bias[D_OUT];
    static float W_1[EMBEDDING_SIZE * HIDDEN_SIZE_1];
    static float W_2[HIDDEN_SIZE_2 * HIDDEN_SIZE_1];
    static float b1[EMBEDDING_SIZE];
    static float b2[HIDDEN_SIZE_2];
    static float gamma_1[EMBEDDING_SIZE];
    static float gamma_2[EMBEDDING_SIZE];
    static float beta_1[EMBEDDING_SIZE];
    static float beta_2[EMBEDDING_SIZE];
    static float epsilon_1 = 0.0f; 
    static float epsilon_2 = 0.0f;

        // Check if memory allocation was successful
    if (!input_tensor  || !mask || !Out ||
        !W_q || !W_k || !W_v || !W_out ||
        !W_q_bias || !W_k_bias || !W_v_bias || !W_out_bias ||
        !W_1 || !W_2 || !b1 || !b2 ||
        !gamma_1 || !gamma_2 || !beta_1 || !beta_2) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return 1;
   }

    // //  Generate random values for matrices
    generate_random_matrix(W_q, EMBEDDING_SIZE * DK);
    generate_random_matrix(W_k, EMBEDDING_SIZE * DK);
    generate_random_matrix(W_v, EMBEDDING_SIZE * DV);
    generate_random_matrix(W_out, DV * D_OUT);

    generate_random_matrix(W_q_bias, DK);
    generate_random_matrix(W_k_bias, DK);
    generate_random_matrix(W_v_bias, DV);
    generate_random_matrix(W_out_bias, D_OUT);

    generate_random_matrix(W_1, EMBEDDING_SIZE * HIDDEN_SIZE_1);
    generate_random_matrix(W_2, HIDDEN_SIZE_2*HIDDEN_SIZE_1);
    generate_random_matrix(b2, HIDDEN_SIZE_2);
    generate_random_matrix(b1, HIDDEN_SIZE_1);
    generate_random_matrix(input_tensor,BATCH_SIZE* NUM_TOKENS * EMBEDDING_SIZE);

    generate_random_matrix(gamma_1, EMBEDDING_SIZE);
    generate_random_matrix(gamma_2, EMBEDDING_SIZE);
    generate_random_matrix(beta_1, EMBEDDING_SIZE);
    generate_random_matrix(beta_2, EMBEDDING_SIZE);
    epsilon_1 = (float)rand() / RAND_MAX;
    epsilon_2 = (float)rand() / RAND_MAX;

    generate_random_matrix(input_tensor, BATCH_SIZE * NUM_TOKENS * EMBEDDING_SIZE);
    //generate_random_matrix(mask, BATCH_SIZE * NUM_HEADS *  NUM_TOKENS * NUM_TOKENS);
    generate_random_mask(BATCH_SIZE*NUM_TOKENS,mask);
 
   
    // // Save arrays to files
    save_array_to_file(W_q, EMBEDDING_SIZE * DK, "tensor/W_q.txt");
    save_array_to_file(W_k, EMBEDDING_SIZE * DK, "tensor/W_k.txt");
    save_array_to_file(W_v, EMBEDDING_SIZE * DV, "tensor/W_v.txt");
    save_array_to_file(W_out, DV * D_OUT, "tensor/W_out.txt");

    save_array_to_file(W_q_bias, DK, "tensor/W_q_bias.txt");
    save_array_to_file(W_k_bias, DK, "tensor/W_k_bias.txt");
    save_array_to_file(W_v_bias, DV, "tensor/W_v_bias.txt");
    save_array_to_file(W_out_bias, D_OUT, "tensor/W_out_bias.txt");

    save_array_to_file(W_1, EMBEDDING_SIZE * HIDDEN_SIZE_1, "tensor/W_1.txt");
    save_array_to_file(W_2, HIDDEN_SIZE_1 * HIDDEN_SIZE_2, "tensor/W_2.txt");
    save_array_to_file(b1, HIDDEN_SIZE_1, "tensor/b1.txt");
    save_array_to_file(b2, HIDDEN_SIZE_2, "tensor/b2.txt");

    save_array_to_file(input_tensor, BATCH_SIZE * NUM_TOKENS * EMBEDDING_SIZE, "tensor/input_tensor.txt");
   //save_array_to_file(mask, BATCH_SIZE * NUM_HEADS * NUM_TOKENS * NUM_TOKENS, "tensor/mask.txt");
    save_array_to_file(mask, BATCH_SIZE *  NUM_TOKENS, "tensor/mask.txt");

    save_array_to_file(gamma_1, EMBEDDING_SIZE, "tensor/gamma_1.txt");
    save_array_to_file(gamma_2, EMBEDDING_SIZE, "tensor/gamma_2.txt");
    save_array_to_file(beta_1, EMBEDDING_SIZE, "tensor/beta_1.txt");
    save_array_to_file(beta_2, EMBEDDING_SIZE, "tensor/beta_2.txt");

    // Save epsilon values separately
    FILE *file_epsilon = fopen("tensor/epsilon_values.txt", "w");
    if (file_epsilon != NULL) {
        fprintf(file_epsilon, "epsilon_1: %f\n", epsilon_1);
        fprintf(file_epsilon, "epsilon_2: %f\n", epsilon_2);
        fclose(file_epsilon);
    } else {
        fprintf(stderr, "Error: Could not open file for writing epsilon values\n");
    }
   
    clock_t start, end;
    double cpu_time_used;

    int memory_start = print_memory_usage();
    start = clock();
    //Call the encoder block
    encoder_block(input_tensor, BATCH_SIZE, NUM_TOKENS, EMBEDDING_SIZE, mask,
                   W_q, W_q_bias, DK, W_k, W_k_bias, W_v, W_v_bias, DV, W_out, W_out_bias,
                   W_1, b1, HIDDEN_SIZE_1, W_2, b2, HIDDEN_SIZE_2,
                   gamma_1, beta_1, epsilon_1, gamma_2, beta_2, epsilon_2,
                      NUM_HEADS , 0, Out);
     end = clock();                  
    int memory_end = print_memory_usage();
    int memory = end-start;
    printf("Memory usage: %int KB\n", memory);

      save_array_to_file(input_tensor, BATCH_SIZE*NUM_TOKENS*EMBEDDING_SIZE, "tensor/Out.txt"); 
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken for encoder-block: %f seconds\n", cpu_time_used);     
         
    //Print the results
    // printf("Output:\n");
    // print3DArrayAs2D_FLOAT(Out,BATCH_SIZE,NUM_TOKENS,D_OUT);
    printf("C ENCODER-BLOCK ... DONE\n");
    return 0;
}

 













 