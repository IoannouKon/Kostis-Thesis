#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
// #ifndef BAREMETAL
// #include <sys/mman.h> 
// #endif
#include <stdlib.h>
#include <assert.h>
// #include <cblas.h> //TODO to be doned 
#include <sys/resource.h>
#include <unistd.h> 
#include <stddef.h> 
#include <stdint.h>

#define SAVE_FILES (0)

    typedef float elem_t;

    #define BATCH_SIZE 1
    #define NUM_TOKENS 20
    #define EMBEDDING_SIZE 20
    #define D_OUT 20 // must be same as EMBEDDING_SIZE
    #define DK 20
    #define DV 20
    #define HIDDEN_SIZE_1 20
    #define HIDDEN_SIZE_2 20
    #define NUM_HEADS 10


////////////////// DEBUG FUNCTIONS ////////////////////////////////////////////

uint64_t read_cycles() {
    uint64_t cycles; 
    asm volatile ("rdcycle %0" : "=r" (cycles)); 
    return cycles; 
}


bool are_matrices_equal(const elem_t *matrix1, const elem_t *matrix2, int size) {
    for (int i = 0; i < size; ++i) {
        if (matrix1[i] != matrix2[i]) {
            return false; 
        }
    }
    return true; 
}

void generate_random_matrix(elem_t *matrix, int size) {
    for (int i = 0; i < size; ++i) { 
        matrix[i] = (elem_t)rand()/ RAND_MAX; ;
    }
}

void generate_random_mask(int size, elem_t *mask) {
    int total_elements = size;

    for (int i = 0; i < total_elements; i++) {
        mask[i] =  rand() % 2;
    }
}


void save_array_to_file(elem_t *array, int size, const char *filename) {
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


//////////////////////////////// Matrice computations FUNCTIONS //////////////////////////////

elem_t my_exp(elem_t x) { //TODO to be removed
    const int n = 20;  
    elem_t sum = 1.0f;  
    elem_t term = 1.0f; 

    for (int i = 1; i < n; ++i) {
        term *= x / i;  
        sum += term;    
    }
    return sum;
}

void matrixMultiply(elem_t *A, elem_t *B, elem_t *C, int rowA, int colA, int colB) {
    for (int i = 0; i < rowA; ++i) {
        for (int j = 0; j < colB; ++j) {
            C[i * colB + j] = 0;
            for (int k = 0; k < colA; ++k) {
                C[i * colB + j] += A[i * colA + k] * B[k * colB + j];
            }
        }
    }
}

void transpose_manual(elem_t *matrix, elem_t *transposed, int rows, int columns) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            transposed[j * rows + i] = matrix[i * columns + j];
        }
    }
}

void transpose_in_place(elem_t *matrix, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            elem_t temp = matrix[i * n + j];
            matrix[i * n + j] = matrix[j * n + i];
            matrix[j * n + i] = temp;
        }
    }
}

void Scale_2D(elem_t *A, int n, int dk) {
    elem_t sqrt_dk = sqrt(dk);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i * n + j] /= sqrt_dk;
        }
    }
}

void apply_mask(elem_t* dot, elem_t* mask,int num_tokens) {
    for(int i =0;i<num_tokens;i++){ 
        for (int j = 0; j < num_tokens; j++) {
            dot[i*num_tokens+j] = dot[i*num_tokens+j] - 1e6 * (1 - mask[j]);
        }
    }        

}

void apply_dropout(elem_t *matrix, int rows, int cols, elem_t p) {
    elem_t scale = 1.0 / (1.0 - p); 

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if ((rand() / (elem_t)RAND_MAX) < p) {
                matrix[i * cols + j] = 0.0; 
            } else {
                matrix[i * cols + j] *= scale; 
            }
        }
    }
}

void softmax_new(elem_t *matrix, int rows, int cols, elem_t *result) {
    for (int i = 0; i < rows; ++i) {
        elem_t max = matrix[i * cols];
        // Find the max value in the row
        for (int j = 1; j < cols; ++j) {
            if (matrix[i * cols + j] > max) {
                max = matrix[i * cols + j];
            }
        }

        elem_t sum_exp = 0.0f;
        for (int j = 0; j < cols; ++j) {
            result[i * cols + j] = my_exp(matrix[i * cols + j] - max);
            sum_exp += result[i * cols + j];
        }

        for (int j = 0; j < cols; ++j) {
            result[i * cols + j] /= sum_exp;
        }
    }
}

void split_QKV_for_heads(elem_t *Wk, elem_t *Wk_splits, int batch_size, int embedding_size, int dk, int num_heads) {
    int head_dim = dk / num_heads;
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int i = 0; i < embedding_size; ++i) {
                for (int j = 0; j < head_dim; ++j) {
                    int out_idx = b * num_heads * embedding_size * head_dim + h * embedding_size * head_dim + i * head_dim + j;
                    int in_idx = b * embedding_size * dk + i * dk + h * head_dim + j;
                    Wk_splits[out_idx] = Wk[in_idx];
                }
            }
        }
    }
}

void transpose_full(elem_t *X, elem_t *X_t, int batch_size, int num_heads, int num_tokens, int head_dim) {
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

void combine_heads(elem_t *X, elem_t *X_out, int batch_size, int num_heads, int num_tokens, int head_dim) {
    int embed_dim = num_heads * head_dim;
    elem_t X_t[batch_size * num_tokens * num_heads * head_dim ] ; 
    transpose_full(X, X_t, batch_size, num_heads, num_tokens, head_dim);
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < num_tokens; ++t) {
            for (int e = 0; e < embed_dim; ++e) {
                X_out[b * num_tokens * embed_dim + t * embed_dim + e] = 
                    X_t[(b * num_tokens + t) * embed_dim + e];
            }
        }
    }
}

void relu(elem_t* input, int size) {
    for (int i = 0; i < size; i++) {
        input[i] = (input[i] > 0) ? input[i] : 0;
    }
}

// ////////////////////////////////////////////COMPUTE SELF-ATTENTION //////////////////////////////////

// Define the AttentionInputs struct
typedef struct {
    elem_t *Q;    //Q[n][dq]
    elem_t *K;    //K[n][dk] where dk=dq
    elem_t *V;    //V[n][dv]
    elem_t *mask;   
    int batch_size;
    int n;       // number of input tokens   
    int dk;      
    int dv;
    elem_t p;     //propabily for bernulli 
    elem_t *Z;    //output matrice 
} AttentionInputs;

// this not copy the data just give the pointers as initiallize 
void initAttentionInputs(AttentionInputs *ai, elem_t *Q, elem_t *K, elem_t *V, elem_t *mask, int batch_size,int n, int dk, int dv, elem_t p) {
    ai->Q = Q;
    ai->K = K;
    ai->V = V;
    ai->mask = mask;
    ai->batch_size =batch_size;
    ai->n = n;
    ai->dk = dk;
    ai->dv = dv;
    ai->p = p;
}

void Self_Attention(AttentionInputs *ai) {
    // Allocate memory for the scaled attention matrix SA
     elem_t SA[ai->batch_size * ai->n * ai->n]; 
     elem_t K_T[ai->dk * ai->n];                
     elem_t dot[ai->n* ai->n]; 
     elem_t Z_static[ai->batch_size * ai->n * ai->dv];      

    for (int b = 0; b < ai->batch_size; ++b) {
        // Transpose K matrix
        transpose_manual(&ai->K[b * ai->n * ai->dk], K_T, ai->n, ai->dk);

        // Compute Q * K^T
        matrixMultiply(&ai->Q[b * ai->n * ai->dk], K_T, dot, ai->n, ai->dk, ai->n);

        // Scale the dot product results
        Scale_2D(dot, ai->n, ai->dk);

        // Apply the mask
        apply_mask(dot, &ai->mask[b * ai->n], ai->n);

        // Apply softmax to obtain the attention scores
        softmax_new(dot, ai->n, ai->n, &SA[b * ai->n * ai->n]);

        // Perform matrix multiplication for the current batch
        elem_t *V_batch = ai->V + b * ai->n * ai->dv;
        elem_t *Z_batch = Z_static + b * ai->n * ai->dv;
        matrixMultiply(&SA[b * ai->n * ai->n], V_batch, Z_batch, ai->n, ai->n, ai->dv);
    }
    ai->Z = Z_static;

}

////////////////////////////// COMPUTE Q K V /////////////////////////////////////

void linear_transform(elem_t* input_matrix, int batch_size, int input_rows, int input_cols,
                     elem_t* weight_matrix, int weight_rows, int weight_cols,
                     elem_t* bias_vector, elem_t* output_matrix) {

    // Perform matrix multiplication and add bias for each sample in the batch
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
    //inputs//
    elem_t *input_x;
    elem_t *W_k;
    elem_t *W_v;
    elem_t *W_q;
    elem_t *K_biases;
    elem_t *V_biases;
    elem_t *Q_biases;
  
    //parameters//
    int batch_size;
    int num_hidden;
    int num_sequence;
    int dq; // dq = dk
    int dv;

    //outputs//
    elem_t *K;
    elem_t *Q;
    elem_t *V;
} Linear_Inputs;


// Function to initialize Linear_Inputs structure without copying data
void initLinearInputs(Linear_Inputs *li, elem_t *input_x, elem_t *W_k, elem_t *W_v, elem_t *W_q, 
                      elem_t *K_biases, elem_t *V_biases, elem_t *Q_biases, 
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

}


void L_Trasforms( Linear_Inputs *LI  ) {  //ptoduce Q,K,V 

    static elem_t static_K[BATCH_SIZE * NUM_TOKENS* DK];
    static elem_t static_Q[BATCH_SIZE * NUM_TOKENS* DK];
    static elem_t static_V[BATCH_SIZE * NUM_TOKENS* DV];

    // Fill in static matrices with data
    linear_transform(LI->input_x, LI->batch_size, LI->num_sequence, LI->num_hidden, LI->W_k, LI->num_hidden, LI->dq, LI->K_biases, static_K);
    linear_transform(LI->input_x, LI->batch_size, LI->num_sequence, LI->num_hidden, LI->W_q, LI->num_hidden, LI->dq, LI->Q_biases, static_Q);
    linear_transform(LI->input_x, LI->batch_size, LI->num_sequence, LI->num_hidden, LI->W_v, LI->num_hidden, LI->dv, LI->V_biases, static_V);

    // Assign pointers to LI struct
    LI->K = static_K;
    LI->Q = static_Q;
    LI->V = static_V;

//     bool x = are_matrices_equal(LI->K,LI->Q,LI->batch_size* LI->num_sequence *LI-> dq); 
//    if (x) {
//         printf("Matrices are equal.\n");
//     } else {
//         printf("Matrices are not equal.\n");
//     }

 }

////////////////////// One-Head////////////////////////

 void head(
    //inputs/
    elem_t *input_x,
    elem_t *mask, 
    elem_t *Q,
    elem_t *K,
    elem_t *V,

    //inputs dimesnions//
    int batch_size,
    int num_sequence,
    int num_hidden,
    int dq,
    int dv,
    int p,

    //output//
    elem_t *output // or just flot *output and copy the Z to ouput with for loop (trade-off)
     ) { 

  
    //self-attention layer,
    AttentionInputs ai;
    initAttentionInputs(&ai,Q, K, V, mask,batch_size, num_sequence, dq, dv, p);
    Self_Attention(&ai);
    //  *output = ai.Z ; //copy the pointer  

    // // Copy the contents of ai.Z to the output buffer (elem_t *output) considering batch size
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < num_sequence; ++i) {
            for (int j = 0; j < dv; ++j) {
                output[b * num_sequence * dv + i * dv + j] = ai.Z[b * num_sequence * dv + i * dv + j];
            }
        }
    }
}

////////////////////Multi-Head Attention///////////////////////////////////////////////////////////////////////////////


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
void Multi_head_Attention( 
    //inputs//
    elem_t *input_x,  
    elem_t *W_k,      
    elem_t *W_q,      
    elem_t *W_v,      
    elem_t *W_out,   

    elem_t *K_biases, 
    elem_t *V_biases, 
    elem_t *Q_biases, 
    elem_t *Out_biases, 
    elem_t *mask,  
    
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
    elem_t *Out 
    ) { 
    
    // Define dimensions - Assuming embedding_size is divisible by num_heads (head_dim)
    int Query_Size = dk / num_heads; 
    int Value_Size = dv / num_heads; 

    transpose_in_place(W_q,embedding_size);
    transpose_in_place(W_k,embedding_size);
    transpose_in_place(W_v,embedding_size);

   
    // Compute Q, K, V Linear Projections 
    Linear_Inputs li;
    initLinearInputs(&li, input_x, W_k, W_v, W_q, K_biases, V_biases, Q_biases, batch_size, embedding_size, num_tokens, dk, dv);
    
    unsigned long start, end, cpu_time_used;
    start = read_cycles();
    L_Trasforms(&li);   
    end = read_cycles();
    cpu_time_used = ((end - start)) ;
    // printf("Cycles taken for produce Q,K,V: %ld \n", cpu_time_used); 

    elem_t Q_split[dk * batch_size * num_tokens];
    elem_t K_split[dk * batch_size * num_tokens];
    elem_t V_split[dv * batch_size * num_tokens];

    // Splitting Q, K, and V Into Their Heads
    start = read_cycles();
    split_QKV_for_heads(li.K, K_split, batch_size, num_tokens, dk, num_heads);
    split_QKV_for_heads(li.Q, Q_split, batch_size, num_tokens, dk, num_heads);
    split_QKV_for_heads(li.V, V_split, batch_size, num_tokens, dv, num_heads);
    end = read_cycles();
    cpu_time_used = ((end - start)) ;
    // printf("Cycles taken for split Q,K,V: %ld \n", cpu_time_used); 

    // Allocate memory for every output head  
    elem_t Concat_out[batch_size * num_tokens * d_out];
   
    start = read_cycles();
    for (int i = 0; i < num_heads; i++) {   
        head(input_x, mask,
            Q_split + i * batch_size * num_tokens * Query_Size,
            K_split + i * batch_size * num_tokens * Query_Size,
            V_split + i * batch_size * num_tokens * Value_Size,
            batch_size, num_tokens, embedding_size, Query_Size, Value_Size, p,
            Out + i * num_tokens * Value_Size * batch_size); 
    }  
    end = read_cycles(); 
    cpu_time_used = ((end - start)) ;
    // printf("Cycles taken for heads: %ld \n", cpu_time_used); 

    start = read_cycles(); 
    combine_heads(Out, Concat_out, batch_size, num_heads, num_tokens, Value_Size);
    end = read_cycles();
    cpu_time_used = ((end - start)) ;
    // printf("Cycles taken for Concat: %ld \n", cpu_time_used); 

    // Apply linear transformation      
    transpose_in_place(W_out, embedding_size);
    start = read_cycles();   
    linear_transform(Concat_out, batch_size, num_tokens, dv, W_out, dv, d_out, Out_biases, Out);
    end = read_cycles(); 
    cpu_time_used = ((end - start));
    //printf("Cycles taken for Out projection: %ld \n", cpu_time_used); 

    }


void FFN(elem_t* input_X, 
        int batch_size, int num_tokens, int embedding_size,
        elem_t *W1,      
        int hidden_size1, 
        elem_t* b1,    
        elem_t* W2,      
        int hidden_size2,
        elem_t* b2, 
        elem_t *Out,
        int p) { 
         
    // Static allocation for intermediate arrays
    elem_t W1_T[hidden_size1 * embedding_size];
    elem_t W2_T[hidden_size2 * hidden_size1];
    elem_t out1[batch_size * num_tokens * hidden_size1];

    // Transpose operations
    transpose_manual(W1, W1_T, embedding_size, hidden_size1);
    transpose_manual(W2, W2_T, hidden_size2, embedding_size);

    // First linear transformation
    linear_transform(input_X, batch_size, num_tokens, embedding_size, W1_T, embedding_size, hidden_size1, b1, out1);
    relu(out1, batch_size * num_tokens * hidden_size1);

    // Second linear transformation
    linear_transform(out1, batch_size, num_tokens, hidden_size1, W2_T, hidden_size1, hidden_size2, b2, Out);
}


// Function to calculate the mean of an array
elem_t calculate_mean(elem_t* array, int length) {
    elem_t sum = 0.0;
    for (int i = 0; i < length; i++) {
        sum += array[i];
    }
    return sum / length;
}

// Function to calculate the standard deviation of an array
elem_t calculate_std(elem_t* array, int length, elem_t mean) {
    elem_t sum = 0.0;
    for (int i = 0; i < length; i++) {
        sum += (array[i] - mean) * (array[i] - mean);
    }
    // Use Bessel's correction for sample standard deviation
    return sqrt(sum / (length - 1));  // Note: length - 1 for sample std
}

void Layer_Norm(
    elem_t* x ,int batch_size,int num_tokens,int embedding_size,
    elem_t* gamma, //[embedding_size]
    elem_t* beta,  //[embedding_size]
    elem_t epsilon,
    elem_t* output ) { 

    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < num_tokens; t++) {

                elem_t* token_embedding = &x[(b * num_tokens + t) * embedding_size];
                elem_t mean = calculate_mean(token_embedding, embedding_size);
                elem_t std = calculate_std(token_embedding, embedding_size, mean);

                for (int i = 0; i < embedding_size; i++) {
                    output[(b * num_tokens + t) * embedding_size + i] =
                    gamma[i] * ((token_embedding[i] - mean) / (std + epsilon)) + beta[i];
            }
        }
    }

    }


/// @param x output of the previus layer / output after Add&Norm layer too 
/// @param residual input to the layer
void Add_and_Norm(elem_t* x,elem_t* residual,int d1,int d2,int d3,elem_t* gamma,elem_t*beta,elem_t epsilon ){ 
   
   
    //Add operation 
    for(int i=0;i<d1*d2*d3;i++) residual[i] = x[i] +residual[i];
    
    //Normalazation operation  
    Layer_Norm(residual,d1,d2,d3,gamma,beta,epsilon,x);
}


void encoder_block(
    elem_t* input, int batch_size, int num_tokens, int embedding_size,
    elem_t* mask,
    // weights for multi-head //
    elem_t* Wq, elem_t* Q_bias, int dq,
    elem_t* Wk, elem_t* K_Bias,
    elem_t* Wv, elem_t* V_bias, int dv,
    elem_t* W_out, elem_t* Out_bias,
    // weights for FFN //
    elem_t* W1, elem_t* b1, int hidden_size_1,
    elem_t* W2, elem_t* b2, int hidden_size_2,
    // ADD and Norm //
    elem_t* gamma_1, elem_t* beta_1, elem_t epsilon_1,
    elem_t* gamma_2, elem_t* beta_2, elem_t epsilon_2,
    // parameters //
    int num_heads,
    int p
    ) {
    
    unsigned long start, end, cpu_time_used1; 
    elem_t Out[batch_size * num_tokens * embedding_size];
  
    // printf("Input :\n");
   // print3DArrayAs2D(input,batch_size,num_tokens,embedding_size);

    // Multi-Head Attention
    start = read_cycles();
    Multi_head_Attention(input, Wk, Wq, Wv, W_out, K_Bias, V_bias, Q_bias, Out_bias, mask, num_heads, num_tokens, embedding_size, dq, dv, embedding_size, batch_size, p, Out);
    end = read_cycles();                  
    cpu_time_used1 = ((end - start));
    printf("Cycles taken for MHA: %ld \n", cpu_time_used1); 
    if(SAVE_FILES){
    save_array_to_file(Out, batch_size*num_tokens*embedding_size, "tensor/mha_out.txt");  
    }

    // First Add and Norm
    start = read_cycles();
    Add_and_Norm(Out, input, batch_size, num_tokens, embedding_size, gamma_1, beta_1, epsilon_1);
    end = read_cycles();             
    cpu_time_used1 = ((end - start));
    printf("Cycles taken for first Add&Norm: %ld \n", cpu_time_used1);

    // Feed Forward Network (FFN)
    start = read_cycles();
    FFN(Out, batch_size, num_tokens, embedding_size, W1, hidden_size_1, b1, W2, hidden_size_2, b2, input, p);
    end = read_cycles();                 
    cpu_time_used1 = ((end - start));
    printf("Cycles taken for FFN: %ld \n", cpu_time_used1);

    // Second Add and Norm
    start = read_cycles();
    Add_and_Norm(input, Out, batch_size, num_tokens, embedding_size, gamma_2, beta_2, epsilon_2);
    end = read_cycles();                  
    cpu_time_used1 = ((end - start));
    printf("Cycles taken for second Add&Norm: %ld \n", cpu_time_used1);
}



int main() {
    
    // Save epsilon values separately
    if(SAVE_FILES){
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
    }

static elem_t input_tensor[BATCH_SIZE * NUM_TOKENS * EMBEDDING_SIZE];
static elem_t mask[BATCH_SIZE * NUM_TOKENS];
static elem_t W_q[EMBEDDING_SIZE * DK];
static elem_t W_k[EMBEDDING_SIZE * DK];
static elem_t W_v[EMBEDDING_SIZE * DV];
static elem_t W_out[DV * D_OUT];
static elem_t W_q_bias[DK];
static elem_t W_k_bias[DK];
static elem_t W_v_bias[DV];
static elem_t W_out_bias[D_OUT];
static elem_t W_1[EMBEDDING_SIZE * HIDDEN_SIZE_1];
static elem_t W_2[HIDDEN_SIZE_2 * HIDDEN_SIZE_1];
static elem_t b1[EMBEDDING_SIZE];
static elem_t b2[HIDDEN_SIZE_2];
static elem_t gamma_1[EMBEDDING_SIZE];
static elem_t gamma_2[EMBEDDING_SIZE];
static elem_t beta_1[EMBEDDING_SIZE];
static elem_t beta_2[EMBEDDING_SIZE];
static elem_t epsilon_1 = 0.0f; 
static elem_t epsilon_2 = 0.0f;

    // Generate random values for matrices
    generate_random_matrix(W_q, EMBEDDING_SIZE * DK);
    generate_random_matrix(W_k, EMBEDDING_SIZE * DK);
    generate_random_matrix(W_v, EMBEDDING_SIZE * DV);
    generate_random_matrix(W_out, DV * D_OUT);

    generate_random_matrix(W_q_bias, DK);
    generate_random_matrix(W_k_bias, DK);
    generate_random_matrix(W_v_bias, DV);
    generate_random_matrix(W_out_bias, D_OUT);

    generate_random_matrix(W_1, EMBEDDING_SIZE * HIDDEN_SIZE_1);
    generate_random_matrix(W_2, HIDDEN_SIZE_2 * HIDDEN_SIZE_1);
    generate_random_matrix(b2, HIDDEN_SIZE_2);
    generate_random_matrix(b1, HIDDEN_SIZE_1);
    generate_random_matrix(input_tensor, BATCH_SIZE * NUM_TOKENS * EMBEDDING_SIZE);

    generate_random_matrix(gamma_1, EMBEDDING_SIZE);
    generate_random_matrix(gamma_2, EMBEDDING_SIZE);
    generate_random_matrix(beta_1, EMBEDDING_SIZE);
    generate_random_matrix(beta_2, EMBEDDING_SIZE);

    epsilon_1 = (elem_t)rand() / RAND_MAX;
    epsilon_2 = (elem_t)rand() / RAND_MAX;

    generate_random_matrix(input_tensor, BATCH_SIZE * NUM_TOKENS * EMBEDDING_SIZE);
    generate_random_matrix(mask, BATCH_SIZE * NUM_TOKENS);
    generate_random_mask(BATCH_SIZE * NUM_TOKENS, mask);

    unsigned long start, end;
    unsigned long cpu_cycles_used;
    
    //printf("Input :\n");
    //print3DArrayAs2D(input_tensor,BATCH_SIZE,NUM_TOKENS,EMBEDDING_SIZE);

     
if(SAVE_FILES) { // Save arrays to files

    save_array_to_file(W_q, EMBEDDING_SIZE * DK, "tensor/W_q.txt");
    save_array_to_file(W_k, EMBEDDING_SIZE * DK, "tensor/W_k.txt");
    save_array_to_file(W_v, EMBEDDING_SIZE * DV, "tensor/W_v.txt");
    save_array_to_file(W_out, DV * D_OUT, "tensor/W_out.txt");
    // Save arrays to files
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
}

    // Call the encoder block 
    start = read_cycles();
    encoder_block(input_tensor, BATCH_SIZE, NUM_TOKENS, EMBEDDING_SIZE, mask,
                   W_q, W_q_bias, DK, W_k, W_k_bias, W_v, W_v_bias, DV, W_out, W_out_bias,
                   W_1, b1, HIDDEN_SIZE_1, W_2, b2, HIDDEN_SIZE_2,
                   gamma_1, beta_1, epsilon_1, gamma_2, beta_2, epsilon_2,
                   NUM_HEADS, 0);
     end = read_cycles();     
    cpu_cycles_used = ((end - start));
    printf("Cycles taken for encoder-block: %ld\n", cpu_cycles_used);
    if(SAVE_FILES) {
    save_array_to_file(input_tensor, BATCH_SIZE*NUM_TOKENS*EMBEDDING_SIZE, "tensor/Out.txt"); 
    }
    printf("C ENCODER-BLOCK ... DONE\n");
    return 0;
}

