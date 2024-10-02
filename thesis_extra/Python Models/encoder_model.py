
import torch.nn as nn
import math
import time
import psutil

def load_and_reshape(filename, shape):
    array = np.loadtxt(filename, dtype=np.float32)
    return array.reshape(shape)

def calculate_similarity_score(tensor1, tensor2):
    # Cosine Similarity
   # cosine_similarity = F.cosine_similarity(tensor1, tensor2, dim=0).item()

    # Euclidean Distance (we'll convert this to a similarity score)
    euclidean_distance = torch.dist(tensor1, tensor2).item()
    max_dist = torch.sqrt(torch.sum(tensor1**2)).item() + torch.sqrt(torch.sum(tensor2**2)).item()
    euclidean_similarity = 1 - (euclidean_distance / max_dist)

    # Mean Squared Error (inverted to similarity score)
    mse = torch.mean((tensor1 - tensor2) ** 2).item()
    mse_similarity = 1 / (1 + mse)  # This gives higher similarity for smaller errors

    # Pearson Correlation Coefficient
    mean1 = torch.mean(tensor1)
    mean2 = torch.mean(tensor2)
    numerator = torch.sum((tensor1 - mean1) * (tensor2 - mean2))
    denominator = torch.sqrt(torch.sum((tensor1 - mean1) ** 2) * torch.sum((tensor2 - mean2) ** 2))
    pearson_correlation = numerator / denominator

    # Weighted average of the similarities
    similarity_score = ( euclidean_similarity + mse_similarity + pearson_correlation.item()) / 3
    similarity_score *= 100  # Convert to percentage

    return similarity_score
# Define a function to read the dimensions from the file
def read_dimensions(file_path):
    dimensions = {}
    try:
        with open(file_path, "r") as file:
            for line in file:
                # Strip any leading/trailing whitespace
                line = line.strip()
                if line:
                    # Split line into label and value
                    label, value = line.split(": ")
                    # Convert value to integer and store in dictionary
                    dimensions[label] = int(value)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
    except Exception as e:
        print(f"Error: {str(e)}")
    return dimensions

def are_tensors_similar(tensor1, tensor2, decimal_places=2):
    """
    Check if two tensors are the same up to a specified number of decimal places.

    Parameters:
    tensor1 (torch.Tensor): The first tensor.
    tensor2 (torch.Tensor): The second tensor.
    decimal_places (int): The number of decimal places to consider. Default is 2.

    Returns:
    bool: True if the tensors are the same up to the specified decimal places, False otherwise.
    """
    # Round both tensors to the specified number of decimal places
    rounded_tensor1 = torch.round(tensor1 * (10 ** decimal_places)) / (10 ** decimal_places)
    rounded_tensor2 = torch.round(tensor2 * (10 ** decimal_places)) / (10 ** decimal_places)
    
    # Compare the tensors element-wise
    are_same = torch.allclose(rounded_tensor1, rounded_tensor2, atol=10**(-decimal_places))

    return are_same

###################################################


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=True)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=True)
        self.W_out = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, X, mask=None):
        batch_size, num_tokens, embed_dim = X.size()

        # Linear projections
        Q = self.W_q(X)  # (batch_size, num_tokens, embed_dim)
        K = self.W_k(X)  # (batch_size, num_tokens, embed_dim)
        V = self.W_v(X)  # (batch_size, num_tokens, embed_dim) 

        # print("Q\n")
        # print(Q)
        # print("K\n")
        # print(K)
        # print("V\n")
        # print(V)

        # Split into multiple heads
        Q = self.split_heads(Q)  # (batch_size, num_heads, num_tokens, head_dim)
        K = self.split_heads(K)  # (batch_size, num_heads, num_tokens, head_dim)
        V = self.split_heads(V)  # (batch_size, num_heads, num_tokens, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        # print("SA is\n")
        # print(attn)

        context = torch.matmul(attn, V)  # (batch_size, num_heads, num_tokens, head_dim)
        # print("Attntion is\n")
        # print(context)

        context = self.combine_heads(context)  # (batch_size, num_tokens, embed_dim)
        # print("Context\n")
        # print(context)

        # Final linear projection
        out = self.W_out(context)  # (batch_size, num_tokens, embed_dim)
        return out

    def split_heads(self, X):
        batch_size, num_tokens, embed_dim = X.size()
        X = X.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        return X.transpose(1, 2)  # (batch_size, num_heads, num_tokens, head_dim)

    def combine_heads(self, X):
        batch_size, num_heads, num_tokens, head_dim = X.size()
        X = X.transpose(1, 2).contiguous()
        return X.view(batch_size, num_tokens, self.embed_dim)
    
# Define the input parameters
batch_size = 1
num_tokens = 5
embed_dim = 4
num_heads = 2

dk = 4
dv = 4
d_out = 4

# Initialize the MultiHeadAttention module
multi_head_attention = MultiHeadAttention(embed_dim, num_heads)

#####################################################################
#####################################################################

###################################################################################--RANDOM
# # Set random weights and biases for W_q, W_k, W_v, and W_out
# multi_head_attention.W_q.weight.data = torch.rand(( embed_dim, embed_dim))
# multi_head_attention.W_k.weight.data = torch.rand(( embed_dim, embed_dim))
# multi_head_attention.W_v.weight.data = torch.rand(( embed_dim, embed_dim)) 
# multi_head_attention.W_out.weight.data =torch.rand((embed_dim, embed_dim)) 

# multi_head_attention.W_q.bias.data = torch.rand(embed_dim)
# multi_head_attention.W_k.bias.data = torch.rand(embed_dim)
# multi_head_attention.W_v.bias.data = torch.rand(embed_dim)
# multi_head_attention.W_out.bias.data = torch.rand(embed_dim)

# # Define the input tensor
# input_tensor = torch.rand((batch_size, num_tokens, embed_dim))
# mask = torch.rand((batch_size, num_heads,num_tokens, num_tokens))

#####################################################################
#####################################################################
# # Manually set matrix values -----------------BATCH SIZE 1
# W_q_values = torch.tensor([
#     0.1, 0.2, 0.3, 0.4,
#     0.5, 0.6, 0.7, 0.8,
#     0.9, 1.0, 1.1, 1.2,
#     1.3, 1.4, 1.5, 1.6
# ]).reshape((embed_dim, dk))

# W_k_values = torch.tensor([
#     0.2, 0.4, 0.6, 0.8,
#     1.0, 1.2, 1.4, 1.6,
#     1.8, 2.0, 2.2, 2.4,
#     2.6, 2.8, 3.0, 3.2
# ]).reshape((embed_dim, dk))

# W_v_values = torch.tensor([
#     0.3, 0.6, 0.9, 1.2,
#     1.5, 1.8, 2.1, 2.4,
#     2.7, 3.0, 3.3, 3.6,
#     3.9, 4.2, 4.5, 4.8
# ]).reshape((embed_dim, dv))

# W_out_values = torch.tensor([
#     0.4, 0.8, 1.2, 1.6,
#     2.0, 2.4, 2.8, 3.2,
#     3.6, 4.0, 4.4, 4.8,
#     5.2, 5.6, 6.0, 6.4
# ]).reshape((dv, d_out))

# W_q_bias_values = torch.tensor([0.1, 0.2, 0.3, 0.4])
# W_k_bias_values = torch.tensor([0.2, 0.4, 0.6, 0.8])
# W_v_bias_values = torch.tensor([0.3, 0.6, 0.9, 1.2])
# W_out_bias_values = torch.tensor([0.4, 0.8, 1.2, 1.6])

# input_tensor_values = torch.tensor([
#     0.5, 0.5, 0.5, 0.5,
#     0.6, 0.6, 0.6, 0.6,
#     0.7, 0.7, 0.7, 0.7,
#     0.8, 0.8, 0.8, 0.8,
#     0.9, 0.9, 0.9, 0.9
# ]).reshape((batch_size, num_tokens, embed_dim))

# mask_values = torch.tensor([
#     1.0, 0.0, 0.0, 0.0, 0.0,
#     0.0, 1.0, 0.0, 0.0, 0.0,
#     0.0, 0.0, 1.0, 0.0, 0.0,
#     0.0, 0.0, 0.0, 1.0, 0.0,
#     0.0, 0.0, 0.0, 0.0, 1.0,
#     1.0, 0.0, 0.0, 0.0, 0.0,
#     0.0, 1.0, 0.0, 0.0, 0.0,
#     0.0, 0.0, 1.0, 0.0, 0.0,
#     0.0, 0.0, 0.0, 1.0, 0.0,
#     0.0, 0.0, 0.0, 0.0, 1.0
# ]).reshape((batch_size, num_heads, num_tokens, num_tokens))

# # Assuming you have a MultiHeadAttention class instance `multi_head_attention`
# # Update the weights and biases of the multi-head attention instance
# multi_head_attention.W_q.weight.data = W_q_values
# multi_head_attention.W_k.weight.data = W_k_values
# multi_head_attention.W_v.weight.data = W_v_values
# multi_head_attention.W_out.weight.data = W_out_values

# multi_head_attention.W_q.bias.data = W_q_bias_values
# multi_head_attention.W_k.bias.data = W_k_bias_values
# multi_head_attention.W_v.bias.data = W_v_bias_values
# multi_head_attention.W_out.bias.data = W_out_bias_values

# # Define the input tensor and mask
# input_tensor = input_tensor_values
# mask = mask_values

#####################################################################
#####################################################################
# # Manually set matrix values -----------------BATCH SIZE 2

import torch

# Define constants
BATCH_SIZE = 2
NUM_TOKENS = 5
NUM_HEADS = 2
EMBEDDING_SIZE = 4
DK = 4
DV = 4
D_OUT = 4

# Define weights and biases
W_q = torch.tensor([
    0.1, 0.2, 0.3, 0.4,
    0.5, 0.6, 0.7, 0.8,
    0.9, 1.0, 1.1, 1.2,
    1.3, 1.4, 1.5, 1.6
]).reshape(EMBEDDING_SIZE, DK)

W_k = torch.tensor([
    0.2, 0.4, 0.6, 0.8,
    1.0, 1.2, 1.4, 1.6,
    1.8, 2.0, 2.2, 2.4,
    2.6, 2.8, 3.0, 3.2
]).reshape(EMBEDDING_SIZE, DK)

W_v = torch.tensor([
    0.3, 0.6, 0.9, 1.2,
    1.5, 1.8, 2.1, 2.4,
    2.7, 3.0, 3.3, 3.6,
    3.9, 4.2, 4.5, 4.8
]).reshape(EMBEDDING_SIZE, DV)

W_out = torch.tensor([
    0.4, 0.8, 1.2, 1.6,
    2.0, 2.4, 2.8, 3.2,
    3.6, 4.0, 4.4, 4.8,
    5.2, 5.6, 6.0, 6.4
]).reshape(DV, D_OUT)

W_q_bias = torch.tensor([0.1, 0.2, 0.3, 0.4])
W_k_bias = torch.tensor([0.2, 0.4, 0.6, 0.8])
W_v_bias = torch.tensor([0.3, 0.6, 0.9, 1.2])
W_out_bias = torch.tensor([0.4, 0.8, 1.2, 1.6])

# Define input tensor
input_tensor = torch.tensor([
    # Batch 0
    0.5, 0.5, 0.5, 0.5,
    0.6, 0.6, 0.6, 0.6,
    0.7, 0.7, 0.7, 0.7,
    0.8, 0.8, 0.8, 0.8,
    0.9, 0.9, 0.9, 0.9,
    # Batch 1
    0.55, 0.55, 0.55, 0.55,
    0.65, 0.65, 0.65, 0.65,
    0.75, 0.75, 0.75, 0.75,
    0.85, 0.85, 0.85, 0.85,
    0.95, 0.95, 0.95, 0.95
]).reshape(BATCH_SIZE, NUM_TOKENS, EMBEDDING_SIZE)

# Define mask
mask = torch.tensor([
    # Batch 0
    1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0,
    # Batch 1
    1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0,
        # Batch 0
    1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0,
    # Batch 1
    1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0
]).reshape(BATCH_SIZE, NUM_HEADS, NUM_TOKENS, NUM_TOKENS)

#####################################################################################
#####################################################################################


multi_head_attention.W_q.weight.data = W_q
multi_head_attention.W_k.weight.data = W_k
multi_head_attention.W_v.weight.data = W_v
multi_head_attention.W_out.weight.data = W_out

multi_head_attention.W_q.bias.data = W_q_bias
multi_head_attention.W_k.bias.data = W_k_bias
multi_head_attention.W_v.bias.data = W_v_bias
multi_head_attention.W_out.bias.data = W_out_bias

# Define the input tensor and mask
input_tensor = input_tensor
mask = mask
# Forward pass
output = multi_head_attention(input_tensor,mask)
# print("Output\n")
# print(output)

###############################################################################
###############################################################################
########################### --> Produce CSV 

# import os
# import numpy as np

# # Create a directory for saving CSVs
# save_dir_csv = 'tensors_csv'
# os.makedirs(save_dir_csv, exist_ok=True)

# def save_tensor_as_csv(tensor, filename_prefix):
#     tensor_np = tensor.detach().cpu().numpy()  # Detach the tensor first
    
#     if tensor_np.ndim == 1:  # If the tensor is 1D
#         np.savetxt(os.path.join(save_dir_csv, f'{filename_prefix}.csv'), tensor_np, delimiter=',')
    
#     elif tensor_np.ndim == 2:  # If the tensor is 2D
#         np.savetxt(os.path.join(save_dir_csv, f'{filename_prefix}.csv'), tensor_np, delimiter=',')
    
#     elif tensor_np.ndim == 3:  # If the tensor is 3D
#         for i in range(tensor_np.shape[0]):
#             np.savetxt(os.path.join(save_dir_csv, f'{filename_prefix}_slice_{i}.csv'), tensor_np[i], delimiter=',')
    
#     elif tensor_np.ndim == 4:  # If the tensor is 4D
#         for i in range(tensor_np.shape[0]):
#             for j in range(tensor_np.shape[1]):
#                 np.savetxt(os.path.join(save_dir_csv, f'{filename_prefix}_slice_{i}_{j}.csv'), tensor_np[i, j], delimiter=',')
    
#     else:
#         raise ValueError(f"Unsupported tensor dimension: {tensor_np.ndim}. Only 1D, 2D, 3D, and 4D tensors are supported.")

# # Save weights, biases, and inputs as CSV files
# save_tensor_as_csv(multi_head_attention.W_q.weight.data, 'W_q_weight')
# save_tensor_as_csv(multi_head_attention.W_k.weight.data, 'W_k_weight')
# save_tensor_as_csv(multi_head_attention.W_v.weight.data, 'W_v_weight')
# save_tensor_as_csv(multi_head_attention.W_out.weight.data, 'W_out_weight')

# save_tensor_as_csv(multi_head_attention.W_q.bias.data, 'W_q_bias')
# save_tensor_as_csv(multi_head_attention.W_k.bias.data, 'W_k_bias')
# save_tensor_as_csv(multi_head_attention.W_v.bias.data, 'W_v_bias')
# save_tensor_as_csv(multi_head_attention.W_out.bias.data, 'W_out_bias')

# save_tensor_as_csv(input_tensor, 'input_tensor')
# save_tensor_as_csv(mask, 'mask')

# # Optional: Save the output tensor
# save_tensor_as_csv(output, 'output_tensor')

###############################################################################
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the FeedForward class
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    # def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
    #     """
    #     Args:
    #         `x`: shape (batch_size, max_len, d_model)

    #     Returns:
    #         same shape as input x
    #     """
    #     return self.w_2(self.dropout(F.relu(self.w_1(x))))
    def forward(self, x):
        # First linear layer
        x1 = self.w_1(x)
        # print("\nAfter first linear (W_1 * input + b1):")
        # print(x1)
        
        # ReLU activation
        x2 = F.relu(x1)
        # print("\nAfter ReLU activation:")
        # print(x2)
        
        # Dropout (optional)
        # x3 = self.dropout(x2)
        # print("\nAfter dropout:")
        # print(x3)
        
        # Second linear layer
        x4 = self.w_2(x2)
        # print("\nAfter second linear (W_2 * input + b2):")
        # print(x4)
        
        return x4

#########################################################
######################################################### --->  Test for FFN
 
# Test the FeedForward class
def test_feedforward():
    # Define dimensions
    batch_size = 1
    max_len = 3
    d_model = 4
    d_ff = 6
    dropout = 0.1

    # Create an instance of the FeedForward class
    model = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    # Generate some sample input data
    input_tensor = torch.randn(batch_size, max_len, d_model)

    #######################################################################
    # Create an instance of the FeedForward class
    model = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    # Manually set the weights and biases for W_1, W_2, b1, and b2
    model.w_1.weight.data = torch.tensor([
        [0.690001, 0.505418, 0.591490, 0.554784],
        [0.378428, 0.257732, 0.207382, 0.626261],
        [0.340127, 0.843851, 0.068778, 0.409906],
        [0.879994, 0.319480, 0.980568, 0.085004]
    ])
    model.w_1.bias.data = torch.tensor([0.857086, 0.977218, 0.889646, 0.545275])

    model.w_2.weight.data = torch.tensor([
        [0.907628, 0.102509, 0.921977, 0.507551],
        [0.872289, 0.333259, 0.692409, 0.556946],
        [0.361915, 0.031993, 0.858511, 0.098984],
        [0.877067, 0.449258, 0.432192, 0.606267],
        [0.927837, 0.664285, 0.395365, 0.438468],
        [0.652125, 0.928863, 0.949143, 0.307335]
    ])
    model.w_2.bias.data = torch.tensor([0.596482, 0.783076, 0.338030, 0.805252, 0.942196, 0.066473])

    # Create a custom input tensor
    input_tensor = torch.tensor([[
        [0.864371, 0.457772, 0.800219, 0.873705],
        [0.821818, 0.185726, 0.086265, 0.638155],
        [0.233827, 0.462721, 0.007419, 0.635010]
    ]])
    # Print weights w_1 and w_2
    # print("\nWeights w_1:")
    # print(model.w_1.weight.data)
    # print("\nBiases b_1:")
    # print(model.w_1.bias.data)
    
    # print("\nWeights w_2:")
    # print(model.w_2.weight.data)
    # print("\nBiases b_2:")
    # print(model.w_2.bias.data)


    # Print input tensor
    # print("Input Tensor:")
    # print(input_tensor)

    # Perform a forward pass
    output_tensor = model(input_tensor)

    # Print output tensor
    # print("\nOutput Tensor:")
    # print(output_tensor)

    # Verify the output shape
    #assert output_tensor.shape == (batch_size, max_len, d_model), "Output shape mismatch!"

if __name__ == "__main__":
    test_feedforward()



#########################################################################
#########################################################################  
################################# ---> Layer Nomralization
# Define the LayerNorm class as provided
class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        mean = x.mean(-1, keepdim=True)
        # print("mean :")
        # print(mean)
        std = x.std(-1, keepdim=True)
        # print(" std :")
        # print(std)
        return self.a * (x - mean) / (std + self.eps) + self.b

# Example input tensor (same as used in the C example)
batch_size = 2
num_tokens = 3
embedding_size = 4
input_tensor = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0],
    [17.0, 18.0, 19.0, 20.0],
    [21.0, 22.0, 23.0, 24.0]
], dtype=torch.float32).reshape(batch_size, num_tokens, embedding_size)

# Create a LayerNorm instance
layer_norm = LayerNorm(features=embedding_size, eps=1e-6)

# Manually set the gamma (a) and beta (b) values
layer_norm.a.data = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
layer_norm.b.data = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)


# Apply layer normalization
output = layer_norm(input_tensor)

# # Print the normalized output
# print("Normalized Output from PyTorch LayerNorm:")
# print(output)

#################################################################
#################################################################


######################## Encoder Block and Ecoder Stack ###############################
import math
import copy
from typing import Union
from typing import Optional, Tuple, Any


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


###############FFN
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            `x`: shape (batch_size, max_len, d_model)

        Returns:
            same shape as input x
        """
        #return self.w_2(self.dropout(F.relu(self.w_1(x))))
        return self.w_2(F.relu(self.w_1(x)))

############Layer Norm 
class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        # features = d_model
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b

#################Multi-Head
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor,
                mask: Optional[torch.ByteTensor] = None, dropout: Optional[nn.Dropout] = None) -> Tuple[
        torch.Tensor, Any]:
        """
        Args:
            `query`: shape (batch_size, n_heads, max_len, d_q)
            `key`: shape (batch_size, n_heads, max_len, d_k)
            `value`: shape (batch_size, n_heads, max_len, d_v)
            `mask`: shape (batch_size, 1, 1, max_len)
            `dropout`: nn.Dropout

        Returns:
            `weighted value`: shape (batch_size, n_heads, max_len, d_v)
            `weight matrix`: shape (batch_size, n_heads, max_len, max_len)
        """
        d_k = query.size(-1)  # d_k = d_model / n_heads
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # B*H*L*L
        #print("Score\n",scores)
        if mask is not None:
            # print("mask is \n",mask)
            # print("dot before mask\n",scores)

            # scores = scores.masked_fill(mask.eq(0), -1e9)
            # print("dot after mask is\n",scores) #fix mask later
            g =1
        p_attn = F.softmax(scores, dim=-1)  # B*H*L*L
        #print("Softmax is\n",p_attn)
        if dropout is not None:
            p_attn = p_attn #dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
        



class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // n_heads
        self.h = n_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.sdpa = ScaledDotProductAttention()
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor,
                mask: Optional[torch.ByteTensor] = None) -> torch.FloatTensor:
        """
        Args: 
            `query`: shape (batch_size, max_len, d_model)
            `key`: shape (batch_size, max_len, d_model)
            `value`: shape (batch_size, max_len, d_model)
            `mask`: shape (batch_size, max_len)
        
        Returns:
            shape (batch_size, max_len, d_model)
        """
        if mask is not None:
            # Same mask applied to all h heads. B*1*1*L
            mask = mask.unsqueeze(1).unsqueeze(1)
        batch_size = query.size(0)

        # # Print the actual weights of W_q, W_k, W_v
        # print("W_q weights:\n", self.linears[0].weight)
        # print("W_k weights:\n", self.linears[1].weight)
        # print("W_v weights:\n", self.linears[2].weight)

        # print("W_q bias:\n", self.linears[0].bias)
        # print("W_k bias:\n", self.linears[1].bias)
        # print("W_v bias:\n", self.linears[2].bias)

        with torch.no_grad():
            query = input_tensor 
            key = input_tensor
            value = input_tensor
       
    
        # print("Input tensor\n",query)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        
        # print("query is :",query)
        # print("key is :",key)
        # print("value is :",value)

        # 2) Apply attention on all the projected vectors in batch.
        # x: B x H x L x D_v
        x, self.attn = self.sdpa(query, key, value, mask=mask, dropout=None) #dropout=self.dropout 
        #print(x)  
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        #print("CONCAT OUT OF MHA\n",x)
        x = self.linears[-1](x)
        # print("MHA CONCAT is\n",x)
        # Example usage
        similarity_score = calculate_similarity_score(x, mha_out_c)
        if are_tensors_similar(x, mha_out_c, decimal_places=2):
             print(f"Tensors for MHA  are the same up to 2 decimal places with a similarity score of {similarity_score}%.")
        else:
            print(f"Tensors for MHA are different, similarity score is {similarity_score}%.")

        return x

###########Encoder-Layer (encoder-Block)
class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward"""

    def __init__(self, size: int, self_attn: MultiHeadAttention, feed_forward: FeedForward, dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
     with torch.no_grad():
        self.self_attn.linears[0].weight.copy_(W_q)  # W_q
        self.self_attn.linears[0].bias.copy_(W_q_bias)
        self.self_attn.linears[1].weight.copy_(W_k)  # W_k
        self.self_attn.linears[1].bias.copy_(W_k_bias)
        self.self_attn.linears[2].weight.copy_(W_v)  # W_v
        self.self_attn.linears[2].bias.copy_(W_v_bias)
        self.self_attn.linears[3].weight.copy_(W_out)  # W_out
        self.self_attn.linears[3].bias.copy_(W_out_bias)
        

        # Apply Multi-Head Attention (MHA)
        mha_out = self.self_attn(x, x, x, mask)
        #print("Multi-Head Attention Output:\n", mha_out)  # Print the output of MHA before Add & Norm


        with torch.no_grad():
             # LayerNorm parameters (gamma and beta)
             self.sublayer[0].norm.a.copy_(gamma_1)
             self.sublayer[0].norm.b.copy_(beta_1)
             self.sublayer[0].norm.eps =epsilon_1
             self.sublayer[1].norm.a.copy_(gamma_2)
             self.sublayer[1].norm.b.copy_(beta_2)
             self.sublayer[1].norm.eps =epsilon_2
             

        # Apply the first Add & Norm sublayer
        attn_output = self.sublayer[0](x, lambda x: mha_out)
        #print("After Multi-Head Attention (Add & Norm):\n", attn_output)  # Print output after Add & Norm
       

        with torch.no_grad():
             # LayerNorm parameters (gamma and beta)
             self.feed_forward.w_1.weight.copy_(W_1)
             self.feed_forward.w_2.weight.copy_(W_2)
             self.feed_forward.w_1.bias.copy_(b1)
             self.feed_forward.w_2.bias.copy_(b2)
                       

        # Apply Feed Forward Network (FFN)
        ffn_out = self.feed_forward(attn_output)
        #print("Feed Forward Network Output:\n", ffn_out)  # Print the output of FFN before Add & Norm

        # Apply the second Add & Norm sublayer
        ffn_output = self.sublayer[1](attn_output, lambda x: ffn_out)
        #print("After Feed Forward Network (Add & Norm):\n", ffn_output)  # Print output after Add & Norm

        return ffn_output


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    # def forward(self, x: torch.FloatTensor, sublayer: Union[MultiHeadAttention, FeedForward]) -> torch.FloatTensor:
    #     """Apply residual connection to any sublayer with the same size."""
    #     #return x + self.dropout(sublayer(self.norm(x)))
    #     return x + sublayer(self.norm(x))

    def forward(self, x: torch.FloatTensor, sublayer: nn.Module) -> torch.FloatTensor:
        sublayer_output = sublayer(x)                # Apply the sublayer (e.g., attention, feedforward)
        output = self.norm(x + sublayer_output)  # Add the original input (residual connection) + dropout, then normalize
        return output
    
################# Encoder Stack 
class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer: EncoderLayer, N: int):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return x #self.norm(x)


class TransformerEncoder(nn.Module):
    """The encoder of transformer

    Args:
        `n_layers`: number of stacked encoder layers
        `d_model`: model dimension
        `d_ff`: hidden dimension of feed forward layer
        `n_heads`: number of heads of self-attention
        `dropout`: dropout rate, default 0.1
    """

    def __init__(self, d_model: int, d_ff: int, n_heads: int = 1, n_layers: int = 1,
                 dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        self.multi_headed_attention = MultiHeadAttention(n_heads, d_model, dropout) 
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.encoder_layer = EncoderLayer(d_model, self.multi_headed_attention, self.feed_forward, dropout)
        self.encoder = Encoder(self.encoder_layer, n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
        return self.encoder(x, mask)

################################ Testing #######################################3

import numpy as np


# Assuming these values are consistent with the dimensions in your C code

# Read dimensions
dimensions = read_dimensions("tensor/Dimensions.txt")

# Extract values
BATCH_SIZE = dimensions.get("Batch Size", 0)
NUM_TOKENS = dimensions.get("num tokens", 0)
EMBEDDING_SIZE = dimensions.get("Embedding Size", 0)
DK = dimensions.get("DK", 0)
DV = dimensions.get("DV", 0)
D_OUT = EMBEDDING_SIZE
HIDDEN_SIZE_1 = dimensions.get("Hidden size 1", 0)
HIDDEN_SIZE_2 = EMBEDDING_SIZE
NUM_HEADS = dimensions.get("num of heads", 0)


# Load and reshape each tensor

# 3D Tensors
input_tensor = load_and_reshape("tensor/input_tensor.txt", (BATCH_SIZE, NUM_TOKENS, EMBEDDING_SIZE))
mask = load_and_reshape("tensor/mask.txt", (BATCH_SIZE, NUM_TOKENS))

# 2D Tensors
W_q = load_and_reshape("tensor/W_q.txt", (EMBEDDING_SIZE, DK))
W_k = load_and_reshape("tensor/W_k.txt", (EMBEDDING_SIZE, DK))
W_v = load_and_reshape("tensor/W_v.txt", (EMBEDDING_SIZE, DV))
W_out = load_and_reshape("tensor/W_out.txt", (DV, D_OUT))

W_1 = load_and_reshape("tensor/W_1.txt", (EMBEDDING_SIZE, HIDDEN_SIZE_1))
W_2 = load_and_reshape("tensor/W_2.txt", (HIDDEN_SIZE_1, EMBEDDING_SIZE))

# 1D Tensors (biases, gamma, beta)
W_q_bias = load_and_reshape("tensor/W_q_bias.txt", (DK,))
W_k_bias = load_and_reshape("tensor/W_k_bias.txt", (DK,))
W_v_bias = load_and_reshape("tensor/W_v_bias.txt", (DV,))
W_out_bias = load_and_reshape("tensor/W_out_bias.txt", (D_OUT,))

b1 = load_and_reshape("tensor/b1.txt", (HIDDEN_SIZE_1))
b2 = load_and_reshape("tensor/b2.txt", (HIDDEN_SIZE_2))

gamma_1 = load_and_reshape("tensor/gamma_1.txt", (EMBEDDING_SIZE,))
gamma_2 = load_and_reshape("tensor/gamma_2.txt", (EMBEDDING_SIZE,))
beta_1 = load_and_reshape("tensor/beta_1.txt", (EMBEDDING_SIZE,))
beta_2 = load_and_reshape("tensor/beta_2.txt", (EMBEDDING_SIZE,))

out_c = load_and_reshape("tensor/Out.txt", (BATCH_SIZE,NUM_TOKENS,EMBEDDING_SIZE))
mha_out_c =load_and_reshape("tensor/mha_out.txt", (BATCH_SIZE,NUM_TOKENS,EMBEDDING_SIZE))
mha_out_c = torch.tensor(mha_out_c)

# Load epsilon values from a file
with open("tensor/epsilon_values.txt", "r") as file:
    lines = file.readlines()
    epsilon_1 = float(lines[0].split(": ")[1])
    epsilon_2 = float(lines[1].split(": ")[1])

# Convert NumPy arrays to PyTorch tensors
input_tensor = torch.tensor(input_tensor)
mask = torch.tensor(mask)
W_q = torch.tensor(W_q)
W_k = torch.tensor(W_k)
W_v = torch.tensor(W_v)
W_out = torch.tensor(W_out)
W_q_bias = torch.tensor(W_q_bias)
W_k_bias = torch.tensor(W_k_bias)
W_v_bias = torch.tensor(W_v_bias)
W_out_bias = torch.tensor(W_out_bias)
W_1 = torch.tensor(W_1)
W_2 = torch.tensor(W_2)
b1 = torch.tensor(b1)
b2 = torch.tensor(b2)
gamma_1 = torch.tensor(gamma_1)
gamma_2 = torch.tensor(gamma_2)
beta_1 = torch.tensor(beta_1)
beta_2 = torch.tensor(beta_2)
out_c = torch.tensor(out_c)


d_model = EMBEDDING_SIZE
n_heads = NUM_HEADS
batch_size = BATCH_SIZE
max_len = NUM_TOKENS
d_ff = HIDDEN_SIZE_1
dropout = 0.1
n_layers = 1

#test_encoder():
enc = TransformerEncoder(d_model, d_ff, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
x =input_tensor
#mask =  torch.randn(batch_size, max_len).ge(0)


with torch.no_grad():
    # Assuming enc has a multi_headed_attention attribute, we can set its weights:
    enc.multi_headed_attention.linears[0].weight.copy_(W_q)  # W_q
    enc.multi_headed_attention.linears[0].bias.copy_(W_q_bias)
    
    enc.multi_headed_attention.linears[1].weight.copy_(W_k)  # W_k
    enc.multi_headed_attention.linears[1].bias.copy_(W_k_bias)
    
    enc.multi_headed_attention.linears[2].weight.copy_(W_v)  # W_v
    enc.multi_headed_attention.linears[2].bias.copy_(W_v_bias)
    
    enc.multi_headed_attention.linears[3].weight.copy_(W_out.t())  # W_out
    enc.multi_headed_attention.linears[3].bias.copy_(W_out_bias)
    
    # FeedForward weights
    enc.feed_forward.w_1.weight.copy_(W_1.t())  # W_1
    enc.feed_forward.w_1.bias.copy_(b1)
    
    enc.feed_forward.w_2.weight.copy_(W_2.t())  # W_2
    enc.feed_forward.w_2.bias.copy_(b2)
    
    # LayerNorm parameters (gamma and beta)
    enc.encoder_layer.sublayer[0].norm.a.copy_(gamma_1)
    enc.encoder_layer.sublayer[0].norm.b.copy_(beta_1)
    enc.encoder_layer.sublayer[1].norm.a.copy_(gamma_2)
    enc.encoder_layer.sublayer[1].norm.b.copy_(beta_2)

# Create a process object
process = psutil.Process()

# Start measuring CPU and memory usage
start_time = time.time()
start_cpu = process.cpu_percent(interval=None)
start_memory = process.memory_info().rss  # in bytes

# Run your encoder
out = enc(x, mask)
assert x.size() == out.size()

end_time = time.time()
end_cpu = process.cpu_percent(interval=None)
end_memory = process.memory_info().rss  # in bytes

# print("out is\n",out)
# Calculate similarity score
similarity_score = calculate_similarity_score(out, out_c)
print(f"Similarity Score for encoder Output: {similarity_score}%")

# Calculate time taken
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.6f} seconds")

# Calculate CPU and memory usage
cpu_usage = end_cpu - start_cpu
memory_usage = (end_memory - start_memory) / (1024 * 1024)  # Convert bytes to MB

print(f"CPU usage: {cpu_usage}%")
print(f"Memory usage: {memory_usage:.2f} MB")



