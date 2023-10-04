# -*- coding: utf-8 -*-

# @File    : learn_pytorch.py
# @Date    : 2023-09-26
# @Author  : ${ RenJin}
# import torch
# from torch import nn
# data = torch.arange(20).reshape(4, 5)
# devices = [torch.device('cuda:0'), torch.device('cuda:0')]
# split = nn.parallel.scatter(data, devices)
# print('input :', data)
# print('load into', devices)
# print('output:', split)

import torch
import torch.nn as nn
#
# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    devices = ["cuda:0", "cuda:1"]  # Use two GPUs
else:
    devices = ["cpu"]
#
# # Create some example matrices
input_size = 4
matrix_a = torch.randint(0,2,(1,input_size, input_size),dtype=torch.float16).to(devices[0])
matrix_b = torch.randint(0,2,(1,input_size, input_size),dtype=torch.float16).to(devices[0])

# Define a simple model for matrix multiplication
class MatrixMultiplyModel(nn.Module):
    def __init__(self):
        super(MatrixMultiplyModel, self).__init__()

    def forward(self, x, y):
        print(x.shape,y.shape)
        # print(x,y)
        return torch.matmul(x, y)

# Create an instance of the model
model = MatrixMultiplyModel()

# Wrap the model with DataParallel to use multiple GPUs
if len(devices) > 1:
    model = nn.DataParallel(model, device_ids=[0, 0])

# Perform matrix multiplication
result = model(matrix_a, matrix_b)

# Print the result
print(result)


