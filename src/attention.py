import torch
import torch.nn as nn

import numpy as np

from einops import rearrange
import time

import argparse

# Adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py 
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout= 0.):
        super().__init__()
        inner_dim = dim_head * heads

        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads 
        self.scale = dim_head ** -0.5 

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) 
        dots = dots * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = self.to_out(out)
        return out
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', default=1024, type=int)
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--dim_head', default=64, type=int)
    args = parser.parse_args()

    torch.manual_seed(1000)
    attn_module = Attention(dim=args.dim, heads=args.heads, dim_head=args.dim_head)
    attn_module.eval()

    attn_module.state_dict()['to_q.weight'].numpy().tofile('/app/attention/data/query.bin')
    attn_module.state_dict()['to_k.weight'].numpy().tofile('/app/attention/data/key.bin')
    attn_module.state_dict()['to_v.weight'].numpy().tofile('/app/attention/data/value.bin')

    attn_module.state_dict()['to_out.0.weight'].numpy().tofile('/app/attention/data/out_weight.bin')
    attn_module.state_dict()['to_out.0.bias'].numpy().tofile('/app/attention/data/out_bias.bin')
    
    # bz, seq_len, emb_dim
    input_data = torch.rand(1, 1024, 1024)

    input_data.detach().numpy().tofile('/app/attention/data/input.bin')

    start = time.time()
    output = attn_module.forward(input_data)
    end = time.time()
    print(f'[I] Forward took {(end - start) * 1000} ms' )
    

    output.detach().numpy().tofile('/app/attention/data/output.bin')
    torch.onnx.export(attn_module,               # model being run
                  input_data,                         # model input (or a tuple for multiple inputs)
                  "/app/attention/data/attention.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=14,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})