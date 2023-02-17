from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torch.nn.parameter import Parameter
import math
import time
import robe
import pdb

def par_idx_py(indices, dimension, chunk_size, size, A, B, C, P, helper_Eidx_base, helper_Eidx_offset, helper_E1sR):
    #helper_Eidx_base = torch.div(torch.arange(dimension, dtype=torch.int64, device=indices.device), chunk_size, rounding_mode='trunc')
    #helper_Eidx_offset = torch.arange(dimension, dtype=torch.int64, device=indices.device) % chunk_size
    #helper_E1sR = torch.ones(dimension, dtype=torch.int64, device=indices.device) * A
    hashed_idx = ((((((indices.view(-1,1)) * helper_E1sR)  + (helper_Eidx_base+1) * B) + C) %P) % (size - chunk_size) + helper_Eidx_offset)
    return hashed_idx


class RobezIDXFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, embedding_dim, val_offset, P, A, B, C, robez_chunk_size, size, use_gpu, helper_Eidx_base, helper_Eidx_offset, helper_E1sR):
        if use_gpu:
            hashed_idx = par_idx_py(indices + val_offset, embedding_dim, robez_chunk_size, size, A, B, C, P, helper_Eidx_base, helper_Eidx_offset, helper_E1sR)
        else:
            hashed_idx = robe.get_idx_s(indices + val_offset, embedding_dim, robez_chunk_size, size, A, B, C, P)
        #hashed_idx = robe.get_idx_s_power2(indices + val_offset, embedding_dim, robez_chunk_size, 21, A, B, C, P)
        return hashed_idx


    @staticmethod
    def backward(ctx, grad):
        return None, None, None, None, None, None, None, None, None, None, None, None, None

class RobezEmbedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        _weight: torch.Tensor,
        val_offset: int,
        robez_chunk_size = 1,
        sparse = False,
        seed = 1024,
        use_gpu = False)->None:

        super(RobezEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.val_offset = val_offset
        self.seed = seed
        self.weight = _weight # add to parameter
        self.weights_size = self.weight.numel()
        self.robez_chunk_size = robez_chunk_size
        self.sparse = sparse
        self.use_gpu = use_gpu


        r = np.random.RandomState(seed)
        random_numbers = np.concatenate([np.array([2038074743]), r.randint(0, 2038074743, (10,))]) # 10 random numbers
        random_numbers = torch.from_numpy(random_numbers.astype(np.int64))
        #print("[Seed]", seed, "First 5 random numbers: ", random_numbers[:5])
        #print("Robez Embedding Object: num_embeddings:{} dim:{} val_offset:{} seed:{} weights_size:{} robez_chunk_size:{} sparse:{}".format(self.num_embeddings, self.embedding_dim,
        #                  self.val_offset, self.seed, self.weights_size, self.robez_chunk_size, self.sparse), flush=True)

        # helpers to compute
        #helper_Eidx_base = torch.LongTensor(np.arange(self.embedding_dim) / self.robez_chunk_size)
        #helper_Eidx_offset = torch.LongTensor(np.arange(self.embedding_dim) % self.robez_chunk_size) 
        #helper_E1sR = torch.LongTensor(np.ones(self.embedding_dim) * int(random_numbers[3])) # A

        # adding to parameters
        self.random_numbers = nn.Parameter(random_numbers, requires_grad=False)
        #self.helper_Eidx_base = nn.Parameter(helper_Eidx_base, requires_grad=False)
        #self.helper_Eidx_offset = nn.Parameter(helper_Eidx_offset, requires_grad=False)
        #self.helper_E1sR = nn.Parameter(helper_E1sR, requires_grad=False)

        #self.table_8 = nn.Parameter(torch.randperm(2**8), requires_grad=False)
        #self.bits = int(np.log2(self.weight.numel()))
        #self.positions = nn.Parameter(torch.arange(int(self.embedding_dim / self.robez_chunk_size)), requires_grad=False)
        #self.offsets = nn.Parameter(torch.arange(int(self.robez_chunk_size)), requires_grad=False)
        #print(self.positions)

        self.helper_Eidx_base = torch.div(torch.arange(embedding_dim, dtype=torch.int64), robez_chunk_size, rounding_mode='trunc')
        self.helper_Eidx_offset = torch.arange(embedding_dim, dtype=torch.int64) % robez_chunk_size
        self.helper_E1sR = torch.ones(embedding_dim, dtype=torch.int64) * random_numbers[1]

        if use_gpu:
            self.helper_Eidx_base = self.helper_Eidx_base.to("cuda:0")
            self.helper_Eidx_offset = self.helper_Eidx_offset.to("cuda:0")
            self.helper_E1sR = self.helper_E1sR.to("cuda:0")

    def forward(self, indices: torch.Tensor, optional_tensor=None, per_sample_weights=None) -> torch.Tensor:
        idx =  RobezIDXFunction.apply(
            indices,
            self.embedding_dim,
            self.val_offset,
            self.random_numbers[0],
            self.random_numbers[1],
            self.random_numbers[2],
            self.random_numbers[3],
            self.robez_chunk_size,
            self.weights_size,
            self.use_gpu,
            self.helper_Eidx_base,
            self.helper_Eidx_offset,
            self.helper_E1sR,
        )
        #idx = torch.zeros((indices.size(0), self.embedding_dim), device=indices.device, dtype=torch.int64)
        return self.weight[idx]
