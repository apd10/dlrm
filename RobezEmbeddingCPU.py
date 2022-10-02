from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torch.nn.parameter import Parameter
import math
import pdb
import time


class RobezFunction(torch.autograd.Function):
    #hashing = []
    #lookup = []
    #other = []

    @staticmethod
    def forward(ctx, hashed_weights, indices, embedding_dim, val_offset, P, A, B, C, hashed_weights_size, helper_E1sR, helper_Eidx_base, helper_Eidx_offset, robez_chunk_size, sparse):
        assert(indices.dim() == 1) # indices has tobe a one dimensional array of integers.
        # universal hashing
        #hashed_idx = ((((((indices.view(-1,1) + val_offset) * helper_E1sR) %P  + helper_Eidx_base * B) %P  + A) % P) % (hashed_weights_size -robez_chunk_size +1) + helper_Eidx_offset)
        #hashed_idx = ((((((indices.view(-1,1) + val_offset) * helper_E1sR)  + helper_Eidx_base * B) + A) % P) % (hashed_weights_size -robez_chunk_size +1) + helper_Eidx_offset)

        #ts1 = time.perf_counter()
        hashed_idx = ((((((indices.view(-1,1) + val_offset) * helper_E1sR)  + helper_Eidx_base * B) + A) % P) % (hashed_weights_size -robez_chunk_size +1) + helper_Eidx_offset)
        #ts2 = time.perf_counter()
        output = hashed_weights[hashed_idx]
        #ts3 = time.perf_counter()
        ctx.save_for_backward(indices, hashed_idx)
        ctx.hashed_weights_size = hashed_weights_size
        ctx.sparse = sparse
        #ts4 = time.perf_counter()
        #RobezFunction.hashing.append(ts2 - ts1)
        #RobezFunction.lookup.append(ts3 - ts2)
        #RobezFunction.other.append(ts4 - ts3)
        return output


    @staticmethod
    def backward(ctx, grad):
        indices, hashed_idx = ctx.saved_variables
        hashed_weights_size = ctx.hashed_weights_size
        hashed_idx1 = hashed_idx.reshape(-1)
        grad1 = grad.reshape(-1)
        if ctx.sparse:
            unique, inv_idx = torch.unique(hashed_idx1, return_inverse=True)
            values = torch.zeros(unique.shape, device=indices.device, dtype=torch.float32)
            values.scatter_add_(0,inv_idx, grad1)
            weight_grad = torch.sparse_coo_tensor(unique.view(1, -1), values, (ctx.hashed_weights_size,), device=indices.device)
        else:
            weight_grad = torch.zeros((hashed_weights_size,),dtype=torch.float32, device=indices.device) 
            weight_grad.scatter_add_(0, hashed_idx1, grad1)
        return weight_grad, None, None, None, None, None, None, None, None, None, None, None, None, None

class RobezEmbedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        _weight: torch.Tensor,
        val_offset: int,
        robez_chunk_size = 1,
        sparse = False,
        seed = 1024)->None:

        super(RobezEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.val_offset = val_offset
        self.seed = seed
        self.weight = _weight # add to parameter
        self.weights_size = self.weight.numel()
        self.robez_chunk_size = robez_chunk_size
        self.sparse = sparse


        r = np.random.RandomState(seed)
        random_numbers = np.concatenate([np.array([2038074743]), r.randint(0, 2038074743, (10,))]) # 10 random numbers
        random_numbers = torch.from_numpy(random_numbers.astype(np.int64))
        #print("[Seed]", seed, "First 5 random numbers: ", random_numbers[:5])
        #print("Robez Embedding Object: num_embeddings:{} dim:{} val_offset:{} seed:{} weights_size:{} robez_chunk_size:{} sparse:{}".format(self.num_embeddings, self.embedding_dim,
        #                  self.val_offset, self.seed, self.weights_size, self.robez_chunk_size, self.sparse), flush=True)

        # helpers to compute
        helper_Eidx_base = torch.LongTensor(np.arange(self.embedding_dim) / self.robez_chunk_size)
        helper_Eidx_offset = torch.LongTensor(np.arange(self.embedding_dim) % self.robez_chunk_size) 
        helper_E1sR = torch.LongTensor(np.ones(self.embedding_dim) * int(random_numbers[3])) # A

        # adding to parameters
        self.random_numbers = nn.Parameter(random_numbers, requires_grad=False)
        self.helper_Eidx_base = nn.Parameter(helper_Eidx_base, requires_grad=False)
        self.helper_Eidx_offset = nn.Parameter(helper_Eidx_offset, requires_grad=False)
        self.helper_E1sR = nn.Parameter(helper_E1sR, requires_grad=False)


    def forward(self, indices: torch.Tensor, optional_tensor=None, per_sample_weights=None) -> torch.Tensor:

        #def forward(ctx, hashed_weights, indices, embedding_dim, val_offset, P, A, B, hashed_weights_size, helper_E1sR, helper_Eidx):
        embeddings =  RobezFunction.apply(
            self.weight,
            indices,
            self.embedding_dim,
            self.val_offset,
            self.random_numbers[0],
            self.random_numbers[1],
            self.random_numbers[2],
            self.random_numbers[3],
            self.weights_size,
            self.helper_E1sR,
            self.helper_Eidx_base,
            self.helper_Eidx_offset,
            self.robez_chunk_size,
            self.sparse
        )
        return embeddings
