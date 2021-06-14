import torch
import torch.nn.functional as F
import random

def AUGMENT(x, aug='', diff=False):
    if aug != 'noaug':
        for p in policy.split('_'):
            for f in AUGMENT_FNS[p]:
                x = f(x, diff=diff)
        x = x.contiguous()
    return x

def mask(x, prob=0.1,diff=False):
    if diff:
        batch, seq_len, vocab = x.shape
        mask = torch.ones(batch, seq_len, dtype=x.dtype, device=x.device)
        mask_tokens = torch.zeros(batch, seq_len, vocab, dtype=x.dtype, device=x.device)
        num_mask = int(seq_len * prob)
        idx_list = [int(i) for i in range(1,seq_len)] # prevent initial token from masking.
        for bidx in range(batch):
            idx_mask = random.sample(idx_list, num_mask)
            for midx in idx_mask:
                mask[bidx][midx] = 0
                mask_tokens[bidx][midx][4658] = 1
        x = x * mask.unsqueeze(-1) + mask_tokens * (1- mask.unsqueeze(-1))
    else:
        if x.requires_grad:
            print("x has requires grad. something wrong!")
            x = x.detach()
        batch, seq_len = x.shape
        num_mask = int(seq_len * prob)
        idx_list = [int(i) for i in range(1,seq_len)] # prevent initial token from masking.
        for bidx in range(batch):
            idx_mask = random.sample(idx_list, num_mask)
            for midx in idx_mask:
                x[bidx][midx] = 4658
    return x 
    
def rand(x, prob=0.1,diff=False):
    if diff:
        batch, seq_len, vocab = x.shape
        mask = torch.ones(batch, seq_len, dtype=x.dtype, device=x.device)
        mask_tokens = torch.zeros(batch, seq_len, vocab, dtype=x.dtype, device=x.device)
        num_mask = int(seq_len * prob)
        idx_list = [int(i) for i in range(1,seq_len)] # prevent initial token from masking.
        token_list = [int(i) for i in range(1,4658)]
        for bidx in range(batch):
            idx_mask = random.sample(idx_list, num_mask)
            for midx in idx_mask:
                mask[bidx][midx] = 0
                mask_tokens[bidx][midx][random.choice(token_list)] = 1
        x = x * mask.unsqueeze(-1) + mask_tokens * (1- mask.unsqueeze(-1))
    else:
        if x.requires_grad:
            print("x has requires grad. something wrong!")
            x = x.detach()
        batch, seq_len = x.shape
        num_mask = int(seq_len * prob)
        idx_list = [int(i) for i in range(1,seq_len)] # prevent initial token from masking.
        token_list = [int(i) for i in range(1,4658)]
        for bidx in range(batch):
            idx_mask = random.sample(idx_list, num_mask)
            for midx in idx_mask:
                x[bidx][midx] = random.choice(token_list)
    return x 

def swap(x, diff=False):
    if diff:
        x_detach = x.detach()
        batch, seq_len, vocab = x.shape
        mask = torch.ones(batch, seq_len, dtype=x.dtype, device=x.device)
        mask_tokens = torch.zeros(batch, seq_len, vocab, dtype=x.dtype, device=x.device)
        num_mask = int(seq_len * prob)
        idx_list = [int(i) for i in range(1,seq_len)] # prevent initial token from masking.
        token_list = [int(i) for i in range(1,4658)]
        for bidx in range(batch):
            i1, i2 = random.sample(idx_list, 2)
            mask[bidx][i1] = 0
            mask[bidx][i2] = 0
            mask_tokens[bidx][i1] = x_detach[bidx][i1]
            mask_tokens[bidx][i2] = x_detach[bidx][i2]
        x = x * mask.unsqueeze(-1) + mask_tokens * (1- mask.unsqueeze(-1))        
    else:
        if x.requires_grad:
            print("x has requires grad. something wrong!")
            x = x.detach()
        batch, seq_len = x.shape
        num_mask = int(seq_len * prob)
        idx_list = [int(i) for i in range(1,seq_len)] # prevent initial token from masking.
        token_list = [int(i) for i in range(1,4658)]
        for bidx in range(batch):
            i1, i2 = random.sample(idx_list, 2)
            tmp = x[bidx][i1]
            x[bidx][i1] = x[bidx][i2]
            x[bidx][i2] = tmp

AUGMENT_FNS = {
    'mask': [mask],
    'rand': [rand],
    'swap' : [swap],
}