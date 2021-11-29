from typing import List

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
import torch.nn.functional as F

def format_pytorch_version(version):
    return version.split('+')[0]

TORCH_version = torch.__version__
TORCH = format_pytorch_version(TORCH_version)

def format_cuda_version(version):
    return 'cu' + version.replace('.', '')

CUDA_version = torch.version.cuda
CUDA = format_cuda_version(CUDA_version)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def lengths_to_idx(lengths):
    """
    [1, 2] into [0, 1, 1]
    """
    idx = []
    for i, length in enumerate(lengths):
        idx += [i] * length
    return torch.LongTensor(idx).to(device)

def expand_by_lengths(x, lengths):
    """
    Args:
        x: (bs, d)
        lengths: (bs, )
    Returns: (n*t, d)
    """
    # (n*t, )
    idx = lengths_to_idx(lengths)
    # (n*t, d)
    out = x[idx]
    return out

def chunk_pad_by_lengths(x, lengths, batch_first = True):
    """
    Zero padding
    Args:
        x: (n*t, d)
    Returns:
        (t, n, d) if not batch_first
        (n, t, d) if batch_first
    """
    x = x.split(lengths.tolist(), 0) # x = x.split(lengths.tolist(), 0)
    x = nn.utils.rnn.pad_sequence(x, batch_first=batch_first)
    return x

def flat_by_lengths(x, lengths):
    """
    Args:
        x: (n, t, d)
    Returns:
        out: (n*t, d) ## same tensor with x
    """
    mask = ~make_pad_mask(lengths) ## shape: (n, t)
    x = x.flatten(0, 1) ## shape: (n*t, d)
    mask = mask.flatten() ## shape: (n*t)
    out = x[mask].contiguous()
    return out

def flat_by_lengths_max_t(x, lengths, max_t):
    """
    Args:
        x: (n, t, d)
    Returns:
        out: (n*t, d) ## same tensor with x
    """
    mask = ~make_pad_mask_max_t(lengths, max_t) ## shape: (n, t)
    x = x.flatten(0, 1) ## shape: (n*t, d)
    mask = mask.flatten() ## shape: (n*t)
    out = x[mask].contiguous()
    return out

def cdist_vary_lengths(A, A_n, B, B_n, p=2):
    """
    Args:
        A: (n, dim)
        A_n: (bs)
        B: (m, dim)
        B_n: (bs)
    """
    # (bs, t1, dim)
    A = chunk_pad_by_lengths(A, A_n, batch_first=True).type(torch.float32)
    # (bs, t2, dim)
    B = chunk_pad_by_lengths(B, B_n, batch_first=True).type(torch.float32)
    # (bs, t1, t2)
    dists = torch.cdist(A, B, p=p)
    return dists

def make_pad_mask(lengths):
    """
    Args:
        lengths: (bs, 1) or (bs)
    Returns:
        mask: (bs, max_t)
    """
    bs = lengths.shape[0]
    max_t = lengths.max()
    pad_mask = torch.arange(0, max_t).expand(bs, -1).to(device) ### cuda
    pad_mask = pad_mask >= lengths.reshape((-1, 1))
    return pad_mask

def make_pad_mask_max_t(lengths, max_t):
    """
    Args:
        lengths: (bs, 1) or (bs)
    Returns:
        mask: (bs, max_t)
    """
    bs = lengths.shape[0]
    pad_mask = torch.arange(0, max_t).expand(bs, -1).to(device) ### cuda
    pad_mask = pad_mask >= lengths.reshape((-1, 1))
    return pad_mask

def vector_reject_gcr(vec, reject_vec, safe_conf):
    """
    Remove reject vector from vec only if they form obtuse angle and fars.
    Args:
        vec: (n, d)
        reject_vec: (n, d)
    Returns:
        vec: (n, d)
    """
    norm_rej = F.normalize(reject_vec, dim=-1) ## unit vector
    prod = (vec * norm_rej).sum(dim=-1, keepdim=True) ## dot product
    # if obtuse => reject the gradient 
    proj_vec = norm_rej * torch.where(prod < 0, prod, torch.zeros_like(prod))
    
    safe_radius = vec.norm(dim=-1, keepdim=True) * safe_conf
    rej_norm = reject_vec.norm(dim=-1, keepdim=True)
    # if within safe_radius, don't reject
    proj_vec = torch.where(rej_norm < safe_radius, torch.zeros_like(proj_vec), proj_vec)
    
    vec = vec - proj_vec
    return vec

def batch_hungarian_gcr(safe_coef:float = 0):
    
    class Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, GT, len_GT, Pred, len_pred):
            assert all(
                n_gt <= n_pred for n_gt, n_pred in zip(len_GT, len_pred)
            ), f'there must be more predictions than the ground truths'
            
            with torch.no_grad():
                dists = cdist_vary_lengths(GT, len_GT, Pred, len_pred)
                # replace padding with infinity
                for i, (n_a, n_b) in enumerate(zip(len_GT, len_pred)):
                    dists[i, n_a:, :] = float('inf')
                    dists[i, :, n_b:] = float('inf')
            
            pred_offset = 0
            gt_offset = 0
            cols = []
            for i, (dist, n_gt, n_pred) in enumerate(zip(dists, len_GT, len_pred)):
                cost = dist[:n_gt, :n_pred].cpu().detach().numpy() ### cuda ()
                
                if np.any(np.isnan(cost)):
                    print('cost:', cost)
                    raise ValueError('cost matrix contains nan')
                
                row, col = linear_sum_assignment(cost)
                col = torch.LongTensor(col).to(device) ### cuda
                cols.append(pred_offset + col)
                
                pred_offset += n_pred
                gt_offset += n_gt
            
            cols = torch.cat(cols)
            ctx.save_for_backward(cols, Pred, GT)
            return Pred[cols], GT, Pred, cols
        
        @staticmethod
        def backward(ctx, pred_task_grad, gt_latent_grad, pred_latent_grad, *args):
            cols, Pred, GT = ctx.saved_tensors
            
            pred_task_grad_align = torch.zeros_like(Pred)
            gt_task_grad = torch.zeros_like(GT)
            
            pred_task_grad_align[cols] = pred_task_grad
            gt_task_grad = pred_task_grad.clone()
            
            inv_gt_latent_grad = torch.zeros_like(pred_latent_grad)
            inv_gt_latent_grad[cols] = gt_latent_grad
            pred_task_grad_align = vector_reject_gcr(
                pred_task_grad_align,
                pred_latent_grad - inv_gt_latent_grad, safe_coef)
            gt_task_grad = vector_reject_gcr(
                gt_task_grad, gt_latent_grad - pred_latent_grad[cols],
                safe_coef)
            
            pred_grad = pred_task_grad_align + pred_latent_grad
            gt_grad = gt_task_grad + gt_latent_grad
            return (gt_grad, None, pred_grad, None, None, None)
    return Fn.apply