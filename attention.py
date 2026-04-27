import torch
import torch.nn as nn
import torch.nn.functional as F
import math

##
#$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
#

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        
        # 1. Compute Scores (The "Matching" process)
        # We multiply Query by Key Transpose
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 2. Apply Mask (Optional - used to hide future words)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 3. Softmax (The "Selection" process)
        # This turns scores into probabilities that sum to 1
        attention_weights = F.softmax(scores, dim=-1)
        
        # 4. Multiply by Value (The "Output" process)
        output = torch.matmul(attention_weights, v)
        
        return output, attention_weights