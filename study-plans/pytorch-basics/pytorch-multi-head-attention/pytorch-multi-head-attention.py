import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Returns: None
        """
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_k = int(d_model/num_heads)
        self.W_q = nn.Parameter(torch.randn(d_model,d_model))
        self.W_k = nn.Parameter(torch.randn(d_model,d_model))
        self.W_v = nn.Parameter(torch.randn(d_model,d_model))
        self.W_o = nn.Parameter(torch.randn(d_model,d_model))

    def forward(self, Q, K, V):
        """
        Returns: output tensor
        """
        B = Q.shape[0]
        S = Q.shape[1]
        Q_h = Q@self.W_q
        K_h = K@self.W_k
        V_h = V@self.W_v
        Q_h = Q_h.view(B,S,self.num_heads,self.d_k).transpose(1,2).contiguous()
        K_h = K_h.view(B,S,self.num_heads,self.d_k).transpose(1,2).contiguous()
        V_h = V_h.view(B,S,self.num_heads,self.d_k).transpose(1,2).contiguous()
        head = torch.softmax(Q_h@K_h.transpose(-2,-1)/(self.d_k**0.5), dim = -1)@V_h
        head = head.transpose(1,2).contiguous().view(B,S,self.d_model)
        output = head@self.W_o
        return output
        
        