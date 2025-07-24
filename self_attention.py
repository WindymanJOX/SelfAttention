import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, dim_token, heads, dim_head = None):
        super().__init__()
        self.dim_token = dim_token
        self.heads = heads
        assert (dim_token % heads) == 0, 'hidden dimension must be divisible by number of heads'
        self.dim_head = dim_head if dim_head is not None else (dim_token // heads)
        self.dim_hidden = heads * dim_head

        self.W = nn.Linear(dim_token, 3*self.dim_hidden)
        self.out = nn.Linear(self.dim_hidden, dim_token)

    def forward(self, x):
        # x[b, n, dim_token]
        # Q, K, V [b, n, dim_hidden]
        b, n, dim_token = x.shape
        Q, K, V = self.W(x).chunk(3, dim=-1)
        split_heads = lambda x: x.reshape(b, n, -1, self.dim_head).transpose(1,2)
        # q, k, v [b, h, n, dim_head]
        q, k, v = map(split_heads, (Q, K, V))
        # q*k^T / sqrt(dim_head)
        # dots[n,n]
        dots = torch.einsum('bhik,bhjk->bhij', q, k)*(self.dim_head ** -0.5)
        dots = torch.softmax(dots, dim=-1)
        # out[b,h,n,dim_head]
        out = torch.einsum('bhij,bhjk->bhik', dots, v)
        # out[b,n,dim_hidden]
        out = out.transpose(1,2).reshape(b, n, -1)
        out = self.out(out)

        return out


if __name__ == '__main__':
    sattn = SelfAttention(32, 4, 64)    
    data = torch.rand(2, 128, 32)
    out = sattn(data)
    print(out.shape)