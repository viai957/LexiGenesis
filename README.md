# This is a implimentation of Llama 3 paper without crying
### A brief tutorial on how to implement a paper and debug your model.
### I will be updating this repo at every single stange and opensource the code for any individuals to emulate the same 
### I hope an individual one day could build an model equivalent to big AGI houses (ofcourse not the parameter equivant models) like OpenAI, Mistral, Meta .. 
### One step at a time

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
```


```python
# vocabulary length. Llama's real vocab size is 128256. Here let's just use an absurdly small number
v = 10

# Llama's maximum sequence length is 8192, but for inference they cache 3/4 of it and only use an effective length of 2048. more on that later
seq_len = 5

# we'll use a batch size of 1 for simplicity when visualizing our tensors
b = 1

# now let's make ourselves a list of token indices. Each represents somewhere between a letter and a word
tokens = torch.randint(v, (b, seq_len))
tokens.shape, tokens
```




    (torch.Size([1, 5]), tensor([[0, 3, 5, 4, 4]]))



## 1 Initilizing the first residual state


```python
# embedding dimention for toy llama 3b but Llama 3 8b uses 4096
d = 16

# initilizing token embeddings matrix
embedding = nn.Embedding(v, d)
embedding.weight.shape, embedding.weight
# each row in this embedding is high dimentinal representation of its corresponding token
```




    (torch.Size([10, 16]),
     Parameter containing:
     tensor([[-1.9523,  1.1804, -0.0998,  0.2894,  0.6317,  0.9063,  0.7657,  0.7456,
               2.1669,  0.8861,  1.0069,  1.8661,  1.3670, -0.4832,  0.9648, -0.0686],
             [ 0.2916,  2.6673,  0.3926, -0.3763, -0.3817, -0.8909,  0.8489,  1.0036,
               0.2149, -1.0360,  1.0438, -0.6540,  0.4794,  0.7502,  0.7995,  1.5562],
             [ 1.1286, -1.0695,  2.4272,  0.9768,  0.3309,  1.3637, -0.9788, -0.5404,
               0.7235,  0.1102,  0.6464, -0.1549,  1.0444,  1.8228, -0.8798,  0.0605],
             [ 0.6251, -0.1145,  0.3131,  1.2454, -0.0343, -2.0792, -2.7760, -1.1495,
              -0.3918, -1.3473,  0.2471, -0.1025, -0.2905, -0.9974,  0.0240, -0.1240],
             [ 1.2631, -0.8634, -1.5885, -0.8875, -3.2862, -1.6752,  0.7157, -1.3220,
               1.9481, -1.3358,  0.9652, -0.2016,  1.3535, -0.0838, -0.8162, -1.8322],
             [-0.3132, -1.7827, -0.4264, -0.3776,  0.6806, -1.6354, -2.8349, -0.8069,
              -0.6635,  0.1175, -0.9579, -0.8525, -1.0838,  1.7114, -1.0976,  0.1767],
             [ 0.0799,  1.3000, -0.2754, -0.3399,  0.2467, -1.6563, -0.4473, -0.7398,
               0.7428,  1.3637,  1.3537,  1.3914, -0.0813,  0.1352, -1.5494, -1.0002],
             [ 0.6338, -0.1329,  0.1626,  0.5850,  2.3522, -0.6831,  1.4146, -0.9606,
               1.0548,  0.3083,  1.5986, -0.0112, -1.0014, -0.8421, -1.4244, -0.6690],
             [ 0.4093,  0.6850, -1.7959,  0.2445,  1.4543,  1.5417, -0.2112,  0.5542,
              -1.2217,  0.3964,  0.8091, -0.1371,  1.6120, -0.0984,  0.9954, -1.2202],
             [-0.6936,  1.1523,  2.2328, -0.5796, -1.5640, -0.4542, -0.4845,  0.3693,
               0.2376, -1.5587, -0.7319,  0.5022,  0.1904,  0.8125, -0.8786,  2.3439]],
            requires_grad=True))




```python
# grambbing the embeddings that correspond to our sequence of token indices
x = embedding(tokens)
x.shape, x
# at this points many models would multiply the embeddings by the square root of the embedding dimension, but Llama 3 foregoes that strategy
```




    (torch.Size([1, 5, 16]),
     tensor([[[-1.9523,  1.1804, -0.0998,  0.2894,  0.6317,  0.9063,  0.7657,
                0.7456,  2.1669,  0.8861,  1.0069,  1.8661,  1.3670, -0.4832,
                0.9648, -0.0686],
              [ 0.6251, -0.1145,  0.3131,  1.2454, -0.0343, -2.0792, -2.7760,
               -1.1495, -0.3918, -1.3473,  0.2471, -0.1025, -0.2905, -0.9974,
                0.0240, -0.1240],
              [-0.3132, -1.7827, -0.4264, -0.3776,  0.6806, -1.6354, -2.8349,
               -0.8069, -0.6635,  0.1175, -0.9579, -0.8525, -1.0838,  1.7114,
               -1.0976,  0.1767],
              [ 1.2631, -0.8634, -1.5885, -0.8875, -3.2862, -1.6752,  0.7157,
               -1.3220,  1.9481, -1.3358,  0.9652, -0.2016,  1.3535, -0.0838,
               -0.8162, -1.8322],
              [ 1.2631, -0.8634, -1.5885, -0.8875, -3.2862, -1.6752,  0.7157,
               -1.3220,  1.9481, -1.3358,  0.9652, -0.2016,  1.3535, -0.0838,
               -0.8162, -1.8322]]], grad_fn=<EmbeddingBackward0>))



## Precompute our RoPE Frequencies


```python
theta = 10000 # 10,000 is the most common value but Llama 3 uses 50,000. In theory smaller models should use a smaller value
num_heads = 4 # Llama 3 8b has 32 total attention heads
head_dim = d // num_heads # Llama 3 ties its head dimension to the embedding dimension. This value comes out to 128 in Llama 3, which is purposeful to

freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
print(f'freqs: {freqs.shape}\n{freqs}\n')

t = torch.arange(seq_len * 2, device=freqs.device, dtype=torch.float32)
print(f't: {t.shape}\n{t}\n')

freqs = torch.outer(t, freqs)
print(f'freqs: {freqs.shape}\n{freqs}\n')

freqs_cis = torch.polar(torch.ones_like(freqs), freqs)[:seq_len]  # complex64
print(f'freqs_cis: {freqs_cis.shape}\n{freqs_cis}') 
```

    freqs: torch.Size([2])
    tensor([1.0000, 0.0100])
    
    t: torch.Size([10])
    tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    
    freqs: torch.Size([10, 2])
    tensor([[0.0000, 0.0000],
            [1.0000, 0.0100],
            [2.0000, 0.0200],
            [3.0000, 0.0300],
            [4.0000, 0.0400],
            [5.0000, 0.0500],
            [6.0000, 0.0600],
            [7.0000, 0.0700],
            [8.0000, 0.0800],
            [9.0000, 0.0900]])
    
    freqs_cis: torch.Size([5, 2])
    tensor([[ 1.0000+0.0000j,  1.0000+0.0000j],
            [ 0.5403+0.8415j,  0.9999+0.0100j],
            [-0.4161+0.9093j,  0.9998+0.0200j],
            [-0.9900+0.1411j,  0.9996+0.0300j],
            [-0.6536-0.7568j,  0.9992+0.0400j]])
    

### 1d. Precomputing the Causal Mask
<a id='d'></a>

Similar to RoPE embeddings, the causal mask is another part of the attention mechanism that we can create ahead of time to then be reused in every layer.

The basic idea of a causal mask is that by default, attention mechanisms allow every single token to pay attention to every single other token. This is okay or even preferable for some model types, but Llama is auto-regressive, meaning it would be bad if a given token to be predicted was able to see itself and future tokens during training but not during inference. The negative infinity's in the upper-triangle prevent the model from attending to the corresponding token; how this works will be more clear later when we do the attention softmax


```python
mask = torch.full(
    (seq_len, seq_len),
    float("-inf")
)
mask = torch.triu(mask, diagonal=1)
mask
```




    tensor([[0., -inf, -inf, -inf, -inf],
            [0., 0., -inf, -inf, -inf],
            [0., 0., 0., -inf, -inf],
            [0., 0., 0., 0., -inf],
            [0., 0., 0., 0., 0.]])



## Normalization (RMS Norm)
<a id='e'></a>

Root Mean Square Normalization has also been the norm for quite awhile. Like its predecessor LayerNorm, RMSNorm restricts the variability of the entries in each embedding vector such that the vector lies on a hypersphere with radius $\sqrt{d}$. However unlike LayerNorm which centers that hypersphere with a mean of zero, RMSNorm does not mess with the mean, which is an important source of data for networks that utilize residual connections.


```python
# first setup the residual connection that will be used later
h = x
print(f"h: {h.shape}\n{h}\n")
```

    h: torch.Size([1, 5, 16])
    tensor([[[-1.9523,  1.1804, -0.0998,  0.2894,  0.6317,  0.9063,  0.7657,
               0.7456,  2.1669,  0.8861,  1.0069,  1.8661,  1.3670, -0.4832,
               0.9648, -0.0686],
             [ 0.6251, -0.1145,  0.3131,  1.2454, -0.0343, -2.0792, -2.7760,
              -1.1495, -0.3918, -1.3473,  0.2471, -0.1025, -0.2905, -0.9974,
               0.0240, -0.1240],
             [-0.3132, -1.7827, -0.4264, -0.3776,  0.6806, -1.6354, -2.8349,
              -0.8069, -0.6635,  0.1175, -0.9579, -0.8525, -1.0838,  1.7114,
              -1.0976,  0.1767],
             [ 1.2631, -0.8634, -1.5885, -0.8875, -3.2862, -1.6752,  0.7157,
              -1.3220,  1.9481, -1.3358,  0.9652, -0.2016,  1.3535, -0.0838,
              -0.8162, -1.8322],
             [ 1.2631, -0.8634, -1.5885, -0.8875, -3.2862, -1.6752,  0.7157,
              -1.3220,  1.9481, -1.3358,  0.9652, -0.2016,  1.3535, -0.0838,
              -0.8162, -1.8322]]], grad_fn=<EmbeddingBackward0>)
    
    


```python
# perfroming first normalization
# first squash each entry in x and then take the mean of those values across each embedding
mean_squared = x.pow(2).mean(dim=-1,  keepdim=True)
mean_squared
```




    tensor([[[1.2922],
             [1.1588],
             [1.4290],
             [2.1207],
             [2.1207]]], grad_fn=<MeanBackward1>)




```python
# then multiply x by the recirocal of the square roots of mean_squared
# 1e-6 is very small number added for stability just in case an entry happens to be equal to 0 (since you can't divide by 0)
x_normed = x / torch.rsqrt(mean_squared + 1e-6)
print(f'x_normed: {x_normed.shape}\n{x_normed}\n')
```

    x_normed: torch.Size([1, 5, 16])
    tensor([[[-2.2193,  1.3418, -0.1134,  0.3290,  0.7180,  1.0303,  0.8704,
               0.8476,  2.4632,  1.0073,  1.1446,  2.1213,  1.5540, -0.5493,
               1.0967, -0.0779],
             [ 0.6729, -0.1232,  0.3370,  1.3407, -0.0369, -2.2382, -2.9882,
              -1.2374, -0.4218, -1.4503,  0.2660, -0.1104, -0.3127, -1.0737,
               0.0259, -0.1335],
             [-0.3744, -2.1311, -0.5097, -0.4514,  0.8135, -1.9549, -3.3889,
              -0.9645, -0.7931,  0.1405, -1.1451, -1.0191, -1.2956,  2.0458,
              -1.3121,  0.2112],
             [ 1.8394, -1.2574, -2.3133, -1.2924, -4.7855, -2.4394,  1.0422,
              -1.9252,  2.8369, -1.9453,  1.4056, -0.2936,  1.9710, -0.1220,
              -1.1886, -2.6681],
             [ 1.8394, -1.2574, -2.3133, -1.2924, -4.7855, -2.4394,  1.0422,
              -1.9252,  2.8369, -1.9453,  1.4056, -0.2936,  1.9710, -0.1220,
              -1.1886, -2.6681]]], grad_fn=<DivBackward0>)
    
    


```python
# now time to multiply the learnable parameter the gamma and beta 
# This scale is initialized to 1's but if we were to train then these values will change
rms_scale = torch.ones(d)
print(f'rms_scale: {rms_scale.shape}\n{rms_scale}\n')

x_normed *= rms_scale
print(f'x_normed: {x_normed.shape}\n{x_normed}\n')
```

    rms_scale: torch.Size([16])
    tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    
    x_normed: torch.Size([1, 5, 16])
    tensor([[[-2.2193,  1.3418, -0.1134,  0.3290,  0.7180,  1.0303,  0.8704,
               0.8476,  2.4632,  1.0073,  1.1446,  2.1213,  1.5540, -0.5493,
               1.0967, -0.0779],
             [ 0.6729, -0.1232,  0.3370,  1.3407, -0.0369, -2.2382, -2.9882,
              -1.2374, -0.4218, -1.4503,  0.2660, -0.1104, -0.3127, -1.0737,
               0.0259, -0.1335],
             [-0.3744, -2.1311, -0.5097, -0.4514,  0.8135, -1.9549, -3.3889,
              -0.9645, -0.7931,  0.1405, -1.1451, -1.0191, -1.2956,  2.0458,
              -1.3121,  0.2112],
             [ 1.8394, -1.2574, -2.3133, -1.2924, -4.7855, -2.4394,  1.0422,
              -1.9252,  2.8369, -1.9453,  1.4056, -0.2936,  1.9710, -0.1220,
              -1.1886, -2.6681],
             [ 1.8394, -1.2574, -2.3133, -1.2924, -4.7855, -2.4394,  1.0422,
              -1.9252,  2.8369, -1.9453,  1.4056, -0.2936,  1.9710, -0.1220,
              -1.1886, -2.6681]]], grad_fn=<MulBackward0>)
    
    


```python
# RMS function
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight 
```

## Initilize Multi-Query Attention
<a id='f'></a>
[multi-query attention](https://arxiv.org/abs/1911.02150) is the de facto standard for saving on parameter counts in order to get a bigger model. The idea is that the model can make multiple queries to the residual state and have those many queries be answered by shared keys & values.


```python
# x is for the residual connection and x_normed will go into our Attention calculation
h, x_normed
```




    (tensor([[[-1.9523,  1.1804, -0.0998,  0.2894,  0.6317,  0.9063,  0.7657,
                0.7456,  2.1669,  0.8861,  1.0069,  1.8661,  1.3670, -0.4832,
                0.9648, -0.0686],
              [ 0.6251, -0.1145,  0.3131,  1.2454, -0.0343, -2.0792, -2.7760,
               -1.1495, -0.3918, -1.3473,  0.2471, -0.1025, -0.2905, -0.9974,
                0.0240, -0.1240],
              [-0.3132, -1.7827, -0.4264, -0.3776,  0.6806, -1.6354, -2.8349,
               -0.8069, -0.6635,  0.1175, -0.9579, -0.8525, -1.0838,  1.7114,
               -1.0976,  0.1767],
              [ 1.2631, -0.8634, -1.5885, -0.8875, -3.2862, -1.6752,  0.7157,
               -1.3220,  1.9481, -1.3358,  0.9652, -0.2016,  1.3535, -0.0838,
               -0.8162, -1.8322],
              [ 1.2631, -0.8634, -1.5885, -0.8875, -3.2862, -1.6752,  0.7157,
               -1.3220,  1.9481, -1.3358,  0.9652, -0.2016,  1.3535, -0.0838,
               -0.8162, -1.8322]]], grad_fn=<EmbeddingBackward0>),
     tensor([[[-2.2193,  1.3418, -0.1134,  0.3290,  0.7180,  1.0303,  0.8704,
                0.8476,  2.4632,  1.0073,  1.1446,  2.1213,  1.5540, -0.5493,
                1.0967, -0.0779],
              [ 0.6729, -0.1232,  0.3370,  1.3407, -0.0369, -2.2382, -2.9882,
               -1.2374, -0.4218, -1.4503,  0.2660, -0.1104, -0.3127, -1.0737,
                0.0259, -0.1335],
              [-0.3744, -2.1311, -0.5097, -0.4514,  0.8135, -1.9549, -3.3889,
               -0.9645, -0.7931,  0.1405, -1.1451, -1.0191, -1.2956,  2.0458,
               -1.3121,  0.2112],
              [ 1.8394, -1.2574, -2.3133, -1.2924, -4.7855, -2.4394,  1.0422,
               -1.9252,  2.8369, -1.9453,  1.4056, -0.2936,  1.9710, -0.1220,
               -1.1886, -2.6681],
              [ 1.8394, -1.2574, -2.3133, -1.2924, -4.7855, -2.4394,  1.0422,
               -1.9252,  2.8369, -1.9453,  1.4056, -0.2936,  1.9710, -0.1220,
               -1.1886, -2.6681]]], grad_fn=<MulBackward0>))




```python
# let's define the hyperparameters of MQA
num_kv_heads = 2 # Llama uses 8 key and value heads per layer
assert num_heads % num_kv_heads == 0 # each q needs to match up to a kv
print(f"as a reminder: num_heads = {num_heads}, head_dim = {head_dim}")
```

    as a reminder: num_heads = 4, head_dim = 4
    


```python
# self-attention weight matrices
wq = nn.Linear(d, num_heads * head_dim, bias=False)
wk = nn.Linear(d, num_kv_heads * head_dim, bias=False)
wv = nn.Linear(d, num_kv_heads * head_dim, bias=False)
print("Attention weights: ", wq.weight.shape, wk.weight.shape, wv.weight.shape)

# and project x_normed out to get our queries, keys and values
xq = wq(x_normed)
xk = wk(x_normed)
xv = wv(x_normed)
print("Attention projections: ", xq.shape, xk.shape, xv.shape)

# then reshape them to separate out by head
xq = xq.view(b, seq_len, num_heads, head_dim)
xk = xk.view(b, seq_len, num_kv_heads, head_dim)
xv = xv.view(b, seq_len, num_kv_heads, head_dim)
print("Reshaped: ", xq.shape, xk.shape, xv.shape)
```

    Attention weights:  torch.Size([16, 16]) torch.Size([8, 16]) torch.Size([8, 16])
    Attention projections:  torch.Size([1, 5, 16]) torch.Size([1, 5, 8]) torch.Size([1, 5, 8])
    Reshaped:  torch.Size([1, 5, 4, 4]) torch.Size([1, 5, 2, 4]) torch.Size([1, 5, 2, 4])
    

## RoPE 


```python
xq = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
xk = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
print(f'xq: {xq.shape}\n{xq}\n')
print(f'xk: {xk.shape}\n{xk}')
```

    xq: torch.Size([1, 5, 4, 2])
    tensor([[[[ 0.0872-3.8958e-01j,  0.4076-1.0552e+00j],
              [-1.1601-4.4989e-01j, -1.5056+3.2144e-01j],
              [-1.0277+5.1741e-01j, -0.6346+4.4304e-01j],
              [ 0.6110+5.3759e-01j,  0.5315-7.8816e-01j]],
    
             [[-0.3966-4.2153e-01j, -0.3305-3.7835e-01j],
              [ 0.3346+9.6514e-01j, -0.4682-9.5921e-01j],
              [ 0.7337-9.6495e-01j,  0.1643+3.3382e-01j],
              [ 0.7725+1.1221e-01j, -0.7880+2.0778e-01j]],
    
             [[ 0.3630+1.0475e+00j, -0.7706-5.1892e-02j],
              [ 0.1665+8.8541e-01j,  0.0831-2.9273e-01j],
              [ 0.3958-1.0110e-03j, -0.2582+9.2673e-01j],
              [-0.1161-3.5325e-01j, -0.0788+7.3024e-01j]],
    
             [[-2.1524-3.0648e+00j,  0.7570+9.1593e-01j],
              [-0.8320+1.1806e+00j,  0.0913+8.1849e-02j],
              [ 1.3518-9.1061e-01j, -0.7899-3.0661e-01j],
              [ 2.0851+1.0060e-01j,  0.6795-1.7184e+00j]],
    
             [[-2.1524-3.0648e+00j,  0.7570+9.1593e-01j],
              [-0.8320+1.1806e+00j,  0.0913+8.1849e-02j],
              [ 1.3518-9.1061e-01j, -0.7899-3.0661e-01j],
              [ 2.0851+1.0060e-01j,  0.6795-1.7184e+00j]]]],
           grad_fn=<ViewAsComplexBackward0>)
    
    xk: torch.Size([1, 5, 2, 2])
    tensor([[[[-0.2452+0.3009j,  1.1554-0.3450j],
              [-1.4381-0.4582j,  0.9283+0.1765j]],
    
             [[ 0.2996-0.6023j, -0.4934+0.0518j],
              [ 1.8613-0.0526j, -1.0789-0.2581j]],
    
             [[ 0.0725+0.2959j, -0.7653+0.1485j],
              [ 0.9480+0.2951j, -0.3986-0.5960j]],
    
             [[-1.4577-2.4383j,  0.0307+1.3214j],
              [ 1.3563-0.8912j, -1.0929+1.7159j]],
    
             [[-1.4577-2.4383j,  0.0307+1.3214j],
              [ 1.3563-0.8912j, -1.0929+1.7159j]]]],
           grad_fn=<ViewAsComplexBackward0>)
    


```python
ndim = xq.ndim
assert 0 <= 1 < ndim
assert freqs_cis.shape == (xq.shape[1], xq.shape[-1]), f'freqs_cis.shape {freqs_cis.shape} != xq.shape[1], xq.shape[-1] {(xq.shape[1], xq.shape[-1])}'

# reshape our queries
shape = [d if i == 1 or i == xq.ndim - 1 else 1 for i, d in enumerate(xq.shape)]
print(f'shape: {shape}\n')

freqs_cis = freqs_cis.view(*shape)
print(f'freqs_cis: {freqs_cis.shape}\n{freqs_cis}')
```

    shape: [1, 5, 1, 2]
    
    freqs_cis: torch.Size([1, 5, 1, 2])
    tensor([[[[ 1.0000+0.0000j,  1.0000+0.0000j]],
    
             [[ 0.5403+0.8415j,  0.9999+0.0100j]],
    
             [[-0.4161+0.9093j,  0.9998+0.0200j]],
    
             [[-0.9900+0.1411j,  0.9996+0.0300j]],
    
             [[-0.6536-0.7568j,  0.9992+0.0400j]]]])
    


```python
# now multiply the data by the frequencies, turn them back into real numbers, revert the shape and make sure they're of the right type
xq = torch.view_as_real(xq * freqs_cis).flatten(3).type_as(xv)
xk = torch.view_as_real(xk * freqs_cis).flatten(3).type_as(xv)
print(f'xq: {xq.shape}\n{xq}\n')
print(f'xk: {xk.shape}\n{xk}')
```

    xq: torch.Size([1, 5, 4, 4])
    tensor([[[[ 0.0872, -0.3896,  0.4076, -1.0552],
              [-1.1601, -0.4499, -1.5056,  0.3214],
              [-1.0277,  0.5174, -0.6346,  0.4430],
              [ 0.6110,  0.5376,  0.5315, -0.7882]],
    
             [[ 0.1404, -0.5614, -0.3267, -0.3816],
              [-0.6313,  0.8031, -0.4586, -0.9638],
              [ 1.2084,  0.0960,  0.1610,  0.3354],
              [ 0.3230,  0.7107, -0.7900,  0.1999]],
    
             [[-1.1036, -0.1059, -0.7694, -0.0673],
              [-0.8744, -0.2171,  0.0889, -0.2910],
              [-0.1638,  0.3603, -0.2767,  0.9214],
              [ 0.3695,  0.0415, -0.0934,  0.7285]],
    
             [[ 2.5634,  2.7304,  0.7292,  0.9382],
              [ 0.6571, -1.2862,  0.0888,  0.0846],
              [-1.2098,  1.0923, -0.7803, -0.3302],
              [-2.0784,  0.1947,  0.7307, -1.6973]],
    
             [[-0.9126,  3.6322,  0.7197,  0.9455],
              [ 1.4373, -0.1420,  0.0879,  0.0854],
              [-1.5728, -0.4278, -0.7770, -0.3380],
              [-1.2868, -1.6438,  0.7477, -1.6899]]]], grad_fn=<ViewBackward0>)
    
    xk: torch.Size([1, 5, 2, 4])
    tensor([[[[-0.2452,  0.3009,  1.1554, -0.3450],
              [-1.4381, -0.4582,  0.9283,  0.1765]],
    
             [[ 0.6687, -0.0734, -0.4939,  0.0468],
              [ 1.0499,  1.5378, -1.0762, -0.2688]],
    
             [[-0.2993, -0.0572, -0.7681,  0.1331],
              [-0.6628,  0.7392, -0.3866, -0.6038]],
    
             [[ 1.7872,  2.2082, -0.0089,  1.3218],
              [-1.2170,  1.0736, -1.1438,  1.6824]],
    
             [[-0.8925,  2.6969, -0.0221,  1.3216],
              [-1.5610, -0.4440, -1.1606,  1.6709]]]], grad_fn=<ViewBackward0>)
    

## Self Attention Calculation


```python
# If the number of K & V heads is different from the number of query heads, adjusts keys and values to match the query heads count.
if num_kv_heads != num_heads:
  num_queries_per_kv = num_heads // num_kv_heads
  xk = torch.repeat_interleave(xk, num_queries_per_kv, dim=2)
  xv = torch.repeat_interleave(xv, num_queries_per_kv, dim=2)

xq.shape, xk.shape, xv.shape
```




    (torch.Size([1, 5, 4, 4]), torch.Size([1, 5, 4, 4]), torch.Size([1, 5, 4, 4]))




```python
# Transposes Q, K, and V tensors to align them for the batch matrix multiplication in attention calculation.
xq = xq.transpose(1, 2)
xk = xk.transpose(1, 2)
xv = xv.transpose(1, 2)

xq.shape, xk.shape, xv.shape
```




    (torch.Size([1, 4, 5, 4]), torch.Size([1, 4, 5, 4]), torch.Size([1, 4, 5, 4]))




```python
# Calculates attention logits by performing a batch matrix multiplication between queries and keys
scores = torch.matmul(xq, xk.transpose(2, 3))

# then we scale the logits by the reciprocal of the square root of the head dimension
scores = scores / math.sqrt(head_dim)

scores.shape, scores
```




    (torch.Size([1, 4, 5, 5]),
     tensor([[[[ 0.3482, -0.0819, -0.2287, -1.0514, -1.2660],
               [-0.2246,  0.1393,  0.0951, -0.7451, -1.0683],
               [-0.3135, -0.1767,  0.4592, -1.1440,  0.3138],
               [ 0.3560,  0.5989, -0.6792,  5.9220,  3.1499],
               [ 0.9111, -0.5939, -0.1808,  3.8165,  5.9220]],
     
              [[-0.8507,  0.0079,  0.7861, -1.3142,  0.1401],
               [ 0.0996, -0.1499,  0.1834, -0.3124,  0.7328],
               [ 0.1761, -0.3132,  0.0835, -1.2137, -0.0958],
               [-0.2374,  0.2469, -0.0900, -0.7774, -1.9726],
               [-0.1615,  0.4661, -0.2391,  1.1836, -0.7774]],
     
              [[ 0.3650,  0.1403,  0.5207,  1.6388,  1.4257],
               [-0.7866,  0.5765, -0.4974, -0.4936, -0.7776],
               [-0.0119,  0.2161, -0.0373,  1.2264,  0.9782],
               [ 0.2283,  0.6690,  1.0551,  1.4910,  0.8788],
               [ 0.8385, -0.6911,  0.6153,  0.8874,  1.4910]],
     
              [[-0.3854,  0.5540,  0.1314, -1.0502, -1.5631],
               [-0.7441,  1.1143,  0.2480,  0.8050,  0.2156],
               [-0.2543,  0.1782, -0.3090,  0.4637,  0.3652],
               [ 1.6393, -1.1065,  1.1319, -0.4765, -0.2630],
               [ 1.4997, -2.1146,  0.1846, -1.9486, -0.4765]]]],
            grad_fn=<DivBackward0>))




```python
# use the mask that we precomputed earlier
scores = scores + mask

scores.shape, scores
```




    (torch.Size([1, 4, 5, 5]),
     tensor([[[[ 0.3482,    -inf,    -inf,    -inf,    -inf],
               [-0.2246,  0.1393,    -inf,    -inf,    -inf],
               [-0.3135, -0.1767,  0.4592,    -inf,    -inf],
               [ 0.3560,  0.5989, -0.6792,  5.9220,    -inf],
               [ 0.9111, -0.5939, -0.1808,  3.8165,  5.9220]],
     
              [[-0.8507,    -inf,    -inf,    -inf,    -inf],
               [ 0.0996, -0.1499,    -inf,    -inf,    -inf],
               [ 0.1761, -0.3132,  0.0835,    -inf,    -inf],
               [-0.2374,  0.2469, -0.0900, -0.7774,    -inf],
               [-0.1615,  0.4661, -0.2391,  1.1836, -0.7774]],
     
              [[ 0.3650,    -inf,    -inf,    -inf,    -inf],
               [-0.7866,  0.5765,    -inf,    -inf,    -inf],
               [-0.0119,  0.2161, -0.0373,    -inf,    -inf],
               [ 0.2283,  0.6690,  1.0551,  1.4910,    -inf],
               [ 0.8385, -0.6911,  0.6153,  0.8874,  1.4910]],
     
              [[-0.3854,    -inf,    -inf,    -inf,    -inf],
               [-0.7441,  1.1143,    -inf,    -inf,    -inf],
               [-0.2543,  0.1782, -0.3090,    -inf,    -inf],
               [ 1.6393, -1.1065,  1.1319, -0.4765,    -inf],
               [ 1.4997, -2.1146,  0.1846, -1.9486, -0.4765]]]],
            grad_fn=<AddBackward0>))




```python
# now we perform the softmax operation to get our actual probabilities
scores = F.softmax(scores.float(), dim=-1).type_as(xq)
scores
# notice that thanks to the causal mask, 0 probability is placed on future tokens
```




    tensor([[[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
              [0.4100, 0.5900, 0.0000, 0.0000, 0.0000],
              [0.2319, 0.2659, 0.5022, 0.0000, 0.0000],
              [0.0038, 0.0048, 0.0013, 0.9900, 0.0000],
              [0.0059, 0.0013, 0.0020, 0.1076, 0.8833]],
    
             [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
              [0.5620, 0.4380, 0.0000, 0.0000, 0.0000],
              [0.3961, 0.2428, 0.3611, 0.0000, 0.0000],
              [0.2291, 0.3719, 0.2655, 0.1335, 0.0000],
              [0.1223, 0.2291, 0.1132, 0.4694, 0.0661]],
    
             [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
              [0.2038, 0.7962, 0.0000, 0.0000, 0.0000],
              [0.3095, 0.3887, 0.3017, 0.0000, 0.0000],
              [0.1194, 0.1855, 0.2730, 0.4221, 0.0000],
              [0.2005, 0.0434, 0.1604, 0.2106, 0.3851]],
    
             [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
              [0.1349, 0.8651, 0.0000, 0.0000, 0.0000],
              [0.2867, 0.4418, 0.2714, 0.0000, 0.0000],
              [0.5597, 0.0359, 0.3369, 0.0675, 0.0000],
              [0.6822, 0.0184, 0.1831, 0.0217, 0.0946]]]],
           grad_fn=<SoftmaxBackward0>)




```python
# then matmul by our values projection
output = torch.matmul(scores, xv)
output.shape, output
```




    (torch.Size([1, 4, 5, 4]),
     tensor([[[[ 5.6657e-01,  1.1281e-04, -1.2348e+00,  1.6354e-01],
               [-6.5530e-02, -6.4437e-02, -1.0858e-01,  8.1257e-01],
               [-4.1794e-01, -1.8548e-01,  3.6338e-01,  1.3936e+00],
               [-8.4237e-01,  1.3431e-01, -3.5508e-01,  4.6972e-02],
               [-8.4060e-01,  1.3460e-01, -3.5974e-01,  4.4174e-02]],
     
              [[ 5.6657e-01,  1.1281e-04, -1.2348e+00,  1.6354e-01],
               [ 9.7348e-02, -4.7804e-02, -3.9877e-01,  6.4533e-01],
               [-1.9662e-01, -1.3898e-01,  1.2876e-02,  1.1047e+00],
               [-3.9078e-01, -1.0508e-01,  1.6863e-01,  1.0515e+00],
               [-5.9474e-01,  1.2880e-02, -8.2546e-02,  5.5951e-01]],
     
              [[ 2.8111e-01, -1.0612e+00, -1.3763e+00,  1.1994e+00],
               [ 5.9957e-01,  1.9605e-01, -6.8872e-01, -4.5739e-01],
               [ 2.8775e-01,  3.3477e-01, -4.0325e-01,  3.8707e-02],
               [-6.0493e-01,  5.4292e-01, -3.9868e-01,  2.9173e-01],
               [-9.4570e-01,  2.7500e-01, -6.6008e-01,  6.3501e-01]],
     
              [[ 2.8111e-01, -1.0612e+00, -1.3763e+00,  1.1994e+00],
               [ 6.2711e-01,  3.0477e-01, -6.2926e-01, -6.0067e-01],
               [ 3.2394e-01,  3.4005e-01, -4.2142e-01, -3.6441e-02],
               [-2.6759e-03, -3.4595e-02, -5.9505e-01,  6.9929e-01],
               [-2.9252e-02, -3.9124e-01, -9.0726e-01,  8.9164e-01]]]],
            grad_fn=<UnsafeViewBackward0>))




```python
# and reshape to put the sequence length back into place and the outputs of our heads lined up
output = output.transpose(1, 2).contiguous().view(b, seq_len, -1)
output.shape, output
```




    (torch.Size([1, 5, 16]),
     tensor([[[ 5.6657e-01,  1.1281e-04, -1.2348e+00,  1.6354e-01,  5.6657e-01,
                1.1281e-04, -1.2348e+00,  1.6354e-01,  2.8111e-01, -1.0612e+00,
               -1.3763e+00,  1.1994e+00,  2.8111e-01, -1.0612e+00, -1.3763e+00,
                1.1994e+00],
              [-6.5530e-02, -6.4437e-02, -1.0858e-01,  8.1257e-01,  9.7348e-02,
               -4.7804e-02, -3.9877e-01,  6.4533e-01,  5.9957e-01,  1.9605e-01,
               -6.8872e-01, -4.5739e-01,  6.2711e-01,  3.0477e-01, -6.2926e-01,
               -6.0067e-01],
              [-4.1794e-01, -1.8548e-01,  3.6338e-01,  1.3936e+00, -1.9662e-01,
               -1.3898e-01,  1.2876e-02,  1.1047e+00,  2.8775e-01,  3.3477e-01,
               -4.0325e-01,  3.8707e-02,  3.2394e-01,  3.4005e-01, -4.2142e-01,
               -3.6441e-02],
              [-8.4237e-01,  1.3431e-01, -3.5508e-01,  4.6972e-02, -3.9078e-01,
               -1.0508e-01,  1.6863e-01,  1.0515e+00, -6.0493e-01,  5.4292e-01,
               -3.9868e-01,  2.9173e-01, -2.6759e-03, -3.4595e-02, -5.9505e-01,
                6.9929e-01],
              [-8.4060e-01,  1.3460e-01, -3.5974e-01,  4.4174e-02, -5.9474e-01,
                1.2880e-02, -8.2546e-02,  5.5951e-01, -9.4570e-01,  2.7500e-01,
               -6.6008e-01,  6.3501e-01, -2.9252e-02, -3.9124e-01, -9.0726e-01,
                8.9164e-01]]], grad_fn=<ViewBackward0>))




```python
# finally initializing and apply output projection that mixes the information from the heads together
wo = nn.Linear(num_heads * head_dim, d, bias=False)
Xout = wo(output)
Xout.shape, Xout
```




    (torch.Size([1, 5, 16]),
     tensor([[[ 0.2504, -0.8244, -0.4644, -0.0154, -0.6234, -0.2495,  0.0120,
                1.4353, -0.5141, -0.0405,  0.2646, -0.0283, -0.7532,  0.2364,
               -0.6981,  0.1465],
              [-0.1278, -0.0026,  0.0980, -0.2692,  0.1106, -0.4054,  0.3527,
                0.1252,  0.4328,  0.1712,  0.1188, -0.0338, -0.3422,  0.3819,
                0.1002, -0.0415],
              [-0.2136, -0.1844,  0.2839, -0.8724, -0.1721, -0.0745,  0.5935,
               -0.1583,  0.2587,  0.3587, -0.0645,  0.0536, -0.4971,  0.4845,
               -0.1223, -0.6558],
              [ 0.0211, -0.1059,  0.3098, -0.7751, -0.0276, -0.2247,  0.6090,
               -0.0373,  0.1223, -0.1262, -0.2237, -0.4181, -0.3507,  0.2308,
                0.0079, -0.5153],
              [ 0.2461, -0.3450,  0.1507, -0.6386,  0.0501, -0.2892,  0.5576,
                0.1634, -0.0421, -0.0561, -0.2076, -0.4035, -0.3639,  0.0637,
               -0.2539, -0.5370]]], grad_fn=<UnsafeViewBackward0>))



## Our First Residual Conenction


```python
h += Xout
h.shape, h
```




    (torch.Size([1, 5, 16]),
     tensor([[[-1.7019,  0.3560, -0.5642,  0.2740,  0.0083,  0.6568,  0.7777,
                2.1809,  1.6528,  0.8456,  1.2715,  1.8378,  0.6138, -0.2468,
                0.2666,  0.0779],
              [ 0.4973, -0.1171,  0.4110,  0.9762,  0.0762, -2.4846, -2.4233,
               -1.0243,  0.0409, -1.1761,  0.3659, -0.1364, -0.6327, -0.6155,
                0.1242, -0.1655],
              [-0.5268, -1.9671, -0.1425, -1.2500,  0.5084, -1.7098, -2.2414,
               -0.9651, -0.4048,  0.4762, -1.0224, -0.7989, -1.5809,  2.1959,
               -1.2199, -0.4792],
              [ 1.2842, -0.9693, -1.2787, -1.6626, -3.3137, -1.8999,  1.3247,
               -1.3593,  2.0704, -1.4620,  0.7415, -0.6197,  1.0028,  0.1470,
               -0.8083, -2.3475],
              [ 1.5092, -1.2084, -1.4378, -1.5261, -3.2360, -1.9644,  1.2733,
               -1.1586,  1.9060, -1.3919,  0.7576, -0.6051,  0.9896, -0.0201,
               -1.0701, -2.3691]]], grad_fn=<AddBackward0>))




```python
# normalize the current state of our residual for use in our MoE later
pre_ffwd_norm = RMSNorm(d)
h_normed = pre_ffwd_norm(h)
# so now we're working with x, which we'll use later for our next residual conenction, and x_normed which is used by our MoE MLP
```

### The SwiGLU Feedforward Network
<a id='j'></a>

Llama 3 models have surprisingly not opted for a mixture of experts strategy which i was assuming they'd go for by now. Their feedforward networks use the SwiGLU activation which basically uses the activation function as a gate that dynamically determines what information gets through


```python
# first we need to define our actual hidden dimension, which Llama's code does in an unnecessarily complicated manner
hidden_dim = 4 * d # usually i would designate a hyperparameter for this 4, but in llama's code it was just there
print(hidden_dim)
hidden_dim = int(2 * hidden_dim / 3)
print(hidden_dim)
multiple_of = 256 # their description of this was "make SwiGLU hidden layer size multiple of large power of 2"
hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
print(hidden_dim)
# so basically this overly convoluted setup is designed to ensure that hidden_dim is a multiple of 256, likely for hardware efficiency reasons
```

    64
    42
    256
    


```python
up = nn.Linear(d, hidden_dim, bias=False)
gate = nn.Linear(d, hidden_dim, bias=False)
down = nn.Linear(hidden_dim, d, bias=False)
```


```python
up_proj = up(h_normed)
print(up_proj.shape, up_proj)
```

    torch.Size([1, 5, 256]) tensor([[[-1.3826,  0.2002, -0.2022,  ...,  0.4446,  0.6443,  1.3379],
             [ 1.4994, -1.1969,  0.2194,  ..., -0.2797,  0.2374, -1.1965],
             [ 0.5883, -0.4688,  1.2317,  ...,  0.4954, -0.0362, -0.9739],
             [-0.0951, -1.0717, -0.9506,  ..., -0.6338,  0.4064, -1.0823],
             [-0.0810, -1.1284, -0.8850,  ..., -0.6748,  0.4355, -1.1314]]],
           grad_fn=<UnsafeViewBackward0>)
    


```python
gate_proj = F.silu(gate(h_normed))
print(gate_proj.shape, gate_proj)
```

    torch.Size([1, 5, 256]) tensor([[[-0.1556,  0.0905, -0.0172,  ..., -0.2638, -0.1143, -0.1516],
             [-0.2375, -0.2485, -0.1762,  ...,  0.3805, -0.0936,  0.1342],
             [-0.1863, -0.1474, -0.2290,  ...,  0.1437,  1.1082,  0.3493],
             [ 0.1088, -0.0575,  0.2963,  ...,  0.6216, -0.2287, -0.2784],
             [ 0.0857, -0.0883,  0.2486,  ...,  0.6104, -0.2353, -0.2784]]],
           grad_fn=<SiluBackward0>)
    


```python
ffwd_output = down(up_proj * gate_proj)
print(ffwd_output.shape, ffwd_output)
```

    torch.Size([1, 5, 16]) tensor([[[-0.0398, -0.0875,  0.1123, -0.0583,  0.0320, -0.0144, -0.0228,
               0.0814,  0.0616, -0.0204, -0.1008,  0.0124,  0.0400, -0.0506,
               0.1751, -0.0382],
             [-0.0649, -0.0929,  0.0057, -0.0543, -0.1376,  0.0208, -0.0446,
               0.0018, -0.0806,  0.0368, -0.0730,  0.1667, -0.1034, -0.1088,
               0.0230,  0.1700],
             [ 0.0900,  0.0895,  0.0714, -0.1082, -0.0513, -0.0009, -0.2846,
              -0.0148,  0.0209, -0.0102,  0.1059, -0.1719,  0.2289,  0.0312,
              -0.1348, -0.2782],
             [-0.0907, -0.0177, -0.0762,  0.0318,  0.0374,  0.0035,  0.0866,
              -0.0767, -0.1355,  0.1861, -0.0705,  0.0818, -0.0924,  0.0693,
              -0.0582,  0.1364],
             [-0.1031, -0.0116, -0.0939,  0.0256,  0.0344,  0.0006,  0.0807,
              -0.0549, -0.1132,  0.1857, -0.0889,  0.0883, -0.0764,  0.0694,
              -0.0590,  0.1202]]], grad_fn=<UnsafeViewBackward0>)
    


```python
# and then do our final residual connection of this layer
out = h + ffwd_output
print(out.shape, out)
```

    torch.Size([1, 5, 16]) tensor([[[-1.7417,  0.2685, -0.4519,  0.2157,  0.0403,  0.6425,  0.7549,
               2.2623,  1.7144,  0.8252,  1.1707,  1.8502,  0.6538, -0.2975,
               0.4417,  0.0398],
             [ 0.4324, -0.2100,  0.4167,  0.9219, -0.0614, -2.4638, -2.4678,
              -1.0225, -0.0397, -1.1393,  0.2929,  0.0304, -0.7361, -0.7243,
               0.1472,  0.0045],
             [-0.4368, -1.8777, -0.0711, -1.3582,  0.4571, -1.7108, -2.5261,
              -0.9799, -0.3839,  0.4660, -0.9164, -0.9708, -1.3519,  2.2271,
              -1.3547, -0.7574],
             [ 1.1935, -0.9870, -1.3549, -1.6308, -3.2763, -1.8964,  1.4113,
              -1.4360,  1.9349, -1.2759,  0.6710, -0.5379,  0.9104,  0.2164,
              -0.8665, -2.2111],
             [ 1.4062, -1.2200, -1.5317, -1.5005, -3.2016, -1.9637,  1.3540,
              -1.2135,  1.7928, -1.2062,  0.6688, -0.5168,  0.9132,  0.0494,
              -1.1291, -2.2489]]], grad_fn=<AddBackward0>)
    

### Output
<a id='k'></a>
So usually we'd run it back on steps 1e through 1j for however many layers our model has (Llama 3 8b uses 32) using different weight matrices but you get the point. Since our current `out` is of the same shape that it would be if we were to do more layers, let's go ahead and just see what Llama's output mechanism looks like. It's nothing interesting though, just a linear layer. Notably they chose to use a separate linear layer rather than re-using the embedding layer as is relatively common


```python
# first we norm the residual state
final_norm = RMSNorm(d)
out_normed = final_norm(out)
```


```python
# then multiply by the linear layer to get our final output logits
final_output = nn.Linear(d, v, bias=False)
logits = final_output(out_normed).float()
logits.shape, logits
```




    (torch.Size([1, 5, 10]),
     tensor([[[ 0.1116, -0.3063,  0.7228, -0.6840, -0.1686, -0.1784,  0.0416,
               -0.0821,  0.0095, -0.3498],
              [-1.1384, -0.0319, -0.1838, -0.2996,  0.6792,  0.1604,  0.1896,
                0.4529,  0.8159, -0.4756],
              [-0.5640,  0.2507, -0.5232, -0.0683,  0.2641, -0.5735,  0.5249,
                0.6260,  0.6022,  0.0981],
              [-0.2287,  0.6033,  0.1108,  0.6606, -0.4193,  0.0462,  0.6884,
               -0.0721, -0.2592, -0.0524],
              [-0.2553,  0.6691,  0.1058,  0.6548, -0.4101,  0.0317,  0.7300,
               -0.1380, -0.2526, -0.0772]]], grad_fn=<UnsafeViewBackward0>))




```python
# softmax the logits to get the probability for each token's prediction across every token in the sequence
probs = F.softmax(logits, dim=-1)
probs
```




    tensor([[[0.1143, 0.0753, 0.2107, 0.0516, 0.0864, 0.0856, 0.1066, 0.0942,
              0.1032, 0.0721],
             [0.0274, 0.0830, 0.0713, 0.0635, 0.1690, 0.1006, 0.1036, 0.1347,
              0.1937, 0.0532],
             [0.0485, 0.1095, 0.0505, 0.0796, 0.1110, 0.0480, 0.1440, 0.1594,
              0.1556, 0.0940],
             [0.0662, 0.1520, 0.0929, 0.1610, 0.0547, 0.0871, 0.1656, 0.0774,
              0.0642, 0.0789],
             [0.0640, 0.1613, 0.0918, 0.1590, 0.0548, 0.0852, 0.1714, 0.0719,
              0.0642, 0.0765]]], grad_fn=<SoftmaxBackward0>)




```python
# Greedily decode the probabilities to get our final predicted indices
greedy_indices = torch.argmax(probs, dim=-1)
greedy_indices
# if we were performing inference rather than training, that final token in the list would be the one to show the user
```




    tensor([[2, 8, 7, 6, 6]])



### The loss functions
<a id='l'></a>

Of course we use [cross-entropy loss](https://machinelearningmastery.com/cross-entropy-for-machine-learning/) which should need no introduction if this isn't your first machine-learning rodeo, so we'll be skimming past it. Basically the idea is that the single correct value is rewarded and all other values are suppressed


```python
# create some random fake target indices to train on
target_token_indices = torch.randint(0, v, greedy_indices.shape)
print(target_token_indices)

# initialize the loss function
loss_fn = nn.CrossEntropyLoss()

# reshape logits to be compatible and calculate loss
loss = loss_fn(logits.view(1,v,seq_len), target_token_indices)
print(loss)
```

    tensor([[1, 9, 3, 9, 1]])
    tensor(2.3382, grad_fn=<NllLoss2DBackward0>)
    

## Now let's code everything up the correct way into classes so that we can actually build a functioning model


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dataclasses import dataclass
from typing import Optional, Tuple

import time

import os
import json
```

we'll be using a crazy small & simple tokenizer based on the TinyShakespeare dataset
Llama 3 8b's vocabulary size is 128256 including special tokens like <|endoftext|>


```python
import pickle
import os

class SimpleTokenizer:
    def __init__(self, stoi, merges):
        self.stoi = stoi
        self.merges = merges
        self.itos = {i: s for s, i in stoi.items()}  # Inverse mapping for decoding

        self.vocab_len = len(stoi) + len(merges)

    def encode(self, text):
        # Convert the text to a list of token IDs, using space for unknown characters
        tokens = [self.stoi.get(c, self.stoi[' ']) for c in text]

        # Perform merging with the possibility of nested merges
        i = 0
        while i < len(tokens) - 1:
            pair = (tokens[i], tokens[i + 1])
            if pair in self.merges:
                # Replace the current pair with its merged token
                merged_token = self.merges[pair]
                tokens[i] = merged_token
                del tokens[i + 1]

                # Move back to handle possible nested merges
                if i > 0:
                    i -= 1
            else:
                i += 1

        return tokens

    def decode(self, tokens):
        def expand_token(token):
            # Base case: if the token is a direct mapping, return its character
            if token in self.itos:
                return self.itos[token]
            # Recursive case: if the token is a merged token, expand its constituents
            elif token in self.merges.values():
                pair = next(key for key, value in self.merges.items() if value == token)
                return ''.join(expand_token(t) for t in pair)
            # Fallback for unknown tokens
            else:
                return ''

        # Decode each token in the list, handling nested merges recursively
        return ''.join(expand_token(token) for token in tokens)

def load_tokenizer_data(size: int):
    file_name = f'./tokenizers/tiny_shakespeare_tokenizer_{size}.model'
    with open(file_name, 'rb') as f:
        tokenizer_data = pickle.load(f)
    return tokenizer_data

def get_tokenizer(size: int):
    tokenizer_data = load_tokenizer_data(size)
    loaded_stoi = tokenizer_data['stoi']
    loaded_merges = tokenizer_data['merges']
    return SimpleTokenizer(loaded_stoi, loaded_merges)
```


```python
tokenizer = get_tokenizer(size = 512)
```
## Acknowledgments 💖

This project has been inspired and informed by various resources and individuals in the AI and machine learning
community. I'd like to extend my gratitude to the following:

- [Andrej Karpathy for his tutorial on training a GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1058s&ab_channel=AndrejKarpathy).
  His insights into neural network architectures and training methodologies have been invaluable.
- [Umar Jamil's guide on Training LLama2 from scratch](https://www.youtube.com/watch?v=oM4VmoabDAI&ab_channel=UmarJamil).
  This resource provided practical insights and a foundational understanding necessary for this implementation.
- The [Meta LLaMA GitHub repository](https://github.com/meta-llama/llama) has been an essential resource for
  understanding the intricacies of the LLaMA 2 model and its implementation.

I am grateful for the knowledge shared by these individuals and communities, which has significantly contributed to the
development of this project.


