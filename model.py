import torch 
import torch.nn as nn
import math

"""
+---------------------- TRANSFORMER ARCHITECTURE ----------------------+
|                                                                      |
|  +----------------+                        +----------------+        |
|  |     INPUT      |                        | SHIFTED OUTPUT |        |
|  +----------------+                        +----------------+        |
|          |                                          |                |
|          v                                          v                |
|  +----------------+                        +----------------+        |
|  | Input Embedding|                        |Output Embedding|        |
|  +----------------+                        +----------------+        |
|          |                                          |                |
|          v                                          v                |
|  +----------------+                        +----------------+        |
|  |   Positional   |                        |   Positional   |        |
|  |    Encoding    |                        |    Encoding    |        |
|  +----------------+                        +----------------+        |
|          |                                          |                |
|          v                                          v                |
|  +----------------+                        +----------------+        |
|  |                |                        |                |        |
|  |    ENCODER     |                        |    DECODER     |        |
|  |                |                        |                |        |
|  | +------------+ |                        | +------------+ |        |
|  | | Multi-Head  | |                       | |  Masked    | |        |
|  | | Attention   | |                       | | Multi-Head | |        |
|  | +------------+ |                        | | Attention  | |        |
|  |       |        |                        | +------------+ |        |
|  |       v        |                        |       |        |        |
|  | +------------+ |                        |       v        |        |
|  | | Add & Norm | |                        | +------------+ |        |
|  | +------------+ |                        | | Add & Norm | |        |
|  |       |        |                        | +------------+ |        |
|  |       v        |                        |       |        |        |
|  | +------------+ |                        |       v        |        |
|  | |    Feed    | |  --------------------> | +------------+ |        |
|  | |  Forward   | |      Encoder Output    | | Multi-Head  | |        |
|  | +------------+ |                        | | Attention   | |        |
|  |       |        |                        | +------------+ |        |
|  |       v        |                        |       |        |        |
|  | +------------+ |                        |       v        |        |
|  | | Add & Norm | |                        | +------------+ |        |
|  | +------------+ |                        | | Add & Norm | |        |
|  |       |        |                        | +------------+ |        |
|  +-------|--------+                        |       |        |        |
|          |                                 |       v        |        |
|          |                                 | +------------+ |        |
|          |                                 | |    Feed    | |        |
|          |                                 | |  Forward   | |        |
|          |                                 | +------------+ |        |
|          |                                 |       |        |        |
|          |                                 |       v        |        |
|          |                                 | +------------+ |        |
|          |                                 | | Add & Norm | |        |
|          |                                 | +------------+ |        |
|          |                                 |       |        |        |
|          |                                 +-------|--------+        |
|          |                                         |                 |
|          |                                         v                 |
|          |                                 +----------------+        |
|          |                                 |  Linear Layer  |        |
|          |                                 +----------------+        |
|          |                                         |                 |
|          |                                         v                 |
|          |                                 +----------------+        |
|          |                                 |     Softmax    |        |
|          |                                 +----------------+        |
|          |                                         |                 |
|          |                                         v                 |
|          |                                 +----------------+        |
|          |                                 |     OUTPUT     |        |
|          |                                 +----------------+        |
+-----------------------------------------------------------------------+
"""



"""
TRANSFORMER ARCHITECTURE (Simplified for Beginners)
=================================================

INPUT --> [Hello, World!] (text, image, or other data)
           |
           v
[1] TOKEN EMBEDDING: Convert input into numbers
           |
           v
[2] POSITIONAL ENCODING: Add position information
           |
           v
           |
    +------+------+ ENCODER SECTION
    |             |
    v             |
[3] SELF-ATTENTION: Look at all input words at once
    | (What parts of input are important?)
    v
[4] ADD & NORMALIZE: Combine attention with original
    |
    v
[5] FEED FORWARD: Process each position separately
    |
    v
[6] ADD & NORMALIZE: Combine with previous step
    |
    |    (Repeat steps 3-6 several times)
    |
    v
    |             |
    |             v
    |      DECODER SECTION
    |             |
    |             v
    |     [7] MASKED SELF-ATTENTION
    |         | (Look at previous output words only)
    |         v
    |     [8] ADD & NORMALIZE
    |         |
    |         v
    +-------> [9] ENCODER-DECODER ATTENTION
              | (Connect encoder & decoder)
              v
        [10] ADD & NORMALIZE
              |
              v
        [11] FEED FORWARD
              |
              v
        [12] ADD & NORMALIZE
              |
              |  (Repeat steps 7-12 several times)
              |
              v
        [13] LINEAR LAYER
              |
              v
        [14] SOFTMAX: Convert to probabilities
              |
              v
OUTPUT --> [Generated text or classification]

"""
class InputEmbedding(nn.Module):

  def __init__(self, d_model:int, vocab_size: int):

    super().__init__()
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.emmbedding = nn.Embedding(vocab_size, d_model)

  def forward(self, x):
    # (batch, seq_len) --> (batch, seq_len, d_model)

    return self.emmbedding(x) * math.sqrt(self.d_model)  # to increase the variance we multiply it with sqrt of d model
    # Without scaling, the variance of embeddings would be too small, leading to smaller gradients and slower learning


class PositionalEncoding(nn.Module):

  def __init__(self, d_model: int, seq_len:int, dropout: float):

    super().__init__()
    
    self.d_model = d_model
    self.seq_len = seq_len
    self.dropout = nn.Dropout(dropout)
    
    # matrix of size (seq_len,d_model)
    pe = torch.zeros(seq_len, d_model)

    # vector of shape (seq_len, 1)
    position = torch.arange(0, seq_len, dtype=torch.float)
    div = torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model)

    pe[:,0::2] = torch.sin(position * div) # sin(position * (10000 ** (2i / d_model))
    pe[:,1::2] = torch.cos(position * div)

    pe = pe.unsqueeze(0) 
    #Stores pe in the model as a non-trainable tensor.
    self.register_buffer('pe', pe)

  def forward (self, x):
    # if max seq_len changes ; To avoid shape mismatch
    x = x+ (self.pe[:, :x.shape[1], :]).require_grad(False)
    return self.dropout(x)


class LayerNormalization(nn.Module):

  def __init__(self, eps:float = 10**-6):
    super().__init__()
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(1)) 
    self.beta = nn.Parameter(torch.zeros(1))
    
  def forward(self, x):
    mean = x.mean(dim =-1, keepdim = True) #dim=-1: Computes the mean across the last axis (each feature vector independently).
    std = x.std(dim = -1, keepdim = True) # keepdim=True: Maintains the original shape for proper broadcasting.

    return self.alpha * (x-mean)/(std + self.esp) +self.beta




class feedForwardBlock(nn.Module):
    #FFN(x)=W2(Dropout(ReLU(W1x + b1)))+b 2
    # ​d_model → The input and output size of the block (same as model embedding size).
    # d_ff → The hidden layer size (usually much larger than d_model).
    # dropout → Dropout probability to avoid overfitting.
    # self.linear_1 → Expands the input dimension (d_model → d_ff).
    # self.linear_2 → Shrinks it back (d_ff → d_model).
    # self.dropout → Drops some neurons randomly to improve generalization.

    def __init__(self, d_model:int , d_ff:int, dropout:float):
      super().__init__()
      self.layer1 = nn.Linear(d_model, d_ff)
      self.dropout = nn.Dropout(dropout)
      self.layer2 = nn.Linear(dff, d_model)



    def forward(self, x):
      # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)

      return self.layer2(self.dropout(torch.relu(self.layer1(x))))



import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
   
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h  # Dimension of vector seen by each head
        self.dropout = nn.Dropout(dropout)

        assert d_model % h == 0, "d_model is not exactly div by h"
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo

    def attention(self, query, key, value, mask):
        d_k = query.shape[-1]

        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # very low value (indicating -inf) to the positions where mask == 0
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len) # Apply softmax

        attention_scores = self.dropout(attention_scores)

        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # Multi-head attention processes each attention head independently. that's why we doiin transpose
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = self.attention(query, key, value, mask)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        # transpose() does not physically rearrange the memory; it just changes how we view the tensor.
        # contiguous() ensures that the tensor is stored sequentially in memory before using .view().
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):
  def __init__(self, features, dropout:float):
    #Residual connections (x + f(x)) help Transformers train more efficiently and avoid vanishing gradients.
    
    #Prevents Over-smoothing – In deep architectures, without residuals, layers may transform representations too much, leading to loss of important information.

    super().__init__()

    self.dropout = nn.Dropout(dropout)
    self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):

      return x+ self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttention, feed_forward_block: feedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
    

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):

  def __init__(self, features: int, layer: nn.ModuleList):
    super().__init__()
    self.layers = layers
    self.norm =  LayerNormalization(features)

  def forward(self, x, mask):
    for layer in self.layers:
      x = layer(x, mask)
    return self.norm(x)



class DecoderBlock(nn.Module):

    def __init__(self, feature: int, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: feedForwardBlock, dropout:float):
      super().__init__()
      self.self_attention_block = self_attention_block
      self.cross_attention_block = cross_attention_block
      self.feed_forward_block = feed_forward_block
      self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])


    """Masking
       Mask Type	Where Used?	Purpose
       tgt_mask (Target Mask)	Self-attention (decoder)	Prevents looking at future tokens (causal)
       src_mask (Source Mask)	Cross-attention (decoder)	Prevents attending to <PAD> tokens
    """
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


"""📌 How Data Flows?
1️⃣ The input x goes through each DecoderBlock in self.layers.
2️⃣ Each DecoderBlock applies:

Self-attention (using tgt_mask)
Cross-attention with the encoder (using src_mask)
Feed-forward network
Residual connections
3️⃣ The final output is normalized before being returned."""
class Decoder(nn.Module):

  #features: The number of features in the input embeddings (same as d_model in Transformers).
  #layers: A list of DecoderBlock layers passed as an nn.ModuleList.
  # self.norm: Applies layer normalization at the end of the decoding process.
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)# Pass through each decoder block
        return self.norm(x)




class ProjectionLayer(nn.Module):
    """
🔑 Summary
Step	Description
x input	(batch, seq_len, d_model), hidden states from decoder
Linear (nn.Linear)	Projects d_model → vocab_size (word logits)
Softmax	Converts logits into word probabilities
Prediction	Highest probability word is selected
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)

