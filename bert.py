import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base_bert import BertPreTrainedModel
from utils import *

class BertSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key   = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)

    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x)
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)

    # Produce multiple heads for the proj
    # Done by splitting the hidden state to self.num_attention_heads, each of size self.attention_head_size
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)

    # By transposing we have proj of size [bs, num_attention_heads, seq_len, attention_head_size]
    proj = proj.transpose(1, 2)

    return proj

  def attention(self, key, query, value, attention_mask):
    # Each attention is calculated following eq. (1) of https://arxiv.org/pdf/1706.03762.pdf.
    # Attention scores are calculated by multiplying the key and query to obtain
    # a score matrix S of size [bs, num_attention_heads, seq_len, seq_len].
    # S[*, i, j, k] represents the (unnormalized) attention score between the j-th and k-th
    # token, given by i-th attention head.
    # Before normalizing the scores, use the attention mask to mask out the padding token scores.
    # Note that the attention mask distinguishes between non-padding tokens (with a value of 0)
    # and padding tokens (with a value of a large negative number).

    # Make sure to:
    # - Normalize the scores with softmax.
    # - Multiply the attention scores with the value to get back weighted values.
    # - Before returning, concatenate multi-heads to recover the original shape:
    #   [bs, seq_len, num_attention_heads * attention_head_size = hidden_size].

    # Calculate attention scores S = (Q * K^T) / sqrt(d_k)
    # then apply the attention mask
    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    attention_scores = attention_scores / np.sqrt(self.attention_head_size)

    # Apply the attention mask (additive to become effectively zero after softmax)
    attention_scores = attention_scores + attention_mask

    # Normalize the scores to probabilities (0-1) and apply dropout
    attention_probs = nn.Softmax(dim=-1)(attention_scores)
    attention_probs = self.dropout(attention_probs)

    # Get the context by multiplying the probabilities by the value V
    context_layer = torch.matmul(attention_probs, value)

    # Concatenate heads and put in contiguous memory layout
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    return context_layer


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size]
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)

    # Calculate the multi-head attention
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value


class BertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention
    self.self_attention = BertSelfAttention(config)

    # Add-norm for multi-head attention
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)

    # Feed forward
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu

    # Add-norm for feed forward
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    """
    This function is applied after the multi-head attention layer or the feed forward layer.
    input: the input of the previous layer
    output: the output of the previous layer
    dense_layer: used to transform the output
    dropout: the dropout to be applied 
    ln_layer: the layer norm to be applied
    """
    # Hint: Remember that BERT applies dropout to the transformed output of each sub-layer,
    # before it is added to the sub-layer input and normalized with a layer norm.

    # Apply dense layer to output tensor
    dense_output = dense_layer(output)

    # Apply dropout on dense output
    dropout_output = dropout(dense_output)

    # Add the dropout output to the input tensor to perform residual connection
    added_output = input + dropout_output

    # Apply layer normalization to the added output
    normed_output = ln_layer(added_output)

    return normed_output


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: either from the embedding layer (first BERT layer) or from the previous BERT layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf.
    Each block consists of:
    1. A multi-head attention layer (BertSelfAttention).
    2. An add-norm operation that takes the input and output of the multi-head attention layer.
    3. A feed forward layer.
    4. An add-norm operation that takes the input and output of the feed forward layer.
    """
    # Apply multi-head self-attention
    attention_output = self.self_attention(hidden_states, attention_mask)

    # First add-norm layer
    attention_output = self.add_norm(hidden_states, attention_output, 
                                     self.attention_dense, self.attention_dropout, 
                                     self.attention_layer_norm)

    # Apply feed forward network
    intermediate_output = self.interm_dense(attention_output)
    intermediate_output = self.interm_af(intermediate_output)

    # Second add-norm layer
    layer_output = self.add_norm(attention_output, intermediate_output,
                                 self.out_dense, self.out_dropout,
                                 self.out_layer_norm)
    
    return layer_output



class BertModel(BertPreTrainedModel):
  """
  The BERT model returns the final embeddings for each token in a sentence.
  
  The model consists of:
  1. Embedding layers (used in self.embed).
  2. A stack of n BERT layers (used in self.encode).
  3. A linear transformation layer for the [CLS] token (used in self.forward, as given).
  """
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # Embedding layers
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

    # Register position_ids (1, len position emb) to buffer because it is a constant
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # BERT encoder
    self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    # [CLS] token transformations
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    self.init_weights()

  def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]

    # Obtain the input token embeddings
    input_embeds = self.word_embedding(input_ids)

    # Generate position indices and obtain the positional embeddings
    pos_ids = self.position_ids[:, :seq_length]
    pos_embeds = self.pos_embedding(pos_ids)

    # Token type embeddings are not considered in this scenario, thus a tensor of zeros
    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)

    # Sum the embeddings for input tokens, positions, and token types
    embeddings = input_embeds + pos_embeds + tk_type_embeds

    # Apply layer normalization and dropout to the combined embeddings
    embeddings = self.embed_layer_norm(embeddings)
    embeddings = self.embed_dropout(embeddings)

    return embeddings


  def encode(self, hidden_states, attention_mask):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    """
    Get the extended attention mask for self-attention
    and returns an attention mask of size [batch_size, 1, 1, seq_len]
    Distinguishes between non-padding tokens (with a value of 0) and padding tokens
    (with a value of a large negative number)
    """
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # Pass hidden states through encoder layers
    for i, layer_module in enumerate(self.bert_layers):
      # Feed encoding from last bert_layer to the next
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    # Get the embedding for each input token
    embedding_output = self.embed(input_ids=input_ids)

    # Feed to a transformer (a stack of BertLayers)
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

    # Get cls token hidden state
    first_tk = sequence_output[:, 0]
    first_tk = self.pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)

    return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}
