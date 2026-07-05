"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # TODO: build a token-to-id dict with specials first
    # then corpus tokens in first-seen order

    dict = {}
    for token in specials:
        if token not in dict:
            dict[token] = len(dict)
    
    for sentence in sentences:
        for token in sentence.split():
            if token not in dict:
                dict[token] = len(dict)
    return dict

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    return {token_id : token for token, token_id, in token_to_id.items()}

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    
    unk_id = token_to_id[unk_token]
    
    return [token_to_id.get(token, unk_token) for token in sentence.split() ]

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    
    return [id_to_token[id] for id in ids]

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):

    result = list(ids[:max_len])
    result += [pad_id] * (max_len - len(result))
    return result

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # TODO: stack padded id sequences into a (B, L) torch.long tensor

    return torch.tensor(padded_sequences, dtype=torch.long)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    return embeddings * math.sqrt(d_model)

# Step 8 - compute_positional_div_term
import torch
import math

def compute_positional_div_term(d_model):
    
    # tensor of all even numbers [0, 2, 4, ..., d_model - 2]
    two_i = torch.arange(0, d_model, 2).float()

    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    return div_term

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # TODO: build a column vector of position indices from 0 to max_len-1
    positions = torch.arange(max_len).float() 
    return positions.unsqueeze(1)

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe: torch.Tensor, position: torch.Tensor, div_term: torch.Tensor):
    """Fill even feature indices of pe with sin(position * div_term)."""
    
    # Calculate the inner arguments for the sine function using broadcasting
    arguments = position * div_term
    
    # Inject the sine of the arguments into all rows (:), and every even column (0::2)
    pe[:, 0::2] = torch.sin(arguments)
    
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    # TODO: fill the odd-indexed columns of pe with cos(position * div_term)
    arguments = position * div_term

    pe[:, 1::2] = torch.cos(arguments)
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # TODO: build the (max_len, d_model) sinusoidal positional encoding matrix

    pe = torch.zeros(max_len, d_model)

    position = torch.arange(0, max_len, dtype=torch.float)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

    arguments = position.unsqueeze(1) * div_term.unsqueeze(0)
    pe[:, 0::2] = torch.sin(arguments)
    pe[:, 1::2] = torch.cos(arguments)
    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # TODO: add the first L rows of positional_encoding to embedded_batch and return the sum.

    batch_size, seq_len, d_model = embedded_batch.shape 
    pe_sliced = positional_encoding[:seq_len, :]

    return embedded_batch + pe_sliced

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # TODO: build a boolean mask marking non-pad positions, shaped for broadcasting against attention scores
    
    mask = (token_ids != pad_id)
    mask = mask.unsqueeze(1).unsqueeze(2)
    return mask

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # TODO: build a lower-triangular boolean causal mask of shape (1, 1, seq_len, seq_len)
    
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    mask = torch.tril(mask)

    mask = mask.unsqueeze(0).unsqueeze(1)
    return mask

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # TODO: combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).
    return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    
    key_T = key.transpose(-2, -1)
    scores = query @ key_T
    return scores

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    return scores / math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    return scores.masked_fill(mask==False, float('-inf'))

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    # TODO: softmax over the last axis, zeroing rows that are entirely -inf
    weights = torch.softmax(masked_scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)

    return weights

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    context = attention_weights @ value
    return context

# Step 22 - scaled_dot_product_attention
import torch
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    # TODO: chain raw scores, scale by sqrt(d_k), optionally mask, softmax, then mix values
    
    d_model = query.size(-1)

    # 1. Q @ K^T
    key_T = key.transpose(-2, -1)
    attention_raw = query @ key_T

    #2. Scale
    attention_raw = attention_raw / math.sqrt(d_model)

    #3. Mask
    if mask is not None:
        attention_raw = attention_raw.masked_fill(mask ==False, -1e9)

    #4. Softmax
    weights = torch.softmax(attention_raw, -1)

    if mask is not None:
        weights = weights.masked_fill(mask == False, 0.0)
    
    weights = torch.nan_to_num(weights, nan=0.0)
    
    scaled_dot_product = weights @ value
    return scaled_dot_product, weights

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    
    batch_size, seq_length, d_model = tensor.size()
    head_dim = d_model // num_heads 

    tensor = tensor.reshape(batch_size, seq_length, num_heads, head_dim)
    return tensor

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    
    split_tensor = split_tensor.transpose(1, 2)
    return split_tensor

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    
    batch, num_heads, seq_len, d_model = multi_head_tensor.size()
    multi_head_tensor = multi_head_tensor.transpose(2,1)
    multi_head_tensor = multi_head_tensor.reshape(batch, seq_len, num_heads * d_model)
    return multi_head_tensor

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    
    transposed_weight = weight.t()

    proj = x @ transposed_weight
    if bias is not None:
        proj = proj + bias

    return proj

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers
    
    w_q = w_q.t()
    w_k = w_k.t()
    w_v = w_v.t()

    proj_q = x @ w_q
    if b_q is not None:
        proj_q += b_q 

    proj_k = x @ w_k
    if b_k is not None:
        proj_k += b_k

    proj_v = x @ w_v
    if b_v is not None:
        proj_v += b_v

    return proj_q, proj_k, proj_v

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    
    b, l, d_model = q.size()

    q = q.reshape(b, l, num_heads, d_model // num_heads).transpose(1,2)
    k = k.reshape(b, l, num_heads, d_model // num_heads).transpose(1,2)
    v = v.reshape(b, l, num_heads, d_model // num_heads).transpose(1,2)
    return q, k, v

# Step 29 - multi_head_scaled_dot_product_attention
import torch
import math

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # TODO: run scaled dot-product attention over per-head Q, K, V and return (context, weights)
    
    d_k = q_h.size(-1)

    # 1. q_h @ k_h ^ t
    k_h = k_h.transpose(-2, -1)
    attention_raw = q_h @ k_h

    # 2. scale 
    attention_raw = attention_raw / math.sqrt(d_k)

    # 3. mask
    if mask is not None:
        attention_raw = attention_raw.masked_fill(mask == False, -float('inf'))
    
    # 4. softmax
    weights = torch.softmax(attention_raw, -1)
    
    scaled_dot_product = weights @ v_h
    return scaled_dot_product, weights

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    
    # Each head has its own slice of context
    merged = merge_heads_back_to_model_dim(context)
    return apply_linear_projection(merged, w_o, b_o)

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # TODO: project Q/K/V, split into heads, 
    # run scaled dot-product attention, merge heads, output projection.

    # 1. INPUT PROJECTIONS 
    query_proj = query @ w_q.t()
    key_proj = key @ w_k.t() 
    value_proj = value @ w_v.t()

    # 2. SPLIT INTO HEADS
    batch_size, seq_len, d_model = query.size()
    head_dim = d_model // num_heads
    
    query_h = query_proj.reshape(batch_size, -1, num_heads, head_dim).transpose(1, 2)
    key_h = key_proj.reshape(batch_size, -1, num_heads, head_dim).transpose(1, 2)
    value_h = value_proj.reshape(batch_size, -1, num_heads, head_dim).transpose(1, 2)

    # 3. SCALED PRODUCT ATTENTION
    raw_attention = query_h @ key_h.transpose(-2, -1)
    raw_attention = raw_attention / math.sqrt(head_dim)
    if mask is not None:
        raw_attention = raw_attention.masked_fill(mask == False, -float('inf'))
    weights = torch.softmax(raw_attention, -1)
    context = weights @ value_h

    # 4. MERGE HEADS
    # Before: (batch, num_heads, seq_len, head_dim)
    # After:  (batch, seq_len, num_heads * head_dim) 
    context = context.transpose(1, 2)
    merged_heads = context.reshape(batch_size, seq_len, d_model)

    # 5. OUTPUT PROJECTION
    final_output = merged_heads @ w_o.t()
    return final_output

# Step 32 - apply_ffn_first_linear_and_relu
import torch.nn.functional as F

def apply_ffn_first_linear_and_relu(x, w1, b1):
    linear_out = x @ w1 + b1
    return F.relu(linear_out)

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # TODO: project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    
    linear_out = hidden @ w2 + b2
    return linear_out

# Step 34 - position_wise_feed_forward_network
import torch.nn.functional as F
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # TODO: compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).
    
    layer_1 = x @ w1 + b1
    relu = F.relu(layer_1)
    layer_2 = relu @ w2 + b2
    return layer_2

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # TODO: return (mean, variance) reduced over the last dim with shape (..., 1)
    
    #keepdim=True allows us to do (batch, seq, 1) instead of (batch, seq)
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False)
    return mean, variance

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # TODO: standardize x along the last axis then apply gamma and beta affine transform
    
    mean, var = compute_layer_norm_mean_and_variance(x)

    standardized_x = (x - mean) / torch.sqrt(var + eps)

    return gamma * standardized_x + beta

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # TODO: combine the residual with the sublayer output and layer-normalize the result.
    
    x = residual_input + sublayer_output 

    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    normalized_x = (x - mean) / torch.sqrt(eps + var)

    return gamma * normalized_x + beta

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # TODO: multiply x by the boolean keep_mask and rescale by 1/keep_prob.
    return (x * keep_mask) / keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head self-attention on x and wrap with residual add-and-norm.
    
    batch_size, seq_len, d_model = x.size()
    d_n = d_model // num_heads

    # Project X to q, k, v
    query_proj = x @ w_q.t()
    key_proj = x @ w_k.t()
    val_proj = x @ w_v.t()

    # Split into Heads 
    q = query_proj.reshape(batch_size, seq_len, num_heads, d_n).transpose(1,2)
    k = key_proj.reshape(batch_size, seq_len, num_heads, d_n).transpose(1,2)
    v = val_proj.reshape(batch_size, seq_len, num_heads, d_n).transpose(1,2)

    # Compute Scaled Attention Product
    attention_raw = (q @ k.transpose(-2, -1)) / math.sqrt(d_n)
    if src_mask is not None:
        attention_raw = attention_raw.masked_fill(src_mask == False, -float('inf'))
    
    weights = torch.softmax(attention_raw, dim=-1)
    scaled_dot_product = weights @ v

    # Merge Heads back to Model dim
    merged_head = scaled_dot_product.transpose(1,2)
    merged_head = scaled_dot_product.reshape(batch_size, seq_len, d_model)
    
    sublayer_output = merged_head @ w_o.t()

    # Apply Residual 
    x = x + sublayer_output
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_normalized = (x - mean) / torch.sqrt(1e-5 + var)

    return gamma * x_normalized + beta

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on x and wrap it with residual add-and-norm.

    # 1. FFN
    layer_1 = x @ w1 + b1 
    relu = F.relu(layer_1)
    layer_2 = relu @ w2 + b2
    
    # 2. Residual 
    out_layer = x + layer_2
    mean = out_layer.mean(dim=-1, keepdim=True) 
    var = out_layer.var(dim=-1, keepdim=True, unbiased=False)
    out_layer_norm = (out_layer - mean) / torch.sqrt(var + 1e-5)

    return gamma * out_layer_norm + beta

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    b, l, d_model = x.size()
    d_n = d_model // num_heads

    query = (x @ layer_params['w_q'].t()).reshape(b, l, num_heads, d_n).transpose(1, 2)
    key   = (x @ layer_params['w_k'].t()).reshape(b, l, num_heads, d_n).transpose(1, 2)
    value = (x @ layer_params['w_v'].t()).reshape(b, l, num_heads, d_n).transpose(1, 2)

    raw_attention = (query @ key.transpose(-2, -1)) / math.sqrt(d_n)   # fix 2: was missing scaling
    if src_mask is not None:
        raw_attention = raw_attention.masked_fill(src_mask == False, -float('inf'))
    weights = torch.softmax(raw_attention, dim=-1)
    context = weights @ value                                          # fix 1: was raw_attention @ value

    merged_head = context.transpose(1, 2).reshape(b, l, d_model)
    attention_sublayer_output = merged_head @ layer_params['w_o'].t()  # fix 3: was missing .t()

    attention_output = apply_residual_add_and_norm(
        x, attention_sublayer_output, layer_params['attn_gamma'], layer_params['attn_beta'])

    layer_1 = attention_output @ layer_params['w1'] + layer_params['b1']
    relu = F.relu(layer_1)
    layer_2 = relu @ layer_params['w2'] + layer_params['b2']
    return apply_residual_add_and_norm(
        attention_output, layer_2, layer_params['ffn_gamma'], layer_params['ffn_beta'])

# Step 42 - stack_encoder_layers
# def assemble_encoder_layer(x, layer_params, num_heads, src_mask): 
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # TODO: sequentially apply each encoder layer to the running hidden state and return the final tensor.
    for encoder_layer_param in encoder_layer_params_list:
        encoder = assemble_encoder_layer(x, encoder_layer_param, num_heads, src_mask)
        x = encoder
    return x

# Step 43 - decoder_layer_masked_self_attention_sublayer
import math
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    batch_size, tgt_seq_len, d_model = y.size()
    d_n = d_model // num_heads

    query = (y @ w_q.t()).reshape(batch_size, tgt_seq_len, num_heads, d_n).transpose(1, 2)
    key   = (y @ w_k.t()).reshape(batch_size, tgt_seq_len, num_heads, d_n).transpose(1, 2)
    value = (y @ w_v.t()).reshape(batch_size, tgt_seq_len, num_heads, d_n).transpose(1, 2)

    raw_attention = (query @ key.transpose(-2, -1)) / math.sqrt(d_n)  # (B, H, T, T)

    if tgt_mask is not None:
        if tgt_mask.dim() == 2:          # (T, T) causal mask
            mask = tgt_mask[None, None, :, :]      # (1, 1, T, T)
        elif tgt_mask.dim() == 3:        # (B, T, T)
            mask = tgt_mask[:, None, :, :]         # (B, 1, T, T)
        else:                            # already (B, H/1, T, T)
            mask = tgt_mask
        raw_attention = raw_attention.masked_fill(mask == False, -float('inf'))

    weights = torch.softmax(raw_attention, dim=-1)
    context = weights @ value                       # (B, H, T, d_n)

    merged_heads = context.transpose(1, 2).reshape(batch_size, tgt_seq_len, d_model)
    sublayer_output = merged_heads @ w_o.t()

    return apply_residual_add_and_norm(y, sublayer_output, gamma, beta)

# Step 44 - decoder_layer_cross_attention_sublayer
import math
import torch

def _expand_attn_mask(mask, batch_size, key_len):
    """Normalize a boolean attention mask to (B or 1, 1 or H, T_q, T_k)."""
    if mask.dim() == 2:
        if mask.size(0) == batch_size and mask.size(1) == key_len and mask.size(0) != mask.size(1):
            return mask[:, None, None, :]        # (B, S) padding mask
        elif mask.size(0) == mask.size(1) == key_len:
            return mask[None, None, :, :]        # (T, T) causal mask
        else:
            return mask[:, None, None, :]        # fall back to padding interpretation
    elif mask.dim() == 3:
        return mask[:, None, :, :]               # (B, T_q, T_k)
    return mask                                  # already 4D                          # already 4D: (B, 1/H, T_q, T_k)


def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o,
                                           gamma, beta, num_heads, src_mask):
    batch_size, tgt_seq_len, d_model = y.size()
    _, src_seq_len, _ = encoder_output.size()
    d_n = d_model // num_heads

    # Q from decoder input, K/V from encoder output
    query = (y @ w_q.t()).reshape(batch_size, tgt_seq_len, num_heads, d_n).transpose(1, 2)
    key   = (encoder_output @ w_k.t()).reshape(batch_size, src_seq_len, num_heads, d_n).transpose(1, 2)
    value = (encoder_output @ w_v.t()).reshape(batch_size, src_seq_len, num_heads, d_n).transpose(1, 2)

    raw_attention = (query @ key.transpose(-2, -1)) / math.sqrt(d_n)  # (B, H, T, S)

    if src_mask is not None:
        mask = _expand_attn_mask(src_mask, batch_size)
        raw_attention = raw_attention.masked_fill(~mask, -float('inf'))

    weights = torch.softmax(raw_attention, dim=-1)
    context = weights @ value                                          # (B, H, T, d_n)

    merged_heads = context.transpose(1, 2).reshape(batch_size, tgt_seq_len, d_model)
    sublayer_output = merged_heads @ w_o.t()

    return apply_residual_add_and_norm(y, sublayer_output, gamma, beta)

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on y and wrap it with residual add-and-norm

    layer_1 = y @ w1 + b1
    relu = F.relu(layer_1)
    layer_2 = relu @ w2 + b2 
    return apply_residual_add_and_norm(y, layer_2, gamma, beta)

# Step 46 - assemble_decoder_layer
import torch

def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """
    Run a full decoder layer: masked self-attention, cross-attention, then FFN.
    """
    
    # 1. Masked Self-Attention
    # The decoder looks at itself (y) to find context, using the target mask 
    # to hide future words from being seen.
    masked_attn_out = decoder_layer_masked_self_attention_sublayer(
        y, 
        layer_params['w_q_self'], 
        layer_params['w_k_self'], 
        layer_params['w_v_self'], 
        layer_params['w_o_self'],
        layer_params['self_gamma'], 
        layer_params['self_beta'], 
        num_heads, 
        tgt_mask
    ) 

    # 2. Cross-Attention
    # The decoder takes its own context (masked_attn_out) as the Query, and looks 
    # at the encoder_output for the Keys and Values using the source mask.
    cross_attn_out = decoder_layer_cross_attention_sublayer(
        masked_attn_out, 
        encoder_output, 
        layer_params['w_q_cross'], 
        layer_params['w_k_cross'], 
        layer_params['w_v_cross'], 
        layer_params['w_o_cross'],
        layer_params['cross_gamma'], 
        layer_params['cross_beta'], 
        num_heads, 
        src_mask
    ) 

    # 3. Feed-Forward Network
    # Finally, the context is processed through the FFN.
    ffn_out = decoder_layer_feed_forward_sublayer(
        cross_attn_out, 
        layer_params['w1'], 
        layer_params['b1'], 
        layer_params['w2'], 
        layer_params['b2'], 
        layer_params['ffn_gamma'], 
        layer_params['ffn_beta']
    )
    
    return ffn_out

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    # TODO: sequentially apply each decoder layer to the running target hidden state.
    for layer_params in decoder_layer_params_list:
        layer = assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask)
        y = layer 
    return y

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # TODO: project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).

    decoder_output = decoder_output @ output_projection_weight.t()
    if output_projection_bias is not None:
        decoder_output = decoder_output + output_projection_bias
    return decoder_output

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    # TODO: return an output projection weight tied to the token embedding matrix
    return token_embedding_weight.t()

# Step 50 - apply_log_softmax_over_vocab
def apply_log_softmax_over_vocab(logits):
    # TODO: Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.

    return torch.log_softmax(logits, dim=-1)

# Step 51 - run_transformer_forward
def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    # TODO: embed src+tgt, add PE, build masks, run encoder/decoder, project to log probs.

    batch_size, seq_len = tgt_ids.size() 

    # 1. Embed Src + Tgt
    token_weights = model_params['token_embedding']
    _, d_model = token_weights.size()

    src_embeddings = token_weights[src_ids] * math.sqrt(d_model)
    tgt_embeddings = token_weights[tgt_ids] * math.sqrt(d_model)

    src_seq_len = src_ids.size(1)
    tgt_seq_len = tgt_ids.size(1)
    max_seq_len = max(src_seq_len, tgt_seq_len)

    # 2. Add Positional Encoding
    _, d_model = model_params['token_embedding'].size()
    pe = torch.zeros(max_seq_len, d_model, device=src_ids.device) 
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    # exp(2i * -log(10000)/d_model)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.log(10000.0) / d_model)

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    src_embeddings = src_embeddings + pe[:src_seq_len, :]
    tgt_embeddings = tgt_embeddings + pe[:tgt_seq_len, :]

    # 3. Masks
    # (batch, seq_len) --> (batch, 1, 1, seq_len)
    src_mask = (src_ids != pad_id).unsqueeze(1).unsqueeze(2)
    # (seq_len, seq_len) --> (1, 1, seq_len, seq_len) --> (batch, 1, seq_len, seq_len)
    tgt_padding_mask = (tgt_ids != pad_id).unsqueeze(1).unsqueeze(2)
    tgt_causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(1)
    tgt_mask = tgt_padding_mask & tgt_causal_mask

    # 4. Run Encoder 
    enc_output = src_embeddings
    encoder_layers = model_params['encoder_layers']
    for encoder_layer in encoder_layers:
        enc = assemble_encoder_layer(enc_output, encoder_layer, num_heads, src_mask)
        enc_output = enc
    
    # 5. Run Decoder
    dec_output = tgt_embeddings
    decoder_layers = model_params['decoder_layers']
    for decoder_layer in decoder_layers:
        dec = assemble_decoder_layer(dec_output, enc_output, decoder_layer, num_heads, src_mask, tgt_mask)
        dec_output = dec

    # 6. Project to Vocabulary Logits
    proj_weight = model_params['output_projection'] 
    logits = dec_output @ proj_weight.t()

    # 7. Log Probabilities
    return torch.log_softmax(logits, dim=-1)

# Step 52 - init_encoder_layer_parameters
import torch
import math

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    # TODO: allocate w_q, w_k, w_v, w_o, w1, b1, w2, b2, attn_gamma, attn_beta, ffn_gamma, ffn_beta.
    
    # 1. Allocation Attention Weight Matrices
    # Data (batch, seq, d_model) @ Weights (d_model, d_model) = Output (batch, seq, d_model)

    # 2. Allocation FFN internal matrices
    # Expands d_model to d_ff then shrinks down

    # 3. Layer Norm Initializes

    def make_weights(*shape):
        tensor = torch.randn(*shape, dtype=torch.float32) * 0.02
        tensor.requires_grad=True
        return tensor
           
    def make_bias(*shape, init_val=0.0):
        tensor = torch.full(shape, init_val, dtype=torch.float32)
        tensor.requires_grad=True
        return tensor
    
    return {
        'w_q' : make_weights(d_model, d_model),
        'w_k' : make_weights(d_model, d_model),
        'w_v' : make_weights(d_model, d_model),
        'w_o' : make_weights(d_model, d_model),

        'w1' : make_weights(d_model, d_ff),
        'b1' : make_bias(d_ff, init_val=0.0),
        'w2' : make_weights(d_ff, d_model),
        'b2' : make_bias(d_model, init_val=0.0),

        'attn_gamma' : make_bias(d_model, init_val=1.0),
        'attn_beta':  make_bias(d_model, init_val=0.0),
        'ffn_gamma':  make_bias(d_model, init_val=1.0),
        'ffn_beta':   make_bias(d_model, init_val=0.0)
    }

# Step 53 - init_decoder_layer_parameters
import torch

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    # TODO: return a dict of requires_grad tensors for one decoder layer
    def make_weights(*shape):
        tensor = torch.randn(shape, dtype=torch.float32)
        tensor.requires_grad=True
        return tensor

    def make_bias(*shape, init_val=0.0):
        tensor = torch.full(shape, init_val, dtype=torch.float32)
        tensor.requires_grad=True
        return tensor

    return {
        'w_q_self' : make_weights(d_model, d_model),
        'w_k_self' : make_weights(d_model, d_model),
        'w_v_self' : make_weights(d_model, d_model),
        'w_o_self' : make_weights(d_model, d_model),

        'w_q_cross' : make_weights(d_model, d_model),
        'w_k_cross' : make_weights(d_model, d_model),
        'w_v_cross' : make_weights(d_model, d_model),
        'w_o_cross' : make_weights(d_model, d_model),

        'w1' : make_weights(d_model, d_ff),
        'b1' : make_bias(d_ff, init_val=0.0),
        'w2' : make_weights(d_ff, d_model),
        'b2' : make_bias(d_model, init_val=0.0),

        'self_gamma' : make_bias(d_model, init_val=1.0),
        'self_beta':  make_bias(d_model, init_val=0.0),
        'cross_gamma':  make_bias(d_model, init_val=1.0),
        'cross_beta':   make_bias(d_model, init_val=0.0),
        'ffn_gamma':  make_bias(d_model, init_val=1.0),
        'ffn_beta':   make_bias(d_model, init_val=0.0)
    }

# Step 54 - init_embedding_and_projection_parameters
import torch

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    # TODO: allocate three (vocab_size, d_model) tensors with requires_grad=True

    def make_embedding(*shape):
        tensor = torch.randn(shape, dtype=torch.float32)
        tensor.requires_grad = True
        return tensor
    
    src_embedding = make_embedding(vocab_size, d_model)
    tgt_embedding = make_embedding(vocab_size, d_model)

    return {
        'src_embedding': src_embedding,
        'tgt_embedding': tgt_embedding,
        'output_projection' : tgt_embedding if tie_weights else make_embedding(vocab_size, d_model)
    }

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(encoder_layer_params, decoder_layer_params, embedding_params):
    # TODO: walk the encoder, decoder, and embedding dicts and return a flat deduped list of tensors
    
    params_list = []
    seen_ids = set() 

    def add_tensor(tensor):
        t_id = id(tensor)
        if t_id not in seen_ids:
            seen_ids.add(t_id)
            params_list.append(tensor)

    for layer_dict in encoder_layer_params:
        for tensor in layer_dict.values():
            add_tensor(tensor)
    
    for layer_dict in decoder_layer_params:
        for tensor in layer_dict.values():
            add_tensor(tensor)
    
    for tensor in embedding_params.values():
        add_tensor(tensor)

    return params_list

# Step 56 - shift_targets_right_with_start_token
def shift_targets_right_with_start_token(target_ids, start_token_id):
    # TODO: prepend start_token_id and drop the last column so output shape matches target_ids
    start_col = target_ids.new_full((target_ids.size(0), 1), start_token_id)
    return torch.cat([start_col, target_ids[:, :-1]], dim=1)

# Step 57 - compute_noam_learning_rate
def compute_noam_learning_rate(step, d_model, warmup_steps):
    # TODO: return the Noam warmup learning rate for the given step.
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps**-1.5)

# Step 58 - build_uniform_smoothing_distribution
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    # TODO: return a float tensor of `shape` filled with epsilon / (vocab_size - 2).
    value = epsilon / (vocab_size - 2)
    return torch.full(shape, value)

# Step 59 - set_confidence_on_gold_tokens
import torch

def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""
    # TODO: write the confidence value at each gold token id along the vocab axis
    
    # (batch_size, tgt_seq, vocab_size)
    result = smoothed_distribution.clone()
    result = result.scatter_(2, gold_token_ids.unsqueeze(2), confidence)
    return result

# Step 60 - zero_pad_column_and_pad_token_rows
import torch

def zero_pad_column_and_pad_token_rows(smoothed_distribution, gold_token_ids, pad_id):
    # TODO: zero the pad column and the rows where the gold token equals pad_id

    result = smoothed_distribution.clone()
    
    #1. Zero pad column
    result[:, :, pad_id] = 0.0
    
    #2. Zero rows where the gold token equals pad_id
    pad_mask = (gold_token_ids == pad_id).unsqueeze(2)
    result = result.masked_fill(pad_mask, 0.0)
    return result

# Step 61 - compute_label_smoothed_kl_loss
import torch

def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    """Return the summed KL loss over all (batch, time, vocab) entries."""
    # TODO: combine log_probabilities with the smoothed target distribution into a scalar loss
    
    loss = log_probabilities * smoothed_distribution
    loss = torch.nan_to_num(loss, nan=0.0)
    return -torch.sum(loss) + 0.0

# Step 62 - average_loss_over_non_pad_tokens
import torch

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    non_pad_count = (gold_token_ids != pad_id).sum()

    return total_loss / (non_pad_count if non_pad_count > 0 else 1)

# Step 63 - compute_token_accuracy_ignoring_pad
import torch

def compute_token_accuracy_ignoring_pad(log_probabilities, gold_token_ids, pad_id):
    
    predictions = log_probabilities.argmax(dim=-1)
    non_pad_mask = (gold_token_ids != pad_id)
    correct_predictions = (predictions == gold_token_ids) & non_pad_mask

    if non_pad_mask.sum().float() == 0:
            return torch.tensor(0.0)
    return correct_predictions.sum().float() / non_pad_mask.sum().float()

# Step 64 - initialize_adam_optimizer_state
import torch

def initialize_adam_optimizer_state(parameter_list):
    """Allocate Adam m, v zero buffers and a step counter t=0."""
    # TODO: allocate zero buffers for first and second moments, plus step counter
    
    m = []
    v = []
    for parameter in parameter_list:
        m.append(torch.zeros_like(parameter, requires_grad=False))
        v.append(torch.zeros_like(parameter, requires_grad=False))

    return {
        'm': m, 
        'v': v,
        't': 0
    }

# Step 65 - update_adam_first_moment
import torch

def update_adam_first_moment(m_prev, grad, beta1):
    """Return m_t = beta1 * m_prev + (1 - beta1) * grad."""
    # TODO: apply the Adam first-moment EMA update and return the new tensor
    
    return m_prev * beta1 + (1 - beta1) * grad

# Step 66 - update_adam_second_moment
import torch

def update_adam_second_moment(v_prev, grad, beta2):
    """Return v_t = beta2 * v_prev + (1 - beta2) * grad ** 2."""
    # TODO: apply Adam's EMA update for the second moment of the gradient
    
    return beta2 * v_prev + (1 - beta2) * grad ** 2

# Step 67 - apply_adam_bias_correction
import torch

def apply_adam_bias_correction(m_t, v_t, beta1, beta2, step):
    """Return bias-corrected (m_hat, v_hat) for Adam at the given step."""
    # TODO: divide each moment by (1 - beta**step) using its respective beta
    
    return (m_t / (1 - beta1**step), v_t / (1 - beta2**step))

# Step 69 - apply_adam_step_to_all_parameters
import torch

def apply_adam_step_to_all_parameters(parameter_list, optimizer_state, learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9):
    # TODO: increment t, then for each param with a grad update m, v, bias-correct, and subtract delta in place.
    
    # 1. Increment t
    optimizer_state['t'] += 1 
    t = optimizer_state['t'] 

    # 2. For each param with a grad, update
    for i, parameter in enumerate(parameter_list):
        if parameter.grad is None:
            continue
        
        grad = parameter.grad 
        optimizer_state['m'][i] = beta1 * optimizer_state['m'][i] + (1 - beta1) * grad 
        optimizer_state['v'][i] = beta2 * optimizer_state['v'][i] + (1 - beta2) * grad * grad

        m_hat = optimizer_state['m'][i] / (1 - beta1 ** t)
        v_hat = optimizer_state['v'][i] / (1 - beta2 ** t)

        with torch.no_grad():
            parameter -= learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)
    
    return optimizer_state

# Step 70 - zero_all_parameter_gradients
import torch

def zero_all_parameter_gradients(parameter_list):
    """Clear the .grad of every parameter tensor before the next backward pass."""
    
    for parameter in parameter_list: 
        parameter.grad = None

# Step 71 - compute_batch_training_loss
import torch
import torch.nn.functional as F 
import math

def compute_batch_training_loss(src_batch, tgt_batch, model_params, config):
    # 1. Build Decoder 
    gold_targets = tgt_batch.clone()
    pad_id = config['pad_id']
    start_token_id = config['start_id']
    vocab_size = config['vocab_size']
    epsilon = config['smoothing']
    num_heads = config['num_heads']

    start_col = tgt_batch.new_full((tgt_batch.size(0), 1), start_token_id)
    tgt_batch = torch.cat([start_col, tgt_batch[:, :-1]], dim=1)

    # 2. run transformer forward pass
    # a. embed src and tgt
    src_weights = model_params['src_embedding']
    tgt_weights = model_params['tgt_embedding']
    _, d_model = src_weights.size()
    device = src_weights.device

    src_embeddings = src_weights[src_batch] * math.sqrt(d_model)
    tgt_embeddings = tgt_weights[tgt_batch] * math.sqrt(d_model)

    src_seq_len = src_batch.size(1)
    tgt_seq_len = tgt_batch.size(1)
    max_seq_len = max(src_seq_len, tgt_seq_len)

    # b. add positional encodings
    pe = torch.zeros(max_seq_len, d_model, device=device)
    position = torch.arange(0, max_seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * -math.log(10000.0) / d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    src_embeddings = src_embeddings + pe[:src_seq_len, :]
    tgt_embeddings = tgt_embeddings + pe[:tgt_seq_len, :]

    # c. masks (Expanded explicitly to avoid strict broadcasting failures)
    src_key_mask = (src_batch != pad_id).unsqueeze(1).unsqueeze(2) # (B, 1, 1, S_s)
    
    # Encoder self-attention mask
    src_mask = src_key_mask.expand(-1, -1, src_seq_len, -1)
    # Decoder cross-attention mask
    src_mask_for_decoder = src_key_mask.expand(-1, -1, tgt_seq_len, -1)
    
    # Decoder self-attention mask
    tgt_key_mask = (tgt_batch != pad_id).unsqueeze(1).unsqueeze(2)
    tgt_causal_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(1)
    tgt_mask = (tgt_key_mask & tgt_causal_mask).expand(-1, -1, tgt_seq_len, -1)
     
    # d. run encoder
    enc_output = src_embeddings
    for encoder_layer in model_params['encoder_layers']:
        enc_output = assemble_encoder_layer(enc_output, encoder_layer, num_heads, src_mask)

    # e. run decoder
    dec_output = tgt_embeddings
    for decoder_layer in model_params['decoder_layers']:
        dec_output = assemble_decoder_layer(dec_output, enc_output, decoder_layer, num_heads, src_mask_for_decoder, tgt_mask)

    # f. project to vocabulary logits
    proj_weight = model_params['output_projection']
    logits = dec_output @ proj_weight.t() 

    # g. log probabilities
    probabilities = torch.log_softmax(logits, dim=-1) 

    # 3. build smoothed targets 
    # Create entirely detached from probabilities graph to pass strict autograd assertions
    smoothed_targets = torch.full(
        probabilities.shape, 
        epsilon / (vocab_size - 2), 
        device=device, 
        dtype=torch.float
    )
    
    smoothed_targets.scatter_(dim=-1, index=gold_targets.unsqueeze(-1), value=1.0 - epsilon)
    
    # Zero out the column for pad_id
    smoothed_targets[:, :, pad_id] = 0.0
    
    # Zero out the rows completely where gold target is pad_id
    pad_mask = (gold_targets == pad_id).unsqueeze(-1)
    smoothed_targets.masked_fill_(pad_mask, 0.0)

    # 4. average the KL loss over non-pad tokens 
    kl_loss = F.kl_div(probabilities, smoothed_targets)
    per_token_loss = kl_loss.sum(dim=-1)

    # Guarantee padded tokens contribute exactly 0 to the sum
    non_pad_mask = (gold_targets != pad_id)
    per_token_loss = per_token_loss.masked_fill(~non_pad_mask, 0.0)
    
    loss = per_token_loss.sum() / non_pad_mask.sum()

    return loss

def _expand_attn_mask(mask, batch_size):
    """Normalize a boolean attention mask to (B or 1, 1 or H, T_q, T_k)."""
    if mask.dim() == 2:
        if mask.size(0) == batch_size:      # (B, S) padding mask
            return mask[:, None, None, :]
        else:                               # (T, T) causal mask
            return mask[None, None, :, :]
    elif mask.dim() == 3:                   # (B, T_q, T_k)
        return mask[:, None, :, :]
    return mask                             # already 4D

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

