# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import ColumnParallelLinear, ParallelEmbedding, RowParallelLinear
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    adapter_len: int = 10
    adapter_layer: int = 8
    
    quantizer: bool=False


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


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)).cuda()
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)).cuda()
        self.gate = torch.nn.Parameter(torch.zeros(1, self.n_local_heads, 1, 1))

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], random_init, adapter=None
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        if adapter is not None:
            adapter_len = adapter.shape[1]
            adapter_k = self.wk(adapter).view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            adapter_v = self.wv(adapter).view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            adapter_k = adapter_k.transpose(1, 2)
            adapter_v = adapter_v.transpose(1, 2)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores_soft = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores_soft, values)  # (bs, n_local_heads, slen, head_dim)
        if adapter is not None and not random_init:
            adapter_scores = torch.matmul(xq, adapter_k.transpose(2, 3)) / math.sqrt(self.head_dim)
            adapter_scores_soft = self.gate * F.softmax(adapter_scores.float(), dim=-1).type_as(xq)
            output = output + torch.matmul(adapter_scores_soft, adapter_v)
        elif adapter is not None and random_init:
            adapter_scores = torch.matmul(xq, adapter_k.transpose(2, 3)) / math.sqrt(self.head_dim)
            # print(adapter_scores.shape)
            # print(scores.shape)
            total_score = torch.cat([adapter_scores, scores], dim=3)
            total_score_soft = F.softmax(total_score.float(), dim=-1).type_as(xq)
            total_value = torch.cat([adapter_v, values], dim=2)
            
            output = torch.matmul(total_score_soft, total_value)
            
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self.w2 = RowParallelLinear(hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x)
        self.w3 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], random_init, typ_act=None, adapter=None
    ):
        # h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, random_init, adapter)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, random_init, adapter)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

def hyperfanin_init_weight(linear_layer, hypernet_in, mainnet_in):
    bound = 1e-3 * math.sqrt(3 / (hypernet_in * mainnet_in))
    nn.init.uniform_(linear_layer.weight, -bound, bound)
    nn.init.constant_(linear_layer.bias, 0.0)

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
      
class HyperModel(nn.Module):
    def __init__(self, input_dim, output_dim, adapter_len, hidden_dim, acti_func):
        # output_dim: Main Network output_dim
        # input_dim: Main Network in_dim
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        print("===========> Hidden dim of hypermodel is:", self.hidden_dim)
        if self.hidden_dim == 0:
            print("===========> Use MLP one layer.")
            self.mlp_1_layer = True
            self.linear1 = IdentityMap()
            self.hidden_dim = self.input_dim
        else:
            print("===========> Use MLP two layer.")
            self.mlp_1_layer = False
            self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        print("===========> Type of activation in hypernetwork:", acti_func)
        if acti_func == "relu":
            self.activation_fn = nn.ReLU()
        elif acti_func == "leaky_relu":
            self.activation_fn = nn.LeakyReLU(0.1)
        elif acti_func == "gelu":
            self.activation_fn = nn.GELU()
        elif acti_func == "tanh":
            self.activation_fn = nn.Tanh()
        elif acti_func == "sigmoid":
            self.activation_fn = nn.Sigmoid()
        else:
            raise Exception("Wrong type of activation for hypernetwork")
        # output weights
        self.prompt = nn.Linear(self.hidden_dim, output_dim * adapter_len)

        # init weights
        hyperfanin_init_weight(self.prompt, self.hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        if self.mlp_1_layer:
            return self.activation_fn(self.prompt(x))
        else:
            x = self.activation_fn(x)
            return self.prompt(x)

class ParameterGenerator(nn.Module):
    def __init__(self, input_size, output_size, adapter_len, num_hidden_layers, hidden_dim, acti_func):
        # output_dim: Main Network output_dim
        # input_dim: Main Network in_dim
        super().__init__()
        self.layer_embed = nn.Embedding(num_hidden_layers, input_size)
        self.decoder = HyperModel(
            input_size, output_size, adapter_len, hidden_dim, acti_func
        )
        self.output_size = output_size
        self.adapter_len = adapter_len

    def forward(self, layer_idx):
        layer_inputs = self.layer_embed(layer_idx)
        out = self.decoder(layer_inputs)
        return out.view(-1, self.adapter_len, self.output_size)

class DeepShare(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, acti_func):
        # output_dim: Main Network output_dim
        # input_dim: Main Network in_dim
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        print("===========> Hidden dim of hypermodel is:", self.hidden_dim)
        if self.hidden_dim == 0:
            print("===========> Use MLP one layer.")
            self.mlp_1_layer = True
            self.linear1 = IdentityMap()
            self.hidden_dim = self.input_dim
        else:
            print("===========> Use MLP two layer.")
            self.mlp_1_layer = False
            self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        print("===========> Type of activation in hypernetwork:", acti_func)
        if acti_func == "relu":
            self.activation_fn = nn.ReLU()
        elif acti_func == "leaky_relu":
            self.activation_fn = nn.LeakyReLU(0.1)
        elif acti_func == "gelu":
            self.activation_fn = nn.GELU()
        elif acti_func == "tanh":
            self.activation_fn = nn.Tanh()
        elif acti_func == "sigmoid":
            self.activation_fn = nn.Sigmoid()
        else:
            raise Exception("Wrong type of activation for hypernetwork")
        # output weights
        self.prompt = nn.Linear(self.hidden_dim, output_dim)

        # # init weights
        # hyperfanin_init_weight(self.prompt, self.hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        if self.mlp_1_layer:
            return self.activation_fn(self.prompt(x))
        else:
            x = self.activation_fn(x)
            return self.prompt(x)
           
class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(params.vocab_size, params.dim, init_method=lambda x: x)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(params.dim, params.vocab_size, bias=False, init_method=lambda x: x)

        self.freqs_cis = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2)
        # self.adapter_query = nn.Embedding(params.adapter_len * params.adapter_layer, params.dim)
        if params.typ_act == "deepshare":
            self.dim_layer = params.layer_dim
            self.hyperAdapter = DeepShare(self.dim_layer, self.params.dim, params.hidden_dim, params.hid_acti_func)
            self.adapter_query = nn.Embedding(params.adapter_len * params.adapter_layer, self.dim_layer)
        elif params.typ_act == "hypermodel":
            dim_layer = 64
            self.hyperAdapter = ParameterGenerator(dim_layer, self.params.dim, params.adapter_len, params.n_layers, params.hidden_dim, params.hid_acti_func)
        else:
            self.adapter_query = nn.Embedding(params.adapter_len * params.adapter_layer, params.dim)
        
        self.adapter_len = params.adapter_len
        self.adapter_layer = params.adapter_layer

    def sigma_func(self, x, name="identity"):
        if name == "identity" or name == "hypermodel" or name == "deepshare":
            return x
        else:
            raise Exception("Wrong type of activation")
        
    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int, random_init: bool, typ_act: str):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        if typ_act == "deepshare":
            prompt = self.adapter_query.weight.reshape(-1, self.adapter_len, self.dim_layer)
            prompt = self.hyperAdapter(prompt).unsqueeze(1)
        elif typ_act != "hypermodel":
            prompt = self.adapter_query.weight.reshape(
                self.params.adapter_layer, self.params.adapter_len, self.params.dim
            ).unsqueeze(1)
        else:
            layer_idx = torch.arange(self.n_layers - self.adapter_layer, self.n_layers).to(h.device)
            # print(layer_idx)
            prompt = self.hyperAdapter(layer_idx).unsqueeze(1)
            # print(prompt.shape)
        prompt = self.sigma_func(prompt, name=typ_act)
        
        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers[: -1 * self.params.adapter_layer]:
            h = layer(h, start_pos, freqs_cis, mask, random_init)
        layer_index = 0
        for layer in self.layers[-1 * self.params.adapter_layer :]:
            h = layer(h, start_pos, freqs_cis, mask, random_init, typ_act, prompt[layer_index])
            layer_index = layer_index + 1
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()
