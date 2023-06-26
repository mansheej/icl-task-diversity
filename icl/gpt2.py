"""Implementation based on nanoGPT (https://github.com/karpathy/nanoGPT)
"""
from typing import Any

import flax.linen as nn
import jax.numpy as jnp
from flax import struct
from jax import Array


@struct.dataclass
class GPT2Config:
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True
    dtype: Any = jnp.float32


# Linear weights and Embedding weights are initialized with mean 0, stddev 0.02 normal random variables.
# Linear biases are initialized to 0 which is the default.
# Residual projection weights get scaled by a factor of 1/sqrt(2 * n_layer)
# Layer norm weights are initialized to 1 (default), biases are initialized to 0 (default), and epsilon is set to 1e-5
init_fn = nn.initializers.normal(0.02)
get_scaled_init_fn = lambda n_layer: nn.initializers.normal(0.02 / jnp.sqrt(2 * n_layer))


class GPT2SelfAttention(nn.Module):
    config: GPT2Config

    def setup(self):
        self.c_attn = nn.Dense(3 * self.config.n_embd, self.config.bias, self.config.dtype, kernel_init=init_fn)
        self.c_proj = nn.Dense(
            self.config.n_embd, self.config.bias, self.config.dtype, kernel_init=get_scaled_init_fn(self.config.n_layer)
        )
        self.attn_dropout = nn.Dropout(self.config.dropout)
        self.resid_droput = nn.Dropout(self.config.dropout)

    def __call__(self, x: Array, training: bool = False) -> Array:
        B, T, C = x.shape  # batch_size, block_size, n_embd
        q, k, v = jnp.split(self.c_attn(x), 3, axis=2)
        q = q.reshape(B, T, self.config.n_head, C // self.config.n_head).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        k = k.reshape(B, T, self.config.n_head, C // self.config.n_head).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        v = v.reshape(B, T, self.config.n_head, C // self.config.n_head).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        mask = jnp.tril(jnp.ones((1, 1, self.config.block_size, self.config.block_size))).astype(bool)
        att = q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(k.shape[-1])  # (B, nh, T, T)
        att = jnp.where(mask, att, jnp.finfo(self.config.dtype).min)
        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=not training)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.resid_droput(self.c_proj(y), deterministic=not training)
        return y


class GPT2MLP(nn.Module):
    config: GPT2Config

    def setup(self):
        self.c_fc = nn.Dense(4 * self.config.n_embd, self.config.bias, self.config.dtype, kernel_init=init_fn)
        self.c_proj = nn.Dense(
            self.config.n_embd, self.config.bias, self.config.dtype, kernel_init=get_scaled_init_fn(self.config.n_layer)
        )
        self.dropout = nn.Dropout(self.config.dropout)

    def __call__(self, x: Array, training: bool = False) -> Array:
        x = self.c_fc(x)
        x = nn.gelu(x, approximate=True)
        x = self.c_proj(x)
        x = self.dropout(x, deterministic=not training)
        return x


class GPT2Block(nn.Module):
    config: GPT2Config

    def setup(self):
        self.ln_1 = nn.LayerNorm(1e-5, self.config.dtype, use_bias=self.config.bias)
        self.attn = GPT2SelfAttention(self.config)
        self.ln_2 = nn.LayerNorm(1e-5, self.config.dtype, use_bias=self.config.bias)
        self.mlp = GPT2MLP(self.config)

    def __call__(self, x: Array, training: bool = False) -> Array:
        x = x + self.attn(self.ln_1(x), training=training)
        x = x + self.mlp(self.ln_2(x), training=training)
        return x


class GPT2Model(nn.Module):
    config: GPT2Config

    def setup(self):
        self.wpe = nn.Embed(self.config.block_size, self.config.n_embd, self.config.dtype, embedding_init=init_fn)
        self.drop = nn.Dropout(self.config.dropout)
        self.hs = [GPT2Block(self.config) for _ in range(self.config.n_layer)]
        self.ln_f = nn.LayerNorm(1e-5, self.config.dtype, use_bias=self.config.bias)

    def __call__(self, input_embds: Array, training: bool = False) -> Array:
        pos = jnp.expand_dims(jnp.arange(self.config.block_size), axis=0)  # (1, T)
        pos_embds = self.wpe(pos)  # (1, T, n_embd)
        x = input_embds + pos_embds  #  (B, T, n_embd)
        x = self.drop(x, deterministic=not training)
        for h in self.hs:
            x = h(x, training=training)
        x = self.ln_f(x)
        return x
