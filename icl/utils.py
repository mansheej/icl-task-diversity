import hashlib

import jax.numpy as jnp
import jax.random as jr
from jax import Array
from ml_collections import ConfigDict

from icl.models import Transformer


def filter_config(config: ConfigDict) -> ConfigDict:
    with config.unlocked():
        for k, v in config.items():
            if v is None:
                del config[k]
            elif isinstance(v, ConfigDict):
                config[k] = filter_config(v)
    return config


def get_hash(config: ConfigDict) -> str:
    return hashlib.md5(config.to_json(sort_keys=True).encode("utf-8")).hexdigest()


def to_seq(data: Array, targets: Array) -> Array:
    batch_size, seq_len, n_dims = data.shape
    dtype = data.dtype
    data = jnp.concatenate([jnp.zeros((batch_size, seq_len, 1), dtype=dtype), data], axis=2)
    targets = jnp.concatenate([targets[:, :, None], jnp.zeros((batch_size, seq_len, n_dims), dtype=dtype)], axis=2)
    seq = jnp.stack([data, targets], axis=2).reshape(batch_size, 2 * seq_len, n_dims + 1)
    return seq


def seq_to_targets(seq: Array) -> Array:
    return seq[:, ::2, 0]


def tabulate_model(model: Transformer, n_dims: int, n_points: int, batch_size: int) -> str:
    dummy_data = jnp.ones((batch_size, n_points, n_dims), dtype=model.dtype)
    dummy_targets = jnp.ones((batch_size, n_points), dtype=model.dtype)
    return model.tabulate(jr.PRNGKey(0), dummy_data, dummy_targets, training=False, depth=0)
