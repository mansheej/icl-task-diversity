from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax import Array

import icl.utils as u
from icl.gpt2 import GPT2Config, GPT2Model, init_fn

tfd = tfp.distributions


########################################################################################################################
# Utilities                                                                                                            #
########################################################################################################################


def get_model_name(model):
    if isinstance(model, Ridge):
        return "Ridge"
    elif isinstance(model, DiscreteMMSE):
        return "dMMSE"
    elif isinstance(model, Transformer):
        return "Transformer"
    else:
        raise ValueError(f"model type={type(model)} not supported")


########################################################################################################################
# Transformer                                                                                                          #
########################################################################################################################


class Transformer(nn.Module):
    n_points: int
    n_layer: int
    n_embd: int
    n_head: int
    seed: int
    dtype: Any

    def setup(self):
        config = GPT2Config(
            block_size=2 * self.n_points,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dtype=self.dtype,
        )
        self._in = nn.Dense(self.n_embd, False, self.dtype, kernel_init=init_fn)
        self._h = GPT2Model(config)
        self._out = nn.Dense(1, False, self.dtype, kernel_init=init_fn)

    def __call__(self, data: Array, targets: Array, training: bool = False) -> Array:
        input_seq = u.to_seq(data, targets)
        embds = self._in(input_seq)
        outputs = self._h(input_embds=embds, training=training)
        preds = self._out(outputs)
        preds = u.seq_to_targets(preds)
        return preds


########################################################################################################################
# Ridge                                                                                                                #
########################################################################################################################


class Ridge(nn.Module):
    lam: float
    dtype: Any

    def __call__(self, data: Array, targets: Array) -> Array:
        """
        Args:
            xs: batch_size x n_points x n_dims (float)
            ys: batch_size x n_points (float)
        Return:
            batch_size x n_points (float)
        """
        batch_size, n_points, _ = data.shape
        targets = jnp.expand_dims(targets, -1)  # batch_size x n_points x 1
        preds = [jnp.zeros(batch_size, dtype=self.dtype)]
        preds.extend(
            [self.predict(data[:, :_i], targets[:, :_i], data[:, _i : _i + 1], self.lam) for _i in range(1, n_points)]
        )
        preds = jnp.stack(preds, axis=1)
        return preds

    def predict(self, X: Array, Y: Array, test_x: Array, lam: float) -> Array:
        """
        Args:
            X: batch_size x i x n_dims (float)
            Y: batch_size x i x 1 (float)
            test_x: batch_size x 1 x n_dims (float)
            lam: (float)
        Return:
            batch_size (float)
        """
        _, _, n_dims = X.shape
        XT = X.transpose((0, 2, 1))  # batch_size x n_dims x i
        XT_Y = XT @ Y  # batch_size x n_dims x 1, @ should be ok (batched matrix-vector product)
        ridge_matrix = jnp.matmul(XT, X, precision=jax.lax.Precision.HIGHEST) + lam * jnp.eye(n_dims, dtype=self.dtype)  # batch_size x n_dims x n_dims
        # batch_size x n_dims x 1
        ws = jnp.linalg.solve(ridge_matrix.astype(jnp.float32), XT_Y.astype(jnp.float32)).astype(self.dtype)
        pred = test_x @ ws  # @ should be ok (batched row times column)
        return pred[:, 0, 0]


########################################################################################################################
# MMSE                                                                                                                #
########################################################################################################################


class DiscreteMMSE(nn.Module):
    scale: float
    task_pool: Array  # n_tasks x n_dims x 1
    dtype: Any

    def __call__(self, data: Array, targets: Array) -> Array:
        """
        Args:
            data: batch_size x n_points x n_dims (float)
            targets: batch_size x n_points (float)
        Return:
            batch_size x n_points (float)
        """
        _, n_points, _ = data.shape
        targets = jnp.expand_dims(targets, -1)  # batch_size x n_points x 1
        W = self.task_pool.squeeze().T  # n_dims x n_tasks  (maybe do squeeze and transpose in setup?)
        preds = [data[:, 0] @ W.mean(axis=1)]  # batch_size
        preds.extend(
            [
                self.predict(data[:, :_i], targets[:, :_i], data[:, _i : _i + 1], W, self.scale)
                for _i in range(1, n_points)
            ]
        )
        preds = jnp.stack(preds, axis=1)  # batch_size x n_points
        return preds

    def predict(self, X: Array, Y: Array, test_x: Array, W: Array, scale: float) -> Array:
        """
        Args:
            X: batch_size x i x n_dims (float)
            Y: batch_size x i x 1 (float)
            test_x: batch_size x 1 x n_dims (float)
            W: n_dims x n_tasks (float)
            scale: (float)
        Return:
            batch_size (float)
        """
        # X @ W is batch_size x i x n_tasks, Y is batch_size x i x 1, so broadcasts to alpha being batch_size x n_tasks
        alpha = tfd.Normal(0, scale).log_prob(Y - jnp.matmul(X, W, precision=jax.lax.Precision.HIGHEST)).astype(self.dtype).sum(axis=1)
        # softmax is batch_size x n_tasks, W.T is n_tasks x n_dims, so w_mmse is batch_size x n_dims x 1
        w_mmse = jnp.expand_dims(jnp.matmul(jax.nn.softmax(alpha, axis=1), W.T, precision=jax.lax.Precision.HIGHEST), -1)
        # test_x is batch_size x 1 x n_dims, so pred is batch_size x 1 x 1. NOTE: @ should be ok (batched row times column)
        pred = test_x @ w_mmse
        return pred[:, 0, 0]


########################################################################################################################
# Get Model                                                                                                            #
########################################################################################################################

Model = Transformer | Ridge | DiscreteMMSE


def get_model(name: str, **kwargs) -> Model:
    models = {"transformer": Transformer, "ridge": Ridge, "discrete_mmse": DiscreteMMSE}
    return models[name](**kwargs)
