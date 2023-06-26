from typing import Callable

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jax import Array

from icl.models import Model, get_model_name
from icl.tasks import Sampler, Task

# Preds = {
#     task_name: {model_name: Array[n_samples, n_points], ...},
#     ...
# }
Preds = dict[str, dict[str, Array]]


def mse(a: Array, b: Array) -> Array:
    return jnp.square(a - b).mean(0)


def get_oracle_step(task: Task) -> Callable[[Array, Array], Array]:
    def step(xs: Array, ws: Array) -> Array:
        preds = task.evaluate_oracle(xs, ws)
        return preds

    return step


def get_baseline_step(model: Model) -> Callable[[Array, Array], Array]:
    def step(data: Array, targets: Array) -> Array:
        preds = model(data, targets)
        return preds

    return step


def get_bsln_preds(train_task: Task, j_batch_samplers: dict[str, Sampler], n_samples: int, batch_size: int) -> Preds:
    # Initialize preds and compile oracle and baseline models
    preds = {}
    p_oracle = jax.pmap(get_oracle_step(train_task), axis_name="device")
    p_bsln_models = {
        get_model_name(model): jax.pmap(get_baseline_step(model), axis_name="device")
        for model in train_task.get_default_eval_models()
    }
    # Loop through eval tasks
    for task_name, j_sample_batch in j_batch_samplers.items():
        # Initialize task preds
        preds[task_name] = {"True": []}
        for model_name in p_bsln_models:
            preds[task_name][model_name] = []
        # Accumulate preds...
        for i in range(1, n_samples // batch_size + 1):
            xs, ws, ys = j_sample_batch(i)
            _, _, n_points = ys.shape
            preds[task_name]["True"].append(p_oracle(xs, ws).reshape(batch_size, n_points))  # ...for oracle
            for model_name, p_model in p_bsln_models.items():  # ...for baseline models
                preds[task_name][model_name].append(p_model(xs, ys).reshape(batch_size, n_points))
        # Concatenate preds
        preds[task_name]["True"] = jnp.concatenate(preds[task_name]["True"])
        for model_name in p_bsln_models:
            preds[task_name][model_name] = jnp.concatenate(preds[task_name][model_name])
    return preds


def get_model_preds(
    state: TrainState,
    p_eval_step: Callable[[TrainState, Array, Array], Array],
    j_batch_samplers: dict[str, Sampler],
    n_samples: int,
    batch_size: int,
) -> Preds:
    preds = {}
    for task_name, j_sample_batch in j_batch_samplers.items():
        preds[task_name] = {"Transformer": []}
        for i in range(1, n_samples // batch_size + 1):
            xs, _, ys = j_sample_batch(i)
            _, _, n_points = ys.shape
            preds[task_name]["Transformer"].append(p_eval_step(state, xs, ys).reshape(batch_size, n_points))
        preds[task_name]["Transformer"] = jnp.concatenate(preds[task_name]["Transformer"])
    return preds
