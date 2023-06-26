import json
import os

import jax
import jax.numpy as jnp
import jax.random as jr
import orbax.checkpoint as oc
import tensorflow as tf
from absl import logging
from flax import jax_utils
from flax.core import FrozenDict
from flax.training import checkpoints
from flax.training.train_state import TrainState
from jax import Array
from ml_collections import ConfigDict

import icl.utils as u
import wandb
from icl.evaluate import Preds, get_bsln_preds, get_model_preds, mse
from icl.models import Transformer, get_model
from icl.optim import get_optimizer_and_lr_schedule
from icl.tasks import Sampler, Task, get_task, get_task_name


def initialize(model: Transformer, config: ConfigDict) -> tuple[FrozenDict, Array]:
    params_rng, dropout_rng = jr.split(jr.PRNGKey(config.model.seed))
    dummy_data = jnp.ones((config.task.batch_size, config.task.n_points, config.task.n_dims), dtype=model.dtype)
    dummy_targets = jnp.ones((config.task.batch_size, config.task.n_points), dtype=model.dtype)
    variables = jax.jit(model.init)(params_rng, dummy_data, dummy_targets)
    return variables["params"], dropout_rng


def get_sharded_batch_sampler(task: Task) -> Sampler:
    n_devices = jax.local_device_count()

    def sample_batch(step: int) -> tuple[Array, Array, Array]:
        batch = task.sample_batch(step)
        batch = jax.tree_map(lambda x: x.reshape(n_devices, -1, *x.shape[1:]), batch)
        return batch

    return sample_batch


def train_step(state: TrainState, data: Array, targets: Array, dropout_rng: Array) -> TrainState:
    dropout_rng = jr.fold_in(dropout_rng, state.step + 1)

    def loss_fn(params):
        preds = state.apply_fn({"params": params}, data, targets, training=True, rngs={"dropout": dropout_rng})
        loss = jnp.square(preds - targets).mean()
        return loss, preds

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    _, grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name="device")
    state = state.apply_gradients(grads=grads)
    return state


def eval_step(state: TrainState, data: Array, targets: Array) -> Array:
    preds = state.apply_fn({"params": state.params}, data, targets, training=False)
    return preds


def _init_log(bsln_preds: Preds, n_dims: int) -> dict:
    log = {"train/step": [], "train/lr": []}
    for _task_name, _task_preds in bsln_preds.items():
        log[f"eval/{_task_name}"] = {}
        for _bsln_name, _bsln_preds in _task_preds.items():
            log[f"eval/{_task_name}"][f"Transformer | {_bsln_name}"] = []
            if _bsln_name != "True":
                _errs = mse(_bsln_preds, _task_preds["True"]) / n_dims
                log[f"eval/{_task_name}"][f"{_bsln_name} | True"] = _errs.tolist()
    return log


def train(config: ConfigDict) -> None:
    # Setup train experiment
    exp_name = f"train_{u.get_hash(config)}"
    exp_dir = os.path.join(config.work_dir, exp_name)
    logging.info(f"Train Experiment\nNAME: {exp_name}\nCONFIG:\n{config}")
    # Experiment completed?
    if tf.io.gfile.exists(os.path.join(exp_dir, "log.json")):
        logging.info(f"{exp_name} already completed")
        return None
    # Config
    tf.io.gfile.makedirs(exp_dir)
    with tf.io.gfile.GFile(os.path.join(exp_dir, "config.json"), "w") as f:
        f.write(config.to_json())

    # Model, optimizer and lr schedule
    model = get_model(**config.model, dtype=jnp.dtype(config.dtype))
    logging.info(u.tabulate_model(model, config.task.n_dims, config.task.n_points, config.task.batch_size))
    params, dropout_rng = initialize(model, config)
    tx, lr = get_optimizer_and_lr_schedule(**config.training, params=params)
    logging.info("Initialized Model, Optimizer and LR Schedule")

    # Train state
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    state = jax_utils.replicate(state)
    dropout_rngs = jr.split(dropout_rng, jax.local_device_count())
    logging.info("Initialized TrainState")

    # Data samplers
    train_task = get_task(**config.task, dtype=jnp.dtype(config.dtype))
    j_sample_train_batch = jax.jit(get_sharded_batch_sampler(train_task))
    j_samplers_eval_batch = {
        get_task_name(task): jax.jit(get_sharded_batch_sampler(task))
        for task in train_task.get_default_eval_tasks(**config.eval)
    }
    logging.info("Initialized Data Samplers")

    # Steps
    p_train_step = jax.pmap(train_step, axis_name="device", donate_argnums=0)
    p_eval_step = jax.pmap(eval_step, axis_name="device")
    logging.info("Pmap'd Steps")

    # Evaluate baselines
    logging.info("Evaluate Baselines...")
    bsln_preds = get_bsln_preds(train_task, j_samplers_eval_batch, config.eval.n_samples, config.eval.batch_size)

    # Loggers
    log = _init_log(bsln_preds, config.task.n_dims)
    wandb.init(config=config.to_dict(), name=exp_name, **config.wandb)

    # Training loop
    logging.info("Start Train Loop")
    for i in range(1, config.training.total_steps + 1):
        # Train step
        data, _, targets = j_sample_train_batch(i)
        state = p_train_step(state, data, targets, dropout_rngs)

        # Evaluate
        if i % config.eval.every == 0 or i == config.training.total_steps:
            # Log step and lr
            logging.info(f"Step: {i}")
            log["train/step"].append(i)
            log["train/lr"].append(lr(i).item())
            wandb.log({"train/lr": lr(i).item()}, step=i)
            # Evaluate model
            eval_preds = get_model_preds(
                state, p_eval_step, j_samplers_eval_batch, config.eval.n_samples, config.eval.batch_size
            )
            # Log model evaluation
            for _task_name, _task_preds in bsln_preds.items():
                for _bsln_name, _bsln_preds in _task_preds.items():
                    _errs = mse(eval_preds[_task_name]["Transformer"], _bsln_preds) / config.task.n_dims
                    log[f"eval/{_task_name}"][f"Transformer | {_bsln_name}"].append(_errs.tolist())
                    wandb.log({f"eval/{_task_name}/{_bsln_name}": _errs.mean().item()}, step=i)

    # Checkpoint
    ckpter = oc.AsyncCheckpointer(oc.PyTreeCheckpointHandler())
    checkpoints.save_checkpoint(exp_dir, jax_utils.unreplicate(state), i, orbax_checkpointer=ckpter)

    # Save logs
    with tf.io.gfile.GFile(os.path.join(exp_dir, "log.json"), "w") as f:
        f.write(json.dumps(log))

    # Wrap up
    ckpter.wait_until_finished()
    jr.normal(jr.PRNGKey(0)).block_until_ready()
    return None
