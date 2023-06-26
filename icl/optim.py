import jax.tree_util as tu
import optax


def get_optimizer_and_lr_schedule(
    optimizer: str, schedule: str, **kwargs
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    # Lr Schedule
    if schedule == "warmup_cosine_decay":
        lr = optax.warmup_cosine_decay_schedule(0.0, kwargs["lr"], kwargs["warmup_steps"], kwargs["total_steps"])
    elif schedule == "triangle":
        lr = optax.join_schedules(
            [
                optax.linear_schedule(0.0, kwargs["lr"], kwargs["warmup_steps"]),
                optax.linear_schedule(kwargs["lr"], 0.0, kwargs["total_steps"] - kwargs["warmup_steps"]),
            ],
            [kwargs["warmup_steps"]],
        )
    else:
        raise NotImplementedError
    # Optimizer
    if optimizer == "adam":
        tx = optax.adam(lr)
    # Weight decay mask based on nanoGPT (https://github.com/karpathy/nanoGPT)
    elif optimizer == "adamw":
        tx = optax.adamw(
            lr,
            weight_decay=kwargs["weight_decay"],
            mask=tu.tree_map_with_path(lambda kp, _: kp[0].key == "_h" and kp[-1].key == "kernel", kwargs["params"]),
        )
    else:
        raise NotImplementedError
    return tx, lr
