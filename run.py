import jax
from absl import app, flags, logging
from ml_collections import config_flags

import icl.utils as u
from icl.train import train

jax.distributed.initialize()

logging.set_verbosity(logging.INFO)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config")
flags.mark_flag_as_required("config")


def main(_):
    logging.info(f"Process: {jax.process_index() } / {jax.process_count()}")
    logging.info("Local Devices:\n" + "\n".join([str(x) for x in jax.local_devices()]) + "\n")

    config = u.filter_config(FLAGS.config)
    train(config)


if __name__ == "__main__":
    app.run(main)
