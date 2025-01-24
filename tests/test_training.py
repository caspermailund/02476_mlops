import wandb
import os
import hydra
from image_classification.train import main


def test_training():
    wandb.login(key="5e435a892a1324586da2f4425116de5d843168f3")
    os.environ["WANDB_MODE"] = "dryrun"

    # Initialize Hydra with a custom config path
    with hydra.initialize(config_path="../configs"):
        cfg = hydra.compose(config_name="config_unit_test.yaml")

        # Convert relative paths to absolute paths
        cfg.data_folder = os.path.abspath(cfg.data_folder)
        cfg.model_save_path = os.path.abspath(cfg.model_save_path)

        main(cfg)
