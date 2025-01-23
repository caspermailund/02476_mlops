from image_classification.train import main
import wandb
import hydra
import os


def test_training():
    wandb.login(key="5e435a892a1324586da2f4425116de5d843168f3")
    os.environ["WANDB_MODE"] = "dryrun"
    # Initialize Hydra with a custom config path
    with hydra.initialize(config_path="../configs"):
        cfg = hydra.compose(config_name="config_unit_test.yaml")
        # Pass the configuration to your training function
        main(cfg)
