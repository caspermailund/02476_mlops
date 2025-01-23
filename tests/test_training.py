from image_classification.train import main
import wandb
import hydra
import os


def test_training():
    wandb.login(key="5e435a892a1324586da2f4425116de5d843168f3")
    os.environ["WANDB_MODE"] = "dryrun"

    #Check that path exists
    mock_data_dir = os.path.join("data", "mock")
    assert os.path.exists(mock_data_dir), f"Directory does not exist: {mock_data_dir}"
    assert os.path.exists(os.path.join(mock_data_dir, "class_3")), "Class_3 folder is missing"

    # Initialize Hydra with a custom config path
    with hydra.initialize(config_path="../configs"):
        cfg = hydra.compose(config_name="config_unit_test.yaml")
        # Pass the configuration to your training function
        main(cfg)
