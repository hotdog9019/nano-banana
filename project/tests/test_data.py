from src.config import load_config
from src.data import load_dataset, make_split


def test_dataset_has_required_columns_and_classes():
    config = load_config()
    frame = load_dataset(config["data_path"], config["text_column"], config["target_column"])

    assert len(frame) >= 100
    assert config["text_column"] in frame.columns
    assert config["target_column"] in frame.columns
    assert frame[config["target_column"]].nunique() == 5


def test_split_is_stratified_enough_for_all_classes():
    config = load_config()
    frame = load_dataset(config["data_path"], config["text_column"], config["target_column"])
    bundle = make_split(
        frame,
        config["text_column"],
        config["target_column"],
        config["test_size"],
        config["random_state"],
    )

    assert set(bundle.train[config["target_column"]]) == set(bundle.test[config["target_column"]])
