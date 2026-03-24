"""Configuration management module."""

import os
from pathlib import Path

import yaml

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the config file relative to project root

    Returns:
        Dictionary containing configuration
    """
    config_file = PROJECT_ROOT / config_path

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_data_path(relative_path: str) -> str:
    """Get absolute path for data files.

    Args:
        relative_path: Path relative to project root

    Returns:
        Absolute path as string
    """
    return str(PROJECT_ROOT / relative_path)


# Load default config
config = load_config()

# App settings
APP_NAME = config["app"]["name"]
APP_VERSION = config["app"]["version"]
PAGE_TITLE = config["app"]["page_title"]
LAYOUT = config["app"]["layout"]

# Data settings
RAW_DATA_PATH = get_data_path(config["data"]["raw_data_path"])
CLEAN_DATA_PATH = get_data_path(config["data"]["clean_data_path"])
TARGET_COLUMN = config["data"]["target_column"]

# Model settings
DEFAULT_THRESHOLD = config["model"]["default_threshold"]
RANDOM_STATE = config["model"]["random_state"]
TEST_SIZE = config["model"]["test_size"]

# Features
REQUIRED_FEATURES = config["features"]["required"]
DROP_COLUMNS = config["features"]["drop_columns"]

# Paths
MODELS_DIR = get_data_path(config["paths"]["models_dir"])
OUTPUTS_DIR = get_data_path(config["paths"]["outputs_dir"])
DEFAULT_MODEL_PATH = get_data_path(config["paths"]["default_model"])
SCALER_PATH = get_data_path(config["paths"]["scaler"])
