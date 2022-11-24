from typing import Any, Dict

DATASETS_INFO: Dict[str, Dict[str, Any]] = {
    "mnist": {
        "expected_input_dimensions": (1, 32, 32),
        "classes": 10
    },
    "fashion-mnist": {
        "expected_input_dimensions": (1, 32, 32),
        "classes": 10
    },
    "cifar10": {
        "expected_input_dimensions": (3, 32, 32),
        "classes": 10
    },
    "cifar100": {
        "expected_input_dimensions": (3, 32, 32),
        "classes": 100
    }
}
#, "svhn", "cifar10",
# "cifar100-fine", "cifar100-coarse", "tiny-imagenet"]
#INPUT_DIMENSIONS: Tuple[int, int, int] = (1, 32, 32)

OVERALL_BEST_FOLDER = "overall_best"
STATS_FOLDER_NAME = "statistics"
CHANNEL_INDEX = 1
MODEL_FILENAME = "model.pt"
WEIGHTS_FILENAME = "weights.pt"
SEPARATOR_CHAR = "-"