import filecmp
import os
import logging
import shutil
from typing import Any, Dict

from jsonschema import validate # type: ignore
import yaml # type: ignore

from fast_denser.misc.enums import TransformOperation


logger = logging.getLogger(__name__)


class Config():

    def __init__(self, path: str) -> None:
        self.config: Any = self._load(path)
        self._validate_config()
        os.makedirs(self.config['checkpoints_path'], exist_ok=True)
        self._backup_used_config(path, self.config['checkpoints_path'])

    def _load(self, path: str) -> Any:
        with open(path, "r", encoding="utf8") as f:
            return yaml.safe_load(f)

    def _backup_used_config(self, origin_filepath: str, destination: str) -> None:
        destination_filepath: str =  os.path.join(destination, "used_config.yaml")
        # if there is a config file backed up already and it is different than the one we are trying to backup
        if os.path.isfile(destination_filepath) and \
            filecmp.cmp(origin_filepath, destination_filepath) is False:
            raise ValueError("You are probably trying to continue an experiment "
                             "with a different config than the one you used initially. "
                             "This is a gentle reminder to double-check the config you "
                             "have just passed as parameter.")
        if not shutil._samefile(origin_filepath, destination_filepath): # type: ignore
            shutil.copyfile(origin_filepath, destination_filepath)

    def _validate_config(self) -> None:
        schema_path: str = os.path.join("fast_denser", "config", "schema.yaml")
        schema: Any = self._load(schema_path)
        validate(self.config, schema)
        logger.info(f"Type of training: {self.config['network']['learning']['learning_type']}")
        if self.config['network']['learning']['learning_type'] == "supervised" :
            if self.config['network']['learning']['augmentation']['train'] is not None:
                self._validate_augmentation_params(self.config['network']['learning']['augmentation']['train'])
            logger.info(f"Augmentation used in training: {self.config['network']['learning']['augmentation']['train']}")
        else:
            self._validate_augmentation_params(self.config['network']['learning']['augmentation']['train']['input_a'])
            self._validate_augmentation_params(self.config['network']['learning']['augmentation']['train']['input_b'])
            logger.info(f"Augmentation used for input a in training: {self.config['network']['learning']['augmentation']['train']['input_a']}")
            logger.info(f"Augmentation used for input b in training: {self.config['network']['learning']['augmentation']['train']['input_b']}")

        if self.config['network']['learning']['augmentation']['test'] is not None:
            self._validate_augmentation_params(self.config['network']['learning']['augmentation']['test'])

        logger.info(f"Augmentation used in test: {self.config['network']['learning']['augmentation']['test']}")

    def _validate_augmentation_params(self, params: Dict[str, Any]) -> None:
        for key in params.keys():
            assert key in TransformOperation.enum_values(), f"{key} is not recognised as one of the supported transforms"

    def __getitem__(self, key: str) -> Any:
        return self.config[key]
