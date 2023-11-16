import filecmp
import os
import logging
import shutil
from typing import Any, Dict, List, Optional

from jsonschema import validate # type: ignore
import yaml # type: ignore

from evodenss.misc.enums import TransformOperation
from evodenss.networks import ModuleConfig


logger = logging.getLogger(__name__)


class Config():

    def __init__(self, path: str) -> None:
        self.config: Any = self._load(path)
        self._validate_config()
        self.config['network']['learning']['projector'] = self._fix_static_projector_config_if_needed()
        self.config['network']['architecture']['modules'] = self._convert_modules_configs_to_dict()
        os.makedirs(self.config['checkpoints_path'], exist_ok=True)
        self._backup_used_config(path, self.config['checkpoints_path'])

    def _convert_modules_configs_to_dict(self) -> Dict[str, ModuleConfig]:
        modules_configurations: Dict[str, ModuleConfig] = {}
        for module_info in self.config['network']['architecture']['modules']:
            modules_configurations[module_info['name']] = \
                ModuleConfig(min_expansions=module_info['network_structure'][0],
                             max_expansions=module_info['network_structure'][1],
                             initial_network_structure=module_info['network_structure_init'],
                             levels_back=module_info['levels_back'])
        return modules_configurations

    def _fix_static_projector_config_if_needed(self) -> Optional[List[int]]:
        try:
            evolvable_module_names: List[str] = [
                x['name'] for x in self.config['network']['architecture']['modules']
            ]
            # in case the settings for an evolvable projector are not defined
            if 'projector' not in evolvable_module_names:
                # look for a static configuration
                # might trigger an exception, in which case it is None
                return self.config['network']['learning']['projector']  # type: ignore
            else:
                if 'projector' in self.config['network']['learning'].keys():
                    logging.warning("You have defined settings for both an evolvable projector and a static projector. "
                                    "Static projector settings (['network']['learning']['projector']) "
                                    "are going to be ignored.")
                return None
        except KeyError:
            return None

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
                             "just passed as parameter.")
        # pylint: disable=protected-access
        if not shutil._samefile(origin_filepath, destination_filepath): # type: ignore
            shutil.copyfile(origin_filepath, destination_filepath)

    def _validate_config(self) -> None:
        schema_path: str = os.path.join("evodenss", "config", "schema.yaml")
        schema: Any = self._load(schema_path)
        validate(self.config, schema)
        logger.info(f"Type of training: {self.config['network']['learning']['learning_type']}")
        if self.config['network']['learning']['learning_type'] == "supervised" :
            if self.config['network']['learning']['augmentation']['train'] is not None:
                self._validate_augmentation_params(self.config['network']['learning']['augmentation']['train'])
            logger.info(f"Augmentation used in training: {self.config['network']['learning']['augmentation']['train']}")
        else:
            self._validate_augmentation_params(self.config['network']['learning']['augmentation']['pretext']['input_a'])
            self._validate_augmentation_params(self.config['network']['learning']['augmentation']['pretext']['input_b'])
            logger.info(f"Augmentation used for input a in training: "
                        f"{self.config['network']['learning']['augmentation']['pretext']['input_a']}")
            logger.info(f"Augmentation used for input b in training: "
                        f"{self.config['network']['learning']['augmentation']['pretext']['input_b']}")

        if self.config['network']['learning']['augmentation']['test'] is not None:
            self._validate_augmentation_params(self.config['network']['learning']['augmentation']['test'])

        logger.info(f"Augmentation used in test: {self.config['network']['learning']['augmentation']['test']}")

    def _validate_augmentation_params(self, params: Dict[str, Any]) -> None:
        for key in params.keys():
            assert key in TransformOperation.enum_values(), \
                f"{key} is not recognised as one of the supported transforms"

    def __getitem__(self, key: str) -> Any:
        return self.config[key]
