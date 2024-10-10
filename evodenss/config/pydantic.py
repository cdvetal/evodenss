from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import filecmp
import os
import shutil
from typing import Annotated, Any, Iterator, Optional, cast, no_type_check

from pydantic import BaseModel, ConfigDict, Field, FilePath, PositiveInt, ValidationInfo, field_validator
import yaml # type: ignore

from evodenss.misc.enums import DownstreamMode, FitnessMetricName, LearningType, OptimiserType
from evodenss.misc.utils import ConfigPair


# -------------------------------------------
# ----------   EXTRA DEFINITIONS   ----------
# -------------------------------------------

Probability = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
IntOrFloat = Annotated[int, Field(strict=True, ge=1)] | Annotated[float, Field(strict=True, ge=0.0, le=1.0)]

_init_context_var: ContextVar = ContextVar('_init_context_var', default={})

@contextmanager
def init_context(value: dict[str, Any]) -> Iterator[None]:
    token = _init_context_var.set(value) # type: ignore
    try:
        yield
    finally:
        _init_context_var.reset(token)

# -------------------------------------------
# -------------   BASE MODELS   -------------
# -------------------------------------------

class ConfigBuilder:
    _instance: Optional[Config] = None

    @no_type_check
    def __new__(cls,
                config_path: Optional[str] = None,
                config_dict: Optional[dict[Any, Any]] = None,
                args_to_override: list[ConfigPair] = None) -> Config:
        if not cls._instance:
            if config_path is not None and config_dict is None:
                cls._instance = Config.load_from_file(config_path, args_to_override)
            elif config_path is None and config_dict is not None:
                cls._instance = Config.load_from_dict(config_dict)
            elif config_path is not None and config_dict is not None:
                raise ValueError("Singleton object cannot be created when "
                                 "`config_path` and `config_dict` are both set.")
            else:
                raise ValueError("Singleton object cannot be created when "
                                 "`config_path` and `config_dict` are not set.")
        return cls._instance


class Config(BaseModel):
    checkpoints_path: str
    evolutionary: EvolutionaryConfig
    network: NetworkConfig
    model_config = ConfigDict(extra='forbid')

    @classmethod
    def load_from_file(cls, config_path: str, args_to_override: list[ConfigPair]) -> Config:
        with open(config_path, "r", encoding="utf8") as f:
            config = yaml.safe_load(f)
            for arg in args_to_override:
                cls._override_config(config, arg.key, arg.value)
            validated_config: Config = Config.model_validate(config, context=_init_context_var.get())
            _backup_used_config(validated_config, config_path, validated_config.checkpoints_path)
            return validated_config

    def load_from_dict(cls, config_dict: dict[Any, Any]) -> Config:
        return Config.model_validate(config_dict, context=_init_context_var.get())

    @classmethod
    def _override_config(cls, config: dict[str, Any], path: str, value: Any) -> None:
        keys: list[str] = path.split(".")
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

def _backup_used_config(config: Config, origin_filepath: str, destination: str) -> None:
    os.makedirs(os.path.join(config.checkpoints_path), exist_ok=True)
    destination_filepath: str =  os.path.join(destination, "used_config.yaml")
    # if there is a config file backed up already and it is different than the one we are trying to backup
    if os.path.isfile(destination_filepath) and \
        filecmp.cmp(origin_filepath, destination_filepath) is False:
        raise ValueError("You are probably trying to continue an experiment "
                         "with a different config than the one you used initially. "
                         "This is a gentle reminder to double-check the config you "
                         "just passed as parameter.\n"
                         f"To overlap the experiment do: rm -rf {destination_filepath}")
    if not shutil._samefile(origin_filepath, destination_filepath): # type: ignore
        shutil.copyfile(origin_filepath, destination_filepath)

def get_config() -> Config:
    return cast(Config, ConfigBuilder())


class EvolutionaryConfig(BaseModel):
    generations: PositiveInt
    lambda_: PositiveInt = Field(..., alias='lambda')
    max_epochs: PositiveInt
    mutation: MutationConfig
    fitness: FitnessConfig
    model_config = ConfigDict(extra='forbid')


class FitnessConfig(BaseModel):
    metric_name: FitnessMetricName
    parameters: Optional[DownstreamAccuracyParams | KNNAccuracyParams]

    @field_validator('metric_name', mode='after')
    @classmethod
    def add_context(cls, v: FitnessMetricName, info: ValidationInfo) -> FitnessMetricName:
        if info.context is not None:
            info.context.update({'metric_name': v})
        return v

    # TODO: Confirm that these checks are working
    @field_validator('parameters', mode='after')
    @classmethod
    def validate_parameters(cls, value: Any, info: ValidationInfo) -> Any:
        assert info.context is not None
        metric_name: FitnessMetricName = cast(FitnessMetricName, info.context.get('metric_name'))
        if metric_name == FitnessMetricName.ACCURACY and value is not None:
            raise ValueError(f"Metric name {metric_name.value} requires no params")
        if metric_name == FitnessMetricName.DOWNSTREAM_ACCURACY and \
                isinstance(value, DownstreamAccuracyParams) is False:
            raise ValueError(f"Metric name {metric_name.value} requires params: "
                             f"{[k for k in vars(DownstreamAccuracyParams)['model_fields'].keys()]}")
        if metric_name == FitnessMetricName.KNN_ACCURACY and isinstance(value, KNNAccuracyParams) is False:
            raise ValueError(f"Metric name {metric_name.value} requires params: "
                             f"{[k for k in vars(KNNAccuracyParams)['model_fields'].keys()]}")
        return value


class KNNAccuracyParams(BaseModel):
    k: PositiveInt
    t: float


class DownstreamAccuracyParams(BaseModel):
    downstream_mode: DownstreamMode
    downstream_epochs: PositiveInt
    batch_size: PositiveInt
    optimiser_type: OptimiserType
    optimiser_parameters: dict[str, Any]


class MutationConfig(BaseModel):
    add_connection: Probability
    remove_connection: Probability
    add_layer: Probability
    reuse_layer: Probability
    remove_layer: Probability
    dsge_topological: Probability
    dsge_non_topological: Probability
    train_longer: Probability
    model_config = ConfigDict(extra='forbid')


class NetworkConfig(BaseModel):
    prior_representations: Optional[PriorRepresentationsConfig] = None
    architecture: ArchitectureConfig
    learning: LearningConfig
    model_config = ConfigDict(extra='forbid')

    @field_validator('prior_representations', mode='after')
    @classmethod
    def validate_supervised_only_params(cls, value: Any, info: ValidationInfo) -> Any:
        assert info.context is not None
        learning_type: LearningType = cast(LearningType, info.context.get('learning_type'))
        if learning_type == LearningType.self_supervised and value is not None:
            raise ValueError(f"{info.field_name} cannot be set when "
                             f"network.learning.learning_type is `{learning_type.value}`")
        return value



class PriorRepresentationsConfig(BaseModel):
    representations_model_path: FilePath
    representations_weights_path: FilePath
    training_mode: DownstreamMode
    model_config = ConfigDict(extra='forbid')


class ArchitectureConfig(BaseModel):
    reuse_layer: Probability
    extra_components: list[str]
    output: str
    projector: Optional[list[PositiveInt]] = Field(None, min_length=1)
    modules: list[ModuleConfig]
    model_config = ConfigDict(extra='forbid')

    def __init__(self, /, **data: Any) -> None:
        self.__pydantic_validator__.validate_python(
            data,
            self_instance=self,
            context=_init_context_var.get(),
        )

    @field_validator('modules', mode='after')
    @classmethod
    def add_context(cls, v: list[ModuleConfig], info: ValidationInfo) -> list[ModuleConfig]:
        if info.context is not None:
            info.context.update({'module_names': [i.name for i in v]})
        return v

    @field_validator('projector', mode='after')
    @classmethod
    def validate_projector(cls, value: Any, info: ValidationInfo) -> Any:
        assert info.context is not None
        assert info.field_name == "projector"

        learning_type: LearningType = cast(LearningType, info.context.get('learning_type'))
        if learning_type == LearningType.self_supervised and value is None:
            module_names: list[str] = cast(list[str], info.context.get('module_names'))
            if value is None and 'projector' not in module_names:
                raise ValueError("You must select a way to build the projector network, "
                                 "either by evolving it (inside network.architecture.modules) "
                                 "or using a static one (network.architecture.projector)")
            elif value is not None and 'projector' in module_names:
                raise ValueError("projector network is being defined both through ",
                                 "evolvable parameters (inside network.architecture.modules) "
                                 "and static parameters (network.architecture.projector) "
                                 "You must select only one way to build the projector network")
        elif learning_type == LearningType.supervised and value is not None:
            raise ValueError("projector network can only be set when "
                             f"network.learning.learning_type is `{learning_type.value}`")
        return value


class LearningConfig(BaseModel):
    learning_type: LearningType
    default_train_time: PositiveInt
    data_splits: DataSplits
    augmentation: AugmentationConfig
    model_config = ConfigDict(extra='forbid')

    @field_validator('learning_type', mode='after')
    @classmethod
    def add_context(cls, v: LearningType, info: ValidationInfo) -> LearningType:
        print("context ", info.context)
        if info.context is not None:
            info.context.update({'learning_type': v})
        return v


class ModuleConfig(BaseModel):
    name: str
    network_structure_init: NetworkStructure
    network_structure: NetworkStructure
    levels_back: PositiveInt
    model_config = ConfigDict(extra='forbid')


class NetworkStructure(BaseModel):
    min_expansions: PositiveInt
    max_expansions: PositiveInt
    model_config = ConfigDict(extra='forbid')


class DataSplits(BaseModel):
    labelled: Labelled
    unlabelled: Optional[Unlabelled] = None
    model_config = ConfigDict(extra='forbid')

    def __init__(self, /, **data: Any) -> None:
        self.__pydantic_validator__.validate_python(
            data,
            self_instance=self,
            context=_init_context_var.get(),
        )

    @field_validator('unlabelled', mode='after')
    @classmethod
    def validate_ssl_only_params(cls, value: Any, info: ValidationInfo) -> Any:
        assert info.context is not None
        learning_type: LearningType = cast(LearningType, info.context.get('learning_type'))
        if learning_type == LearningType.supervised and value is not None:
            raise ValueError(f"{info.field_name} augmentation cannot be set when "
                             f"network.learning.learning_type is `{learning_type.value}`")
        elif learning_type == LearningType.self_supervised and value is None:
            raise ValueError(f"{info.field_name} augmentation must be set when "
                             f"network.learning.learning_type is `{learning_type.value}`")
        return value


class Labelled(BaseModel):
    percentage: Annotated[int, Field(strict=True, ge=1, le=100)]
    downstream_train: SubsetDefinition
    validation: SubsetDefinition
    evo_test: SubsetDefinition
    model_config = ConfigDict(extra='forbid')


class Unlabelled(BaseModel):
    pretext_train: SubsetDefinitionNoPartition
    model_config = ConfigDict(extra='forbid')

class SubsetDefinitionNoPartition(BaseModel):
    amount_to_use: IntOrFloat
    replacement: bool
    model_config = ConfigDict(extra='forbid')

class SubsetDefinition(BaseModel):
    partition_ratio: float
    amount_to_use: IntOrFloat
    replacement: bool
    model_config = ConfigDict(extra='forbid')


class PretextAugmentation(BaseModel):
    input_a: dict[str, Any]
    input_b: dict[str, Any]
    model_config = ConfigDict(extra='forbid')


class AugmentationConfig(BaseModel):
    pretext: PretextAugmentation = PretextAugmentation(input_a={}, input_b={})
    downstream: dict[str, Any] = {}
    test: dict[str, Any] = {}
    model_config = ConfigDict(extra='forbid')

    @field_validator('pretext', mode='after')
    @classmethod
    def validate_ssl_only_params(cls, value: Any, info: ValidationInfo) -> Any:
        assert info.context is not None
        learning_type: LearningType = cast(LearningType, info.context.get('learning_type'))
        if learning_type == LearningType.supervised:
            if value is not None:
                raise ValueError(f"{info.field_name} augmentation cannot be set when "
                                 f"network.learning.learning_type is `{learning_type.value}`")
            elif info.field_name == 'pretext': # ensure that the dicts exist even if they are empty
                return PretextAugmentation(input_a={}, input_b={})
        elif learning_type == LearningType.self_supervised and value is None:
            raise ValueError(f"{info.field_name} augmentation must be set when "
                             f"network.learning.learning_type is `{learning_type.value}`")
        return value
    
    @field_validator('pretext', 'downstream', 'test', mode='before')
    @classmethod
    def fill_null_augmentations(cls, value: Any, info: ValidationInfo) -> Any:
        if info.field_name == "pretext":
            return value if value is not None else PretextAugmentation(input_a={}, input_b={})
        else:
            return value if value is not None else {}


def get_fitness_extra_params() -> dict[str, Any]:
    fitness_params: DownstreamAccuracyParams | KNNAccuracyParams | None = get_config().evolutionary.fitness.parameters
    if fitness_params is None:
        return {}
    else:
        return fitness_params.model_dump()
