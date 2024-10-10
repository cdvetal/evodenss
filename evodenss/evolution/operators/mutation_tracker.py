from dataclasses import astuple, dataclass
import logging
from typing import Any, Callable, Iterator, Optional, ParamSpec, cast
from typing_extensions import TypeVar

from evodenss.misc.enums import MutationType
from evodenss.evolution.individual import Individual


logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')

@dataclass
class MutationLog:
    mutation_type: MutationType
    gen: int
    data: dict[str, Any]

    def __str__(self) -> str:
        return f"Mutation type [{self.mutation_type}] on gen [{self.gen}] with data: {self.data}"

    def __iter__(self) -> Iterator[Any]:
        return iter(astuple(self))


def enable_tracking(mutation_type: MutationType) -> Callable[[Callable[P, T]], Callable[P, None]]: # pyright: ignore

    def track_mutation(mutation_type: MutationType, data_to_track: dict[str, Any]) -> None:
        individual: Individual = data_to_track.pop("individual")
        assert isinstance(individual, Individual)
        
        generation: int = data_to_track.pop("generation")
        assert isinstance(generation, int)

        if individual.individual_genotype.mutation_tracker is not None:
            individual.individual_genotype.mutation_tracker.append(
                MutationLog(mutation_type, generation, data_to_track)
            )

        if mutation_type == MutationType.REMOVE_LAYER:
            logger.info(f"Individual {individual.id} is going to have a layer removed from "
                        f"Module {data_to_track['module_idx']}: {data_to_track['module_name']}; "
                        f"Position: {data_to_track['remove_idx']}")
        elif mutation_type == MutationType.ADD_LAYER:
            logger.info(f"Individual {individual.id} is going to have an extra layer at "
                        f"Module {data_to_track['module_idx']}: {data_to_track['module_name']}; "
                        f"Position: {data_to_track['insert_pos']}. Reused?: {data_to_track['is_reused']}")
        elif mutation_type == MutationType.TRAIN_LONGER:
            if "from" in data_to_track.keys() and "to" in data_to_track.keys():
                logger.info(f"Individual {individual.id} total training time extended "
                            f"to {individual.total_allocated_train_time}")
            else:
                logger.info(f"Individual {individual.id} total training time was not extended: "
                            "maximum had already been reached")
        elif mutation_type == MutationType.DSGE_TOPOLOGICAL:
            logger.info(f"Individual {individual.id} is going to have a DSGE mutation on "
                        f"Module {data_to_track['module_idx']} "
                        f"Position: {data_to_track['layer_idx']}")
        elif mutation_type == MutationType.DSGE_NON_TOPOLOGICAL:
            logger.info(f"Individual {individual.id} is going to have a DSGE mutation on "
                        f"Non-topological component: {data_to_track['symbol']}")
        elif mutation_type == MutationType.ADD_CONNECTION:
            logger.info(f"Individual {individual.id} is going to have a new connection "
                        f"Module {data_to_track['module_idx']}: {data_to_track['module_name']}; "
                        f"Layer: {data_to_track['layer_idx']}; "
                        f"New connection: {data_to_track['new_input']}")
        elif mutation_type == MutationType.REMOVE_CONNECTION:
            logger.info(f"Individual {individual.id} is going to have a connection removed from "
                        f"Module {data_to_track['module_idx']}: {data_to_track['module_name']}; "
                        f"Layer: {data_to_track['layer_idx']}; "
                        f"Connection removed: {data_to_track['removed_input']}")
    
    def wrap(mutation_function: Callable[P, T]) -> Callable[P, None]:
        def wrapped_f(*args: P.args, **kwargs: P.kwargs) -> None:
            data = cast(Optional[dict[str, Any]], mutation_function(*args, **kwargs))
            if data is not None:
                track_mutation(mutation_type, data)
        return wrapped_f
    return wrap
