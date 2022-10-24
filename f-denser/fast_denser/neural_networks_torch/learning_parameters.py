from dataclasses import dataclass

from torch import optim


@dataclass
class LearningParams:
    #early_stop: int
    batch_size: int
    epochs: int
    torch_optimiser: optim.Optimizer

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LearningParams):
            return self.torch_optimiser.__dict__ == other.torch_optimiser.__dict__ and \
                self.batch_size == other.batch_size and \
                self.epochs == other.epochs
            #return self.torch_optimiser.__dict__ == other.torch_optimiser.__dict__ and \
            #    self.early_stop == other.early_stop and \
            #    self.batch_size == other.batch_size and \
            #    self.epochs == other.epochs
        return False