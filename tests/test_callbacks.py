import os
import shutil
import unittest
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from evodenss.misc.enums import Device
from evodenss.networks.torch.callbacks import EarlyStoppingCallback, \
    ModelCheckpointCallback, TimedStoppingCallback
from evodenss.networks.torch.trainers import Trainer


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stop_training = False
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 10))

    def forward(self, x):
        return self.fc(x)


class Test(unittest.TestCase):

    def setUp(self):
        batch_size = 100
        dataset = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
        self.data_loader = DataLoader(dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=0,
                                      drop_last=True)


    def test_time_stop(self):
        callback_to_test = TimedStoppingCallback(max_seconds=10)
        trainer = Trainer(None, None, None, None, None, None, None, None, callbacks=[callback_to_test])
        callback_to_test.on_train_begin(trainer)
        time.sleep(5)
        callback_to_test.on_epoch_end(trainer)
        self.assertEqual(trainer.stop_training, False, "Error stop training")
        time.sleep(5)
        callback_to_test.on_epoch_end(trainer)
        self.assertEqual(trainer.stop_training, True, "Error stop training")


    def test_model_saving(self):
        folder_name = "test_dir"
        os.makedirs(folder_name, exist_ok=True)
        callback_to_test = ModelCheckpointCallback(model_saving_dir=folder_name)
        model = Model()
        optimiser = optim.RMSprop(params=model.parameters(), lr=0.1, alpha=0.3, weight_decay=0.001)
        trainer = Trainer(model,
                          optimiser,
                          loss_function=nn.CrossEntropyLoss(),
                          train_data_loader=self.data_loader,
                          validation_data_loader=self.data_loader,
                          n_epochs=3,
                          initial_epoch=0,
                          device=Device.CPU,
                          callbacks=[callback_to_test])
        trainer.train()

        restored_model = torch.load(os.path.join(folder_name, "model.pt"))
        restored_model.load_state_dict(torch.load(os.path.join(folder_name, "weights.pt")))

        expected_keys = ['fc.1.weight', 'fc.1.bias']
        for k in expected_keys:
            # pylint: disable=unsubscriptable-object
            self.assertEqual(model.state_dict()[k].numpy().tolist(),
                             restored_model.state_dict()[k].numpy().tolist())
        self.assertEqual(repr(model._modules), repr(restored_model._modules))
        shutil.rmtree(folder_name)


if __name__ == '__main__':
    unittest.main()
