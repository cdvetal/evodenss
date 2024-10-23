import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet50_Weights, resnet50

from evodenss.config.pydantic import AugmentationConfig, DataSplits, Labelled, SubsetDefinition
from evodenss.dataset.dataset_loader import ConcreteDataset, DatasetProcessor, DatasetType
from evodenss.misc.enums import Device
from evodenss.networks.transformers import BarlowTwinsTransformer, LegacyTransformer

torch.set_printoptions(threshold=1e6) # type: ignore


k = 200
t = 0.1
batch_size = 16
device = Device.GPU



def knn_predict(feature: Tensor,
                feature_bank: Tensor,
                feature_labels: Tensor,
                knn_k: int,
                knn_t: float) -> Tensor:
    """
    Helper method to run kNN predictions on features based on a feature bank

    Args:
        feature: Tensor of shape [N, D] consisting of N D-dimensional features
        feature_bank: Tensor of a database of features used for kNN
        feature_labels: Labels for the features in our feature_bank
        classes: Number of classes (e.g. 10 for CIFAR-10)
        knn_k: Number of k neighbors used for kNN
    """

    #print("feature", feature)
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank).nan_to_num(nan=1.0, posinf=1.0, neginf=-1.0)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1) # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices) # [B, K]

    # we do a reweighting of the similarities
    #sim_weight = (sim_weight / knn_t).exp().nan_to_num(nan=1.0, posinf=1.0, neginf=-1.0)
    #print("sim_weight", sim_weight)
    # counts for each class
    n_classes: int = 10 # for cifar 10
    one_hot_label = torch.zeros(feature.size(0) * knn_k, n_classes, device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0) # [B*K, C]
    # print(one_hot_label)

    # weighted score ---> [B, C]
    pred_scores = \
        torch.sum(one_hot_label.view(feature.size(0), -1, n_classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    #print("pred_scores", pred_scores)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    #print(pred_labels)
    return pred_labels


def main() -> None:
    # New weights with accuracy 80.858%
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    #model.fc = nn.Identity()
    model.to(device.value)
    augmentation_config = AugmentationConfig(
        downstream={},
        test={}
    )
    data_splits = DataSplits(
        labelled=Labelled(
            percentage=100,
            downstream_train=SubsetDefinition(partition_ratio=0.8, amount_to_use=1.0, replacement=True),
            validation=SubsetDefinition(partition_ratio=0.0, amount_to_use=1.0, replacement=True),
            evo_test=SubsetDefinition(partition_ratio=0.2, amount_to_use=1.0, replacement=True)
            )
    )
    dataset_processor = DatasetProcessor(ssl_transformer=BarlowTwinsTransformer(augmentation_config.pretext),
                                         train_transformer=LegacyTransformer(augmentation_config.downstream),
                                         test_transformer=LegacyTransformer(augmentation_config.test))

    dataset: dict[DatasetType, Subset[ConcreteDataset]] = \
        dataset_processor.load_partitioned_dataset('cifar10', data_splits, seed=0)
    loaders_dict: dict[DatasetType, DataLoader[ConcreteDataset]] = \
        DatasetProcessor.get_data_loaders(
                dataset,
                [DatasetType.DOWNSTREAM_TRAIN, DatasetType.EVO_TEST, DatasetType.TEST],
                batch_size
        )


    model.eval()
    correct_guesses: float = 0
    size: int = 0
    feature_bank: list[Tensor] = []
    labels_bank: list[Tensor] = []
    feature: Tensor
    # we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        # generate feature bank
        print("begin train")
        n_batches_train: int = len(loaders_dict[DatasetType.DOWNSTREAM_TRAIN])
        for i, (inputs, labels) in enumerate(loaders_dict[DatasetType.DOWNSTREAM_TRAIN]):
            print(f"{i} / {n_batches_train}")
            feature = model(inputs.to(device.value, non_blocking=True))
            # print(feature)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            labels_bank.append(labels.to(device.value, non_blocking=True))
        print("train done")
        feature_bank_tensor: Tensor = torch.cat(feature_bank, dim=0).t().contiguous() # [D, N]
        # [N]
        labels_bank_tensor: Tensor = torch.cat(labels_bank, dim=0).t().contiguous() # [D, N]
        # loop test data to predict the label by weighted knn search
        for data in loaders_dict[DatasetType.EVO_TEST]:
            inputs, labels = data[0].to(device.value, non_blocking=True), \
                data[1].to(device.value, non_blocking=True)
            
            feature = model(inputs)
            feature = F.normalize(feature, dim=1)
            pred_labels: Tensor = knn_predict(
                feature, feature_bank_tensor, labels_bank_tensor, k, t
            )
            predicted = torch.flatten(torch.transpose(pred_labels[:,:1], 0, 1))
            #print("?=?")
            #print(labels)
            #print("----------------------------")
            #import sys
            #sys.exit(0)
            correct_guesses += (predicted == labels).float().sum().item()
            size += len(labels)
    print("accuracy", correct_guesses/size)

if __name__ == '__main__':
    main()
