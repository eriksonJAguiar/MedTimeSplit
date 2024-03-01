from avalanche.benchmarks import SplitMNIST, PermutedMNIST
from avalanche.models import IncrementalClassifier
from avalanche.models import MultiHeadClassifier
from avalanche.models import SimpleCNN
from avalanche.models import as_multitask
from avalanche.benchmarks.datasets import FashionMNIST, MNIST
from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.benchmarks.utils import make_classification_dataset, as_classification_dataset
import torch
import torchvision
import os
from utils import utils

# dataset = SplitMNIST(10, shuffle=False, class_ids_from_zero_in_each_exp=False)
# dataset2 = SplitMNIST(5, shuffle=False, return_task_id=True, class_ids_from_zero_in_each_exp=True)
#dataset = PermutedMNIST(n_experiences=3)

# model = IncrementalClassifier(in_features=784)
# model2 = MultiHeadClassifier(in_features=784)
# model3 = SimpleCNN()
# print(model3)

# print(model2)
# for exp in dataset2.train_stream:
#     model2.adaptation(exp)
#     print(model2)

# train_stream = dataset.train_stream
# test_stream = dataset.test_stream

# for exp in train_stream:
#     print("Start of task: ", exp.task_label)
#     print("Classes in this task: ", exp.classes_in_this_experience)
    
#     current_train_set = exp.dataset
#     print(f"Task {exp.task_label}")
#     print("This task contains", len(current_train_set), "Training examples")
    
#     current_test_set = test_stream[exp.current_experience].dataset
#     print("This task contains", len(current_test_set), "Test examples")


from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop
train_transform = Compose([
    RandomCrop(28, padding=4),
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

# test_transform = Compose([
#     ToTensor(),
#     Normalize((0.1307,), (0.3081,))
# ])

mnist_train = MNIST(
    '../data/mnist', train=True, download=True, transform=train_transform
)
print(type(mnist_train.train))
# mnist_test = MNIST(
#     '../data/mnist', train=False, download=True, transform=test_transform
# )

def run_continual(root_path, csv_path):
    

    values = utils.load_database_federated(
        root_path=root_path,
        csv_path=csv_path,
        num_clients=10,
        batch_size=32, 
        image_size=(12,128),
        is_agumentation=False,
        as_rgb=True,
        #is_stream=True
    )
    
    train = values["train"]
    test = values["test"]
    num_class = values["num_class"]
    
    #print(train[0])
    # scenario = ni_benchmark(
    #     train_dataset=train, 
    #     test_dataset=test,
    #     n_experiences=10,
    #     shuffle=True,
    #     seed=43,
    #     balance_experiences=True
    # )
    
    scenario = dataset_benchmark(
        train_datasets=[train[0]],   
        test_datasets=[test[0]],
    )
    
    print('Without custom task labels:', scenario.train_stream[1].task_label)

    # for exp in scenario.train_stream:
    #     t = exp.task_label
    #     exp_id = exp.current_experience
    #     train_dataset = exp.dataset
        
    #     print('Task {} batch {} -> train'.format(t, exp_id))
    #     print('This batch contains', len(train_dataset), 'patterns')
    