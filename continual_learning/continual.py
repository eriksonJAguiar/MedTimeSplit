from avalanche.benchmarks import SplitMNIST, PermutedMNIST
from avalanche.models import IncrementalClassifier
from avalanche.models import MultiHeadClassifier
from avalanche.models import SimpleCNN
from avalanche.models import as_multitask
from avalanche.benchmarks.datasets import FashionMNIST, MNIST
from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
import torch
import torchvision
import os
from utils import utils

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive
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
# from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop
# from avalanche.evaluation.metrics import (
#     forgetting_metrics,
#     accuracy_metrics,
#     loss_metrics,
# )
# from avalanche.logging import InteractiveLogger
# from avalanche.training.plugins import EvaluationPlugin

# --- CONFIG
device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

# # transformations are managed by the AvalancheDataset
# train_transforms = torchvision.transforms.ToTensor()
# eval_transforms = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Resize((32, 32))
# ])

# train_MNIST = as_classification_dataset(MNIST('mnist', train=True, download=True))
# test_MNIST = as_classification_dataset(MNIST('mnist', train=False, download=True))

# # choose some metrics and evaluation method
# interactive_logger = InteractiveLogger()

# eval_plugin = EvaluationPlugin(
#         accuracy_metrics(
#             minibatch=True, epoch=True, experience=True, stream=True
#         ),
#         loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
#         forgetting_metrics(experience=True),
#         loggers=[interactive_logger],
# )

# model = SimpleMLP(num_classes=10)
# optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
# criterion = CrossEntropyLoss()
# cl_strategy = Naive(
#     model,
#     torch.optim.Adam(model.parameters(), lr=0.001),
#     CrossEntropyLoss(),
#     train_mb_size=1,
#     eval_mb_size=32,
#     device=device,
#     evaluator=eval_plugin,
# )

# scenario
#benchmark = SplitMNIST(n_experiences=5, seed=1)

#benchmark = 


# from avalanche.benchmarks.classic import SplitMNIST

# bm = SplitMNIST(
#     n_experiences=5,  # 5 incremental experiences
#     return_task_id=True,  # add task labels
#     seed=1  # you can set the seed for reproducibility. This will fix the order of classes
# )

# # streams have a name, used for logging purposes
# # each metric will be logged with the stream name
# print(f'--- Stream: {bm.train_stream.name}')
# # each stream is an iterator of experiences
# for exp in bm.train_stream:
#     # experiences have an ID that denotes its position in the stream
#     # this is used only for logging (don't rely on it for training!)
#     eid = exp.current_experience
#     # for classification benchmarks, experiences have a list of classes in this experience
#     clss = exp.classes_in_this_experience
#     # you may also have task labels
#     tls = exp.task_labels
#     print(f"EID={eid}, classes={clss}, tasks={tls}")
#     # the experience provides a dataset
#     print(f"data: {len(exp.dataset)} samples")

# for exp in bm.test_stream:
#     print(f"EID={exp.current_experience}, classes={exp.classes_in_this_experience}, task={tls}")




def run_continual(root_path, csv_path):
    
    train, test, num_class = utils.load_database_df(
        root_path=root_path,
        csv_path=csv_path,
        batch_size=32, 
        image_size=(128,128),
        is_agumentation=False,
        as_rgb=True,
        is_stream=True
    )

    benchmark = ni_benchmark(
        train_dataset=train, 
        test_dataset=test,
        n_experiences=5, 
        task_labels=False,
        balance_experiences=True,
    )
    
    model = utils.make_model_pretrained("resnet50", num_class=7)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    cl_strategy = Naive(
        model, optimizer, criterion,
        train_mb_size=100, train_epochs=10, eval_mb_size=100
    )
    
    results = []
    print('Starting experiment...')
    print("Training...")
    for experience in benchmark.train_stream:
        print("Start of experience ", experience.current_experience)
        #experiences have an ID that denotes its position in the stream
        # this is used only for logging (don't rely on it for training!)
        eid = experience.current_experience
        # for classification benchmarks, experiences have a list of classes in this experience
        clss = experience.classes_in_this_experience
        print(f"EID={eid}, classes={clss}")
        # the experience provides a dataset
        print(f"data: {len(experience.dataset)} samples")
        
        cl_strategy.train(experience)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.eval(benchmark.test_stream))
    
    print(results)
    
    print("Testing...")
    for exp in benchmark.test_stream:
        print(f"EID={exp.current_experience}, classes={exp.classes_in_this_experience}")
        
    
    # for experience in benchmark.train_stream:
    #     print("Start of experience: ", experience.current_experience)
    #     print("Current Classes: ", experience.classes_in_this_experience)

    #     cl_strategy.train(experience)
    #     print('Training completed')

    #     print('Computing accuracy on the whole test set')
    #     results.append(cl_strategy.eval(benchmark.test_stream))