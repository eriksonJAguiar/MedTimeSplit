from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image
import numpy as np
import pandas as pd
import torch
import os
import cv2
import torchvision
import timm
from sklearn.model_selection import train_test_split
import torchvision.models as models
import matplotlib.pyplot as plt


#device for GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define seed for reproducing experiments
RANDOM_SEED = 43
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_database_df(root_path, csv_path, batch_size, image_size=(128,128), is_agumentation=False, test_size=None, as_rgb=False, is_stream=False, domain_type=None):
    """load images from csv and split into train and testing resulting train and test dataloader

    Args:
        root_path (str): root path is located images
        csv_path (str): path of csv file to get images.
        batch_size (int): number of batch in training and test
        image_size (tuple, optional): _description_. Defaults to (128,128).
        is_agumentation (bool, optional): if is True, we use augmentation in dataset. Defaults to False.
        test_size (float, optional): if is not None, you should set up a float number that indicates partication will be split to train. 0.1 indicates 10% of test set. Defaults to None.
        as_rgb (bool, optional): if is True is a colored image. Defaults to False.
        is_stream (bool, optional): define if dataset is prepared for streaming training or not. Defaults to False.
        domain_type (str, optional): if is stream,
    Returns:
        train_loader (torch.utils.data.Dataloader): images dataloader for training
        test_loader (torch.utils.data.Dataloader): images dataloader for testing
        num_class (int): number of classes in the dataset
    """
    if is_agumentation:
        tf_image = transforms.Compose([#transforms.ToPILImage(),
                                    transforms.Resize(image_size),
                                    #transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomRotation(degrees=(20, 150)),
                                    transforms.RandomCrop(size=(100,100)),
                                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
                                    #transforms.RandomAffine(degrees=3, shear=0.01),
                                    #transforms.RandomResizedCrop(size=image_size, scale=(0.875, 1.0)),
                                    transforms.ColorJitter(brightness=(0.7, 1.5)),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
    else:
        tf_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    if test_size is None:
        train = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, task="Train", as_rgb=as_rgb, domain_type=domain_type)
        test = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, task="Test", as_rgb=as_rgb, domain_type=domain_type)
        num_class = len(train.cl_name.values())
            
        print(train.cl_name)
        
        if is_stream:
            return train, test, num_class
            
        train_loader = DataLoader(train, batch_size=batch_size, num_workers=95, shuffle=True)
        test_loader = DataLoader(test, batch_size=batch_size, num_workers=95, shuffle=False)
    else:
        data = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, as_rgb=as_rgb)
            
        print({k: cl for k, cl in enumerate(data.cl_name)})
            
        train, test = train_test_split(list(range(len(data))), test_size=test_size, shuffle=True, random_state=RANDOM_SEED)
            
        # index_num = int(np.floor(0.1*len(test)))
        # test_index = test[:len(test)-index_num]
            
        train_sampler = SubsetRandomSampler(train)
        test_sampler = SubsetRandomSampler(test)
            
        num_class = len(data.cl_name.values())
        
        if is_stream:
            return train_sampler, test_sampler, num_class
            
        train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler, num_workers=95)
        test_loader = DataLoader(data, batch_size=batch_size, sampler=test_sampler, num_workers=95)
            
        #print(Counter(train_loader.dataset))

    return train_loader, test_loader, num_class

def load_database_hold_out(root_path, csv_path, batch_size, image_size=(128,128), is_agumentation=False, test_size=None, as_rgb=False, is_stream=False, domain_type=None):
    """load images from csv and split into train and testing resulting train and test dataloader

    Args:
        root_path (str): root path is located images
        csv_path (str): path of csv file to get images.
        batch_size (int): number of batch in training and test
        image_size (tuple, optional): _description_. Defaults to (128,128).
        is_agumentation (bool, optional): if is True, we use augmentation in dataset. Defaults to False.
        test_size (float, optional): if is not None, you should set up a float number that indicates partication will be split to train. 0.1 indicates 10% of test set. Defaults to None.
        as_rgb (bool, optional): if is True is a colored image. Defaults to False.
        is_stream (bool, optional): define if dataset is prepared for streaming training or not. Defaults to False.
        domain_type (str, optional): if is stream,
    Returns:
        train_loader (torch.utils.data.Dataloader): images dataloader for training
        test_loader (torch.utils.data.Dataloader): images dataloader for testing
        num_class (int): number of classes in the dataset
    """
    if is_agumentation:
        tf_image = transforms.Compose([#transforms.ToPILImage(),
                                    transforms.Resize(image_size),
                                    #transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    #transforms.RandomAffine(degrees=3, shear=0.01),
                                    #transforms.RandomResizedCrop(size=image_size, scale=(0.875, 1.0)),
                                    #transforms.ColorJitter(brightness=(0.7, 1.5)),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
    else:
        tf_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    if test_size is None:
        train = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, task="Train", as_rgb=as_rgb, domain_type=domain_type)
        test = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, task="Test", as_rgb=as_rgb, domain_type=domain_type)
        num_class = len(train.cl_name.values())
            
        print(train.cl_name)
        
        if is_stream:
            return train, test, num_class
            
        train_loader = DataLoader(train, batch_size=batch_size, num_workers=4, shuffle=True)
        test_loader = DataLoader(test, batch_size=batch_size, num_workers=4, shuffle=False)
    else:
        data = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, as_rgb=as_rgb)
            
        print({k: cl for k, cl in enumerate(data.cl_name)})
            
        train, test = train_test_split(list(range(len(data))), test_size=test_size, shuffle=True, random_state=RANDOM_SEED)
            
        # index_num = int(np.floor(0.1*len(test)))
        # test_index = test[:len(test)-index_num]
            
        train_sampler = SubsetRandomSampler(train)
        test_sampler = SubsetRandomSampler(test)
            
        num_class = len(data.cl_name.values())
        
        if is_stream:
            return train_sampler, test_sampler, num_class
            
        train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler, num_workers=4)
        test_loader = DataLoader(data, batch_size=batch_size, sampler=test_sampler, num_workers=4)
            
        #print(Counter(train_loader.dataset))

    return train_loader, test_loader, num_class

def split_database_continous(root_path, csv_path, batch_size, image_size=(128,128), is_agumentation=False, test_size=None, as_rgb=False, domain_type=None, task=None):
    """load images from csv and split into train and testing resulting train and test dataloader

    Args:
        root_path (str): root path is located images
        csv_path (str): path of csv file to get images.
        batch_size (int): number of batch in training and test
        image_size (tuple, optional): _description_. Defaults to (128,128).
        is_agumentation (bool, optional): if is True, we use augmentation in dataset. Defaults to False.
        test_size (float, optional): if is not None, you should set up a float number that indicates partication will be split to train. 0.1 indicates 10% of test set. Defaults to None.
        as_rgb (bool, optional): if is True is a colored image. Defaults to False.
        domain_type (str, optional): if is stream,
    Returns:
        train_loader (torch.utils.data.Dataloader): images dataloader for training
        test_loader (torch.utils.data.Dataloader): images dataloader for testing
        num_class (int): number of classes in the dataset
    """
    if is_agumentation:
        tf_image = transforms.Compose([#transforms.ToPILImage(),
                                    transforms.Resize(image_size),
                                    #transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    #transforms.RandomAffine(degrees=3, shear=0.01),
                                    #transforms.RandomResizedCrop(size=image_size, scale=(0.875, 1.0)),
                                    #transforms.ColorJitter(brightness=(0.7, 1.5)),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
    else:
        tf_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    train_samples, test_samples = None
    if test_size is None:
        train_samples = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, task="Train", as_rgb=as_rgb, domain_type=domain_type)
        test_samples = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, task="Test", as_rgb=as_rgb, domain_type=domain_type)
        num_class = len(train.cl_name.values())
            
        print(train.cl_name)
        
        #if is_stream:
        #    return train, test, num_class
            
        #train_loader = DataLoader(train, batch_size=batch_size, num_workers=4, shuffle=True)
        #test_loader = DataLoader(test, batch_size=batch_size, num_workers=4, shuffle=False)
    else:
        data = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, as_rgb=as_rgb)
            
        print({k: cl for k, cl in enumerate(data.cl_name)})
            
        train, test = train_test_split(list(range(len(data))), test_size=test_size, shuffle=True, random_state=RANDOM_SEED)
            
        # index_num = int(np.floor(0.1*len(test)))
        # test_index = test[:len(test)-index_num]
            
        test_samples = SubsetRandomSampler(train)
        test_samples = SubsetRandomSampler(test)
            
        num_class = len(data.cl_name.values())
        
        #if is_stream:
        #    return train_sampler, test_sampler, num_class
            
        #train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler, num_workers=4)
        #test_loader = DataLoader(data, batch_size=batch_size, sampler=test_sampler, num_workers=4)
            
        #print(Counter(train_loader.dataset))

    return test_samples, test_samples, num_class

def load_database_federated(root_path, csv_path, batch_size, num_clients, image_size=(128,128), is_agumentation=False, test_size=None, as_rgb=False, is_stream=False):
    """load images from csv and split into train and test resulting partitions of clients on Federated learning

    Args:
        root_path (str): root path is located images
        csv_path (str): path of csv file to get images.
        batch_size (int): number of batch in training and test
        num_clients (int): number of clients on federated network
        image_size (tuple, optional): _description_. Defaults to (128,128).
        is_agumentation (bool, optional): if is True, we use augmentation in dataset. Defaults to False.
        test_size (float, optional): if is not None, you should set up a float number that indicates partication will be split to train. 0.1 indicates 10% of test set. Defaults to None.
        as_rgb (bool, optional): if is True is a colored image. Defaults to False.

    Returns:
        parameters (dict): contains keys: train, test, and num_class that represents train and test loader for each client
    """
    if is_agumentation:
        tf_image = transforms.Compose([#transforms.ToPILImage(),
                                    transforms.Resize(image_size),
                                    #transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    #transforms.RandomAffine(degrees=3, shear=0.01),
                                    #transforms.RandomResizedCrop(size=image_size, scale=(0.875, 1.0)),
                                    #transforms.ColorJitter(brightness=(0.7, 1.5)),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
    else:
        tf_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    train_loader_clients = []
    test_loader_clients  = []
    
    if test_size is None:
        train = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, task="Train", as_rgb=as_rgb)
        test = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, task="Test", as_rgb=as_rgb)
        num_class = len(train.cl_name.values())
        print(train.cl_name)
        
        partition_size = len(train) // num_clients
        diff = len(train) - (partition_size*num_clients)
        lengths = [partition_size] * num_clients
        lengths[-1] = lengths[-1]+diff
        train_split = torch.utils.data.random_split(train, lengths, torch.Generator().manual_seed(RANDOM_SEED))
        
        partition_size = len(test) // num_clients
        diff = len(test) - (partition_size*num_clients)
        lengths = [partition_size] * num_clients
        lengths[-1] = lengths[-1]+diff
        test_split = torch.utils.data.random_split(test, lengths, torch.Generator().manual_seed(RANDOM_SEED))
        
        for ds_train in train_split:
            train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
            train_loader_clients.append(ds_train if is_stream else train_loader)
        
        for ds_test in test_split:
            test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
            test_loader_clients.append(ds_test if is_stream else test_loader)
        
    else:
        data = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, as_rgb=as_rgb)
        num_class = len(train.cl_name.values())
        print(data.cl_name)
        
        partition_size = len(data) // num_clients
        lengths = [partition_size] * num_clients
        datasets = torch.utils.data.random_split(data, lengths, torch.Generator().manual_seed(RANDOM_SEED))
        
        for ds in datasets:
            len_test = len(ds) // (100*test_size)
            len_train = len(ds) - len_test
            
            train, test = torch.utils.data.random_split(ds, [len_train, len_test], torch.Generator().manual_seed(RANDOM_SEED))
            
            train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
            
            train_loader_clients.append(train if is_stream else train_loader)
            test_loader_clients.append(test if is_stream else test_loader)
            
    parameters = {
            "train": train_loader_clients,
            "test": test_loader_clients,
            "num_class": num_class,
    }
            
    return parameters

def make_model_pretrained(model_name, num_class):
        """function to select models pre-trained on image net and using tochvision architectures

        Args:
            model_name (str): string to describe the name of the models, such as Resnet50, Resnet18, inceptionv3, densenet201, vgg16, vgg18, and efficientnet
            num_class (int): number of class extracted from dataset

        Returns:
            model (torch.nn.Module): a pre-trained on ImageNet architecture with torchvision
        """
        model = None
        out_features_model = num_class if num_class > 2 else 1

        if model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            
            #self._freeze_layers(model, 5)
            for param in model.parameters():
                param.requires_grad = False
            
            model.fc = torch.nn.Sequential(
                torch.nn.Linear(model.fc.in_features, 224),
                torch.nn.BatchNorm1d(224),
                torch.nn.ReLU(),
                #torch.nn.Dropout(0.5),
                #torch.nn.Linear(512, 128),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(224, out_features_model)
            )
           
        elif model_name == "resnet101":
            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            
            #self._freeze_layers(model, 5)
            for param in model.parameters():
                param.requires_grad = False
            
            model.fc = torch.nn.Sequential(
                torch.nn.Linear(model.fc.in_features, 224),
                torch.nn.BatchNorm1d(224),
                torch.nn.ReLU(),
                #torch.nn.Dropout(0.5),
                #torch.nn.Linear(512, 128),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(224, out_features_model)
            )
        
        elif model_name == "resnet152":
            model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
            
            #self._freeze_layers(model, 5)
            for param in model.parameters():
                param.requires_grad = False
            
            model.fc = torch.nn.Sequential(
                torch.nn.Linear(model.fc.in_features, 224),
                torch.nn.BatchNorm1d(224),
                torch.nn.ReLU(),
                #torch.nn.Dropout(0.5),
                #torch.nn.Linear(512, 128),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(224, out_features_model)
            )
        
        elif model_name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
            model.fc = torch.nn.Sequential(
               torch.nn.Linear(model.fc.in_features, out_features_model),
               torch.nn.Softmax()
            )
            #model = ResNet18(3, out_features_model)
            #model = ResNet50(1, num_classes=num_class)
                
        elif model_name == "densenet":

            model = models.densenet.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            
            for param in model.parameters():
                param.requires_grad = False
            
            #self._freeze_layers(model, 10)
            
            num_ftrs = model.classifier.in_features
            model.classifier = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, out_features_model),
                #torch.nn.Linear(num_ftrs, out_features_model),
                #torch.nn.Softmax()
            )
            #model.classifier = nn.Linear(num_ftrs, out_features=out_features_model) 
        
        elif model_name == "inceptionv3":
            model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
            model.aux_logits = False
            #model = models.inception_v3(pretrained=True, aux_logits=False)
            
            #self._freeze_layers(model, 10)
            
            for param in model.parameters():
                param.requires_grad = False

            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, out_features_model),
            )

        elif model_name == "vgg16":
            model = models.vgg.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            
            for param in model.parameters():
                param.requires_grad = False
            
            #self._freeze_layers(model, 10)
            
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, out_features_model),
            )
            #model.classifier[6].register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
            #model.classifier[6] = torch.nn.Linear(num_ftrs, out_features=out_features_model)
             
        elif model_name == "vgg19":
            model = models.vgg.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            
            for param in model.parameters():
                param.requires_grad = False
            
            #self._freeze_layers(model, 10)
            
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Sequential(
                #torch.nn.Linear(num_ftrs, out_features_model),
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, out_features_model),
                # torch.nn.Dropout(0.5),
                # torch.nn.Linear(224, out_features_model),
                #torch.nn.Softmax(),
            )

        elif model_name == "efficientnetb7":
            model = models.efficientnet.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
            
            for param in model.parameters():
                param.requires_grad = False
            
            #self._freeze_layers(model, 10)
            
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, out_features_model),
                #torch.nn.Softmax(),
            )
            #model.classifier[1] = nn.Linear(num_ftrs, out_features=out_features_model)
        
        elif model_name == "nasnetlarge":
            model = timm.create_model('nasnetalarge', pretrained=True)
            
            for param in model.parameters():
                param.requires_grad = False
            
            num_ftrs = model.last_linear.in_features
            model.last_linear = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, out_features_model),
                #torch.nn.Softmax(),
            )
        
        elif model_name == "alexnet":
            model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
            
            for param in model.parameters():
                param.requires_grad = False
            
            #self._freeze_layers(model, 10)
            
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, out_features_model),
                #torch.nn.Softmax(),
            )
        
        elif model_name == "inceptionresnet":
            model = timm.create_model('inception_resnet_v2', pretrained=True)
            
            for param in model.parameters():
                param.requires_grad = False
            
            # #self._freeze_layers(model, 10)
            
            num_ftrs = model.classif.in_features
            model.classif = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, out_features_model),
                #torch.nn.Softmax(),
            )
        
        else:
            print("Ivalid model name, exiting...")
            exit()
            
        model = model.to(device)

        return model

def show_images(dataset_loader, db_name, path_to_save, batch_index=0):
    """function that show images from dataloader

    Args:
        dataset_loader (torch.utils.data.Dataloader): images dataloader
        db_name (str): database name
        path_to_save (str): path to save images

    """
    os.makedirs(path_to_save, exist_ok=True)
    
    for i, batch in enumerate(dataset_loader):
        if i == batch_index:
            images, labels = batch
            break
    else:
        raise IndexError(f"Batch index {batch_index} out of range.")
        
    plt.figure(figsize=(11, 11))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(make_grid(images[:32], padding=2, normalize=True), (1, 2, 0)))
    #plt.savefig("./attack-images/preview_train_{}.png".format(db_name), bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join(path_to_save, f"preview_train_{db_name}_batch_{batch_index}.png"))

def show_all_images(dataset_loader, db_name, path_to_save):
    """function that show images from dataloader

    Args:
        dataset_loader (torch.utils.data.Dataloader): images dataloader
        db_name (str): database name
        path_to_save (str): path to save images

    """
    os.makedirs(path_to_save, exist_ok=True)
    #images, labels = dataloader_to_numpy(dataset_loader)

    i = 0
    for x, y in dataset_loader:
        for idx in range(x.shape[0]):
            plt.figure(figsize=(6, 6))
            plt.axis("off")
            #plt.title("Training Images")
            plt.imshow(np.transpose(make_grid(x[idx], padding=0, normalize=True), (1, 2, 0)))
            #plt.savefig("./attack-images/preview_train_{}.png".format(db_name), bbox_inches='tight', pad_inches=0)
            plt.savefig(os.path.join(path_to_save, f"train_{i}_label{y[idx]}.png".format(db_name)), bbox_inches='tight', pad_inches=0, dpi=400)
            i += 1 

def show_one_image(image, label, path_to_save, image_name):
    """function that show images from dataloader

    Args:
        dataset_loader (torch.utils.data.Dataloader): images dataloader
        db_name (str): database name
        path_to_save (str): path to save images

    """
    os.makedirs(path_to_save, exist_ok=True)
    #images, labels = dataloader_to_numpy(dataset_loader)
    #print(image.shape)
    #image = image.squeeze(2)
    #image_np = image.cpu().numpy().transpose((1, 2, 0))
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    #plt.title("Training Images")
    plt.imshow(np.transpose(make_grid(image, padding=0, normalize=True), (1, 2, 0)))
    #plt.savefig("./attack-images/preview_train_{}.png".format(db_name), bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join(path_to_save, f"train_{image_name}_label{np.argmax(label)}.png"), bbox_inches='tight', pad_inches=0, dpi=400)

def numpy_to_dataloader(images, labels, batch_size, is_transform=False):
    """convert numpy dataset to dataloader

    Args:
        images (np.ndarray): numpy array images
        labels (np.ndarray): numpy array labels
        batch_size (int): batch size
        is_transform (bool, optional): if true, we sould apply the transformation on image. Defaults is False.

    Returns:
        loader (torch.utils.data.Dataloader): torch dataloader with images and labels  
    """
    image_size = (images.shape[-1], images.shape[-1])
    if is_transform:
        tf_image = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        tf_image = None
        
    dataset  = CustomDataset(images, labels, tf_image=tf_image)
    
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=95)
    
    return loader

def dataloader_to_numpy(dataloader):
    """convert dataloader dataset to numpy array

    Args:
        dataloader (torch.utils.data.Dataloader): pytorch dataloder with images and labels

    Returns:
        images (np.ndarray): numpy array images
        labels (np.array): numpy array labels
    """    
    #images, labels = zip(*[torch.from_numpy(dataloader.dataset[i]) for i in range(len(dataloader.dataset))])
    #images = torch.stack(images).numpy()
    #labels = np.array(labels)
    images, labels = zip(*[(x.numpy(), y.numpy()) for x, y in dataloader])
    
    images = np.concatenate(images)
    labels = np.concatenate(labels)
    
    return images, labels 

def read_model_from_checkpoint(model_path, model_name, nb_class):
    """load a trained model using checkpoint

    Args:
        model_path (str): model weights path location
        model_name (str): model name
        nb_class (int): number of the classes in the dataset

    Returns:
        model (toch.nn.Module): pre-trained model selected by model name and weights
    """
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = {key[6:] : checkpoint['state_dict'][key] for key in checkpoint['state_dict']}
    model = get_model_structure(model_name, nb_class)
    model.load_state_dict(state_dict)
    
    return model

def get_model_structure(model_name, nb_class):
    """get model achitecture using torchvision

    Args:
        model_name (str): model selected name. Selected one of them "resnet50" "vgg16" "vgg19" "inceptionv3" "densenet" "efficientnet".
        nb_class (int): number of classes

    Returns:
        model (toch.nn.Module): pre-trained model selected by model name and weights
    """
    model = None
    #"resnet50" "vgg16" "vgg19" "inceptionv3" "densenet" "efficientnet"
    nb_class = nb_class if nb_class > 2 else 1
    if model_name == "resnet50":
        model = torchvision.models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 224),
                torch.nn.BatchNorm1d(224),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(224, nb_class)
        )
    
    elif model_name == "resnet101":
        model = torchvision.models.resnet101()
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 224),
                torch.nn.BatchNorm1d(224),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(224, nb_class)
        )

    elif model_name == "vgg16":
        model = torchvision.models.vgg.vgg16()
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, nb_class),
        )
    
    elif model_name == "vgg19":
        model = torchvision.models.vgg.vgg19()
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, nb_class),
        )
    
    elif model_name == "inceptionv3":
        model = torchvision.models.inception_v3()
        model.aux_logits = False
        
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, nb_class),
        )       
    
    elif model_name == "efficientnet":
        model = torchvision.models.efficientnet_b0()
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, nb_class),
        )
    
    elif model_name == "densenet":
        model = torchvision.models.densenet121()
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, nb_class),
        )
    
    return model

def normalize_imageNet(image):
    """normalize image transformed with imageNet settings

    Args:
        image (np.ndarray): image will be converted
        is_cuda (bool, optional): if True use operations on GPU_. Defaults to True.

    Returns:
        img_norm (np.ndarray): normalized image
    """
    if len(image.shape) == 4:
            return np.array([ 
                normalize_imageNet(single_img) 
                for single_img in image
            ])
       
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    img_norm = image * STD[:, None, None] + MEAN[:, None, None]
        
    return img_norm

def generate_domain(image, domain_type="normal"):
    """generate dataset domain shift

    Args:
        image (np.ndarray): input image to be convert to another domain
        domain_type (str, optional): domain can be a fixed string, such as normal, illumination or occlusion. Defaults to "normal".

    Returns:
        noise_image (PIL.Image): image modified with gaussian noise, illumination, or occlusion
    """
    image = np.asarray(image)
    noise_image = image.copy()
    
    if domain_type == "normal":
        normal_noise = np.random.normal(0, 25, image.shape)
        noise_image = image + normal_noise
        noise_image = np.clip(noise_image, 0, 255).astype(np.uint8)
    elif domain_type == "illumination":
        brightness = 10 
        contrast = 1.4  
        noise_image = cv2.addWeighted(noise_image, contrast, np.zeros(image.shape, image.dtype), 0, brightness) 
    elif domain_type == "occlusion":
        h, w, _ = image.shape
        noise_image = cv2.rectangle(noise_image, (w, h), (w - (w//8), h - (h//8)), (255, 255, 255), cv2.FILLED)
    
    noise_image = Image.fromarray(noise_image)
    
    return noise_image
    
class CustomDatasetFromCSV(Dataset):
    """Generating custom dataset for importing images from csv
    """    
    def __init__(self, path_root, tf_image, csv_name, as_rgb=False, task=None, domain_type=None):
        table_data = pd.read_csv(csv_name)
        if task is not None:
            table_data.query("Task == @task", inplace=True)
        
        self.data = table_data["x"].tolist()
        self.targets = table_data["y"].tolist()
        self.domain_type = domain_type
        
        self.as_rgb = as_rgb
        self.tf_image = tf_image
        self.root = path_root
        self.cl_name = {c: i for i, c in enumerate(np.unique(self.targets))}
        self.targets = [self.cl_name[k] for k in self.targets]
        self.BARVALUE = "/" if not os.name == "nt" else "\\"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #x_path = os.path.join(self.root, self.data.iloc[idx, 0].split(self.BARVALUE)[-2], self.data.iloc[idx, 0].split(self.BARVALUE)[-1])
        x_path = os.path.join(self.root, self.data[idx])
        #print(f"target: {self.targets[idx]}")
        #y = self.cl_name[self.targets[idx]]
        y = self.targets[idx]
        
        X = Image.open(x_path).convert("RGB")
        if self.domain_type is not None:
            X  = generate_domain(X, domain_type=self.domain_type)
        #X = cv2.cvtColor(cv2.imread(x_path), cv2.COLOR_BGR2RGB) if self.as_rgb else cv2.imread(x_path, cv2.IMREAD_GRAYSCALE)
 
        if self.tf_image:
            X = self.tf_image(X)
        
        return X, y

class CustomDataset(Dataset):
    """Generating custom dataset for converting to dataloader
    """  
    def __init__(self, images, labels, tf_image=None):
        self.images = images
        self.labels = labels
        self.transform = tf_image
        self.mode = "RGB" if images[0].shape[0] == 3 else "L"
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        X = self.images[idx]
        y = int(self.labels[idx])
        
        if self.transform:
            #X = Image.fromarray((X.transpose(1,2,0) * 255).astype(np.uint8))
            X = X.transpose(1,2,0).squeeze(axis=2) if self.mode == "L" else X.transpose(1,2,0)
            X = Image.fromarray(X, mode=self.mode)
            X = self.transform(X)
        
        return X, y

class CustomDatasetContinous(Dataset):
    """Generating custom dataset for importing images from csv
    """    
    def __init__(self, path_root, tf_image, csv_name, as_rgb=False, task=None, domain_type=None):
        table_data = pd.read_csv(csv_name)
        if task is not None:
            table_data.query("Task == @task", inplace=True)
        
        self.data = table_data["x"].tolist()
        self.targets = table_data["y"].tolist()
        self.domain_type = domain_type
        
        self.as_rgb = as_rgb
        self.tf_image = tf_image
        self.root = path_root
        self.cl_name = {c: i for i, c in enumerate(np.unique(self.targets))}
        self.targets = [self.cl_name[k] for k in self.targets]
        self.BARVALUE = "/" if not os.name == "nt" else "\\"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #x_path = os.path.join(self.root, self.data.iloc[idx, 0].split(self.BARVALUE)[-2], self.data.iloc[idx, 0].split(self.BARVALUE)[-1])
        x_path = os.path.join(self.root, self.data[idx])
        #print(f"target: {self.targets[idx]}")
        #y = self.cl_name[self.targets[idx]]
        y = self.targets[idx]
        
        X = Image.open(x_path).convert("RGB")
        if self.domain_type is not None:
            X  = generate_domain(X, domain_type=self.domain_type)
        #X = cv2.cvtColor(cv2.imread(x_path), cv2.COLOR_BGR2RGB) if self.as_rgb else cv2.imread(x_path, cv2.IMREAD_GRAYSCALE)
 
        if self.tf_image:
            X = self.tf_image(X)
        
        return X, y