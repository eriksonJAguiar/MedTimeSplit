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
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
from avalanche.benchmarks.utils import AvalancheDataset, concat_datasets
from avalanche.benchmarks import nc_benchmark, ni_benchmark


#device for GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define seed for reproducing experiments
RANDOM_SEED = 37
torch.manual_seed(RANDOM_SEED)
#np.random.seed(RANDOM_SEED)

def __sample_dirichlet(data, alpha, random_state=None):
    np.random.seed(random_state)
    proportions = np.random.dirichlet(alpha, size=1)[0]
    sampled_data = data.sample(frac=1.0, replace=False, random_state=random_state).reset_index(drop=True)
    split_indices = (proportions * len(data)).astype(int)
    split_data = np.split(sampled_data, np.cumsum(split_indices)[:-1])
    
    return split_data

def load_database_df(root_path, csv_path, batch_size, image_size=(128,128), is_agumentation=False, test_size=None, as_rgb=False, is_stream=False, tf_transformer=None):
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
    if tf_transformer is None:
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
    else:
        tf_image = tf_transformer
        
    if test_size is None:
        train = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, task="Train", as_rgb=as_rgb)
        test = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, task="Test", as_rgb=as_rgb)
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

def load_database_hold_out(root_path, csv_path, batch_size, image_size=(128,128), is_agumentation=False, test_size=None, as_rgb=False, is_stream=False, domain_type=None):
    """load images from csv and split into train and testing resulting train and test dataloader

    Args:
        root_path (str): root path is located images
        csv_path (str): path of csv file to get images.
        batch_size (int): number of batch in training and test.
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

def load_database_federated_continous(root_path, csv_path, hyperparams_client, image_size=(128,128), test_size=None, as_rgb=False, K=5):
    """load images from csv and partition into train and testing based on temporal strategy resulting train and test dataloader

    Args:
        root_path (str): root path is located images
        csv_path (str): path of csv file to get images.
        batch_size (int): number of batch in training and test
        hyperparams_client(dict): patients features related to domains
        image_size (tuple, optional): _description_. Defaults to (128,128).
        test_size (float, optional): if is not None, you should set up a float number that indicates partication will be split to train. 0.1 indicates 10% of test set. Defaults to None.
        as_rgb (bool, optional): if is True is a colored image. Defaults to False.
        K (int, optional): The number of clients Defaults to 5. 
    Returns:
        train_loader (torch.utils.data.Dataloader): images dataloader for training
        test_loader (torch.utils.data.Dataloader): images dataloader for testing
        num_class (int): number of classes in the dataset
    """
    
    #1st variable to save data partitions 
    # train_clients = []
    # test_clients  = []
    # train_split = None
    # test_split = None
    train_clients = []
    test_clients  = []
    
    #2nd read csv file with values 
    csv_data = pd.read_csv(csv_path)

    if test_size is None:
        train_csv = csv_data[csv_data["Task"] == "Train"]
        test_csv = csv_data[csv_data["Task"] == "Test"]
    
        #train_csv = train_csv.sample(frac=1.0, replace=False, random_state=RANDOM_SEED).reset_index(drop=True)
        #test_csv = test_csv.sample(frac=1.0, replace=False, random_state=RANDOM_SEED).reset_index(drop=True)
        
    else:
        #shuffle_df = csv_data.sample(frac=1.0, replace=False, random_state=RANDOM_SEED).reset_index(drop=True)
        shuffle_df = csv_data
        train_size = int(len(csv_data) * (1-test_size))
        train_csv = shuffle_df[:train_size]
        test_csv = shuffle_df[train_size:]
    
    #using dirilechlet distribution to selecte random examples
    alpha = [0.5] * K
    train_split = __sample_dirichlet(train_csv, alpha, random_state=RANDOM_SEED)
    test_split = __sample_dirichlet(test_csv, alpha, random_state=RANDOM_SEED)
    
    for cl in range(K):
            #print(f"Domain k: {cl}")
            domain_j = {
                "clean": transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.ToTensor(),
                        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
                "noise": transforms.Compose([
                        transforms.Resize(image_size),
                        CustomTransformNoise(mean=hyperparams_client[str(cl)]["mean"], sigma=hyperparams_client[str(cl)]["sigma"]),
                        transforms.ToTensor(),
                        #transforms.v2.GaussianNoise(mean=hyperparams_client[str(cl)]["mean"], sigma=hyperparams_client[str(cl)]["sigma"]),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
                "illumination": transforms.Compose([
                        transforms.Resize(image_size),
                        CustomTransformIllumination(brightness=hyperparams_client[str(cl)]["brightness"], contrast=hyperparams_client[str(cl)]["contrast"]),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
                "occlusion": transforms.Compose([
                        transforms.Resize(image_size),
                        CustomTransformOcclusion(occlusion_size=hyperparams_client[str(cl)]["occlusion_size"]),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            }
            
            aval_train_clients = []
            aval_test_clients = []
            
            for k, tf in domain_j.items():
                train_cl = CustomDatasetContinous(path_root=root_path, csv_data=train_split[cl], tf_image=tf, as_rgb=as_rgb)
                test_cl = CustomDatasetContinous(path_root=root_path, csv_data=test_split[cl], tf_image=tf, as_rgb=as_rgb)
                num_class = len(train_cl.cl_name.values())

                #print(f"Classes in train_cl: {set(train_cl.cl_name.values())}")
                
                aval_train_clients.append(train_cl)
                aval_test_clients.append(test_cl)
            
            avl_ds_train = ConcatDataset(aval_train_clients)
            avl_ds_test = ConcatDataset(aval_test_clients)
            train_clients.append(avl_ds_train)
            test_clients.append(avl_ds_test)
            
    # parameters = {
    #         "train": train_clients,
    #         "test": test_clients,
    #         "num_class": num_class,
    # }
            
    return train_clients, test_clients, num_class

def load_database_federated_continousSplit(root_path, csv_path, image_size=(128,128), test_size=None, as_rgb=False, K=5):
    """load images from csv and partition into train and testing based SplitMnist strategy

    Args:
        root_path (str): root path is located images
        csv_path (str): path of csv file to get images.
        batch_size (int): number of batch in training and test
        image_size (tuple, optional): _description_. Defaults to (128,128).
        as_rgb (bool, optional): if is True is a colored image. Defaults to False.
        K (int, optional). Defaults to 5. The number of d
    Returns:
        train_loader (torch.utils.data.Dataloader): images dataloader for training
        test_loader (torch.utils.data.Dataloader): images dataloader for testing
        num_class (int): number of classes in the dataset
    """
    
    #1st variable to save data partitions 
    # train_clients = []
    # test_clients  = []
    # train_split = None
    # test_split = None
    train_clients = []
    test_clients  = []
    
    #2nd read csv file with values 
    csv_data = pd.read_csv(csv_path)
    
    np.random.seed(RANDOM_SEED)
    rotation_direction = np.random.randint(30, 90, size=K, dtype=int)
    contrast = np.random.uniform(0.5, 0.9, size=K)
    salt_prob = np.random.uniform(0.01, 0.1, size=K)
    pepper_prob = np.random.uniform(0.01, 0.1, size=K)

    if test_size is None:
        train_csv = csv_data[csv_data["Task"] == "Train"]
        test_csv = csv_data[csv_data["Task"] == "Test"]
    
        train_csv = train_csv.sample(frac=1.0, replace=False, random_state=RANDOM_SEED).reset_index(drop=True)
        test_csv = test_csv.sample(frac=1.0, replace=False, random_state=RANDOM_SEED).reset_index(drop=True)
        
        train_split = np.array_split(train_csv, K)
        test_split = np.array_split(test_csv, K)
        
    else:
        shuffle_df = csv_data.sample(frac=1.0, replace=False, random_state=RANDOM_SEED).reset_index(drop=True)
        train_size = int(len(csv_data) * (1-test_size))
        train_csv = shuffle_df[:train_size]
        test_csv = shuffle_df[train_size:]
        
        train_split = np.array_split(train_csv, K)
        test_split = np.array_split(test_csv, K)
    
    for cl in range(K):
            #print(f"Domain k: {cl}")
            domain_j = {
                "clean": transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.ToTensor(),
                        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
                "noise": transforms.Compose([
                        transforms.Resize(image_size),
                        CustomTransformSaltPepperNoise(salt_prob=salt_prob[cl], pepper_prob=pepper_prob[cl]),
                        transforms.ToTensor(),
                        #transforms.v2.GaussianNoise(mean=hyperparams_client[str(cl)]["mean"], sigma=hyperparams_client[str(cl)]["sigma"]),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
                "constrast": transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.ColorJitter(contrast=contrast[cl]),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
                "rotation": transforms.Compose([
                        transforms.Resize(image_size),
                        #CustomTransformOcclusion(occlusion_size=hyperparams_client[str(cl)]["occlusion_size"]),
                        transforms.RandomRotation(rotation_direction[cl]),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            }
            
            # domain_j =  transforms.Compose([
            #             transforms.Resize(image_size),
            #             #CustomTransformOcclusion(occlusion_size=hyperparams_client[str(cl)]["occlusion_size"]),
            #             CustomTransformSaltPepperNoise(salt_prob=salt_prob, pepper_prob=pepper_prob), #"noise"
            #             transforms.ColorJitter(contrast=contrast[cl]), #"constrast
            #             transforms.RandomRotation(rotation_direction[cl]), #"rotation"
            #             transforms.ToTensor(),
            #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # ])
            
            #train_split = train_csv.sample(frac=1.0, replace=False).reset_index(drop=True)
            #test_split = test_csv.sample(frac=1.0, replace=False).reset_index(drop=True)
        
            #print(f"Classes in train_cl: {set(train_cl.cl_name.values())}")
            
            num_domains = len(domain_j.keys())
            
            aval_train_clients = []
            aval_test_clients = []
            
            train_split_domain = np.array_split(train_split[cl], num_domains)
            test_split_domain = np.array_split(test_split[cl], num_domains)
            
            for i, (k, tf) in enumerate(domain_j.items()):
                train_cl = CustomDatasetContinous(path_root=root_path, csv_data=train_split_domain[i], tf_image=tf, as_rgb=as_rgb)
                test_cl = CustomDatasetContinous(path_root=root_path, csv_data=test_split_domain[i], tf_image=tf, as_rgb=as_rgb)
                num_class = len(train_cl.cl_name.values())
                
                aval_train_clients.append(train_cl)
                aval_test_clients.append(test_cl)

            avl_ds_train = ConcatDataset(aval_train_clients)
            avl_ds_test = ConcatDataset(aval_test_clients)
            
            train_clients.append(avl_ds_train)
            test_clients.append(avl_ds_test)
            
    # parameters = {
    #         "train": train_clients,
    #         "test": test_clients,
    #         "num_class": num_class,
    # }
            
    return train_clients, test_clients, num_class

def load_database_federated_continousPermuted(root_path, csv_path, image_size=(128,128), test_size=None, as_rgb=False, K=5):
    """load images from csv and partition into train and testing based PermutedMnist strategy

    Args:
        root_path (str): root path is located images
        csv_path (str): path of csv file to get images.
        batch_size (int): number of batch in training and test
        hyperparams_client(dict): patients features related to domains
        image_size (tuple, optional): _description_. Defaults to (128,128).
        test_size (float, optional): if is not None, you should set up a float number that indicates partication will be split to train. 0.1 indicates 10% of test set. Defaults to None.
        as_rgb (bool, optional): if is True is a colored image. Defaults to False.
        K (int, optional). Defaults to 5. The number of d
    Returns:
        train_loader (torch.utils.data.Dataloader): images dataloader for training
        test_loader (torch.utils.data.Dataloader): images dataloader for testing
        num_class (int): number of classes in the dataset
    """
    #1st variable to save data partitions 
    # train_clients = []
    # test_clients  = []
    # train_split = None
    # test_split = None
    train_clients = []
    test_clients  = []
    
    #2nd read csv file with values 
    csv_data = pd.read_csv(csv_path)
    seeds = np.random.randint(40, 123, K)
    
    #np.random.seed(RANDOM_SEED)

    if test_size is None:
        train_csv = csv_data[csv_data["Task"] == "Train"]
        test_csv = csv_data[csv_data["Task"] == "Test"]
    
        train_csv = train_csv.sample(frac=1.0, replace=False, random_state=RANDOM_SEED).reset_index(drop=True)
        test_csv = test_csv.sample(frac=1.0, replace=False, random_state=RANDOM_SEED).reset_index(drop=True)
        
        train_split = np.array_split(train_csv, K)
        test_split = np.array_split(test_csv, K)
        
    else:
        shuffle_df = csv_data.sample(frac=1.0, replace=False, random_state=RANDOM_SEED).reset_index(drop=True)
        train_size = int(len(csv_data) * (1-test_size))
        train_csv = shuffle_df[:train_size]
        test_csv = shuffle_df[train_size:]
        
        train_split = np.array_split(train_csv, K)
        test_split = np.array_split(test_csv, K)
    
    for cl in range(K):
            #print(f"Domain k: {cl}")
        domain_j = transforms.Compose([
            transforms.Resize(image_size),
            CustomTransformPermuted(image_size=image_size, seed=seeds[cl]),   
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
            
            #train_split = train_csv.sample(frac=1.0, replace=False).reset_index(drop=True)
            #test_split = test_csv.sample(frac=1.0, replace=False).reset_index(drop=True)
            
        train_cl = CustomDatasetContinous(path_root=root_path, csv_data=train_split[cl], tf_image=domain_j, as_rgb=as_rgb)
        test_cl = CustomDatasetContinous(path_root=root_path, csv_data=test_split[cl], tf_image=domain_j, as_rgb=as_rgb)
        num_class = len(train_cl.cl_name.values())
                
        train_clients.append(train_cl)
        test_clients.append(test_cl)
            
    return train_clients, test_clients, num_class

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
        
    tf_image_test = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    train_loader_clients = []
    test_loader_clients  = []
    
    if test_size is None:
        train = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, task="Train", as_rgb=as_rgb)
        test = CustomDatasetFromCSV(root_path, tf_image=tf_image_test, csv_name=csv_path, task="Test", as_rgb=as_rgb)
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

def load_database_federated_non_iid(root_path, csv_path, batch_size, num_clients, hyperparams_client, image_size=(128,128), test_size=None, as_rgb=False, is_stream=False):
    """load images from csv and split into train and test resulting partitions of clients on Federated learning

    Args:
        root_path (str): root path is located images
        csv_path (str): path of csv file to get images.
        batch_size (int): number of batch in training and test
        num_clients (int): number of clients on federated network
        hyperparams_domians (dict): paramters for each clients to generate non-iid data
        image_size (tuple, optional): _description_. Defaults to (128,128).
        is_agumentation (bool, optional): if is True, we use augmentation in dataset. Defaults to False.
        test_size (float, optional): if is not None, you should set up a float number that indicates partication will be split to train. 0.1 indicates 10% of test set. Defaults to None.
        as_rgb (bool, optional): if is True is a colored image. Defaults to False.
        is_stream (bool, optional): define if dataset is prepared for streaming training or not. Defaults to False.

    Returns:
        parameters (dict): contains keys: train, test, and num_class that represents train and test loader for each client
    """ 
    train_clients = []
    test_clients  = []
    train_split = None
    test_split = None
    
    csv_data = pd.read_csv(csv_path)

    if test_size is None:
        train_csv = csv_data[csv_data["Task"] == "Train"]
        test_csv = csv_data[csv_data["Task"] == "Test"]
    
        train_csv = train_csv.sample(frac=1.0, replace=False, random_state=RANDOM_SEED).reset_index(drop=True)
        test_csv = test_csv.sample(frac=1.0, replace=False, random_state=RANDOM_SEED).reset_index(drop=True)

        train_split = np.array_split(train_csv, num_clients)
        test_split = np.array_split(test_csv, num_clients)
        #random sampling of clients and D^k domains
        #train_split = [train_csv.sample(frac=1.0, replace=False).reset_index(drop=True) for _ in range(num_clients)]
        #test_split =  [test_csv.sample(frac=1.0, replace=False).reset_index(drop=True) for _ in range(num_clients)]
        
    else:
        shuffle_df = csv_data.sample(frac=1.0, replace=False, random_state=RANDOM_SEED).reset_index(drop=True)
        train_size = int(len(csv_data) * (1-test_size))
        train_csv = shuffle_df[:train_size]
        test_csv = shuffle_df[train_size:]
        
        train_split = np.array_split(train_csv, num_clients)
        test_split = np.array_split(test_csv, num_clients)
        #random sampling of clients and D^k domains
        #train_split = [train_csv.sample(frac=1.0, replace=False).reset_index(drop=True) for _ in range(num_clients)]
        #test_split =  [test_csv.sample(frac=1.0, replace=False).reset_index(drop=True) for _ in range(num_clients)]
    
    for cl, (train, test) in enumerate(zip(train_split, test_split)):
            tf_image = {
                "clean": transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.ToTensor(),
                        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
                "noise": transforms.Compose([
                        transforms.Resize(image_size),
                        CustomTransformNoise(mean=hyperparams_client[str(cl)]["mean"], sigma=hyperparams_client[str(cl)]["sigma"]),
                        transforms.ToTensor(),
                        #transforms.v2.GaussianNoise(mean=hyperparams_client[str(cl)]["mean"], sigma=hyperparams_client[str(cl)]["sigma"]),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
                "illumination": transforms.Compose([
                        transforms.Resize(image_size),
                        CustomTransformIllumination(brightness=hyperparams_client[str(cl)]["brightness"], contrast=hyperparams_client[str(cl)]["contrast"]),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
                "occlusion": transforms.Compose([
                        transforms.Resize(image_size),
                        CustomTransformOcclusion(occlusion_size=hyperparams_client[str(cl)]["occlusion_size"]),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            }
            
            train_cl = CustomDatasetFromCSVNonIID(path_root=root_path, csv_data=train, tf_image=tf_image, batch=batch_size, as_rgb=as_rgb)
            test_cl = CustomDatasetFromCSVNonIID(path_root=root_path, csv_data=test, tf_image=tf_image, batch=batch_size, as_rgb=as_rgb)
            num_class = len(train_cl.cl_name.values())
            
            if not is_stream:
                 train_clients.append(DataLoader(train_cl, batch_size=batch_size, shuffle=True, num_workers=95))
                 test_clients.append(DataLoader(test_cl, batch_size=batch_size, shuffle=False, num_workers=95))
            else:
                train_clients.append(train_cl)
                test_clients.append(test_cl)
            
    parameters = {
            "train": train_clients,
            "test": test_clients,
            "num_class": num_class,
    }
            
    return parameters

def generate_domain_no_iid(image, domain_type="clean"):
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
    def __init__(self, path_root, tf_image, csv_name, as_rgb=False, task=None):
        table_data = pd.read_csv(csv_name)
        if task is not None:
            table_data.query("Task == @task", inplace=True)
        
        self.data = table_data["x"].tolist()
        self.targets = table_data["y"].tolist()
        
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
    def __init__(self, path_root, tf_image, csv_data, as_rgb=False):
        
        self.data = csv_data["x"].tolist()
        self.targets = csv_data["y"].tolist()
        
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
        if self.tf_image:
            X = self.tf_image(X)
        
        return X, y

    def get_targets(self):
        return self.labels

class CustomDatasetFromCSVNonIID(Dataset):
    """Generating custom dataset for importing images from csv
    """    
    def __init__(self, path_root, tf_image, csv_data, batch, as_rgb=False):
        
        self.data = csv_data["x"].tolist()
        self.targets = csv_data["y"].tolist()
        self.batch_counter = 0
        self.samples_per_domain = batch//4
        self.current_domain = 0
        self.domains = ["clean", "noise", "illumination", "occlusion"]
        
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
        
        #define the number of domains calculus to split into partitions 
        self.batch_counter = 0 if self.batch_counter == self.samples_per_domain else self.batch_counter
        self.current_domain = 0 if self.current_domain == 4 else self.current_domain
        
        tf_image_domain = self.tf_image[self.domains[self.current_domain]]
        
        #x_path = os.path.join(self.root, self.data.iloc[idx, 0].split(self.BARVALUE)[-2], self.data.iloc[idx, 0].split(self.BARVALUE)[-1])
        x_path = os.path.join(self.root, self.data[idx])
        #print(f"target: {self.targets[idx]}")
        #y = self.cl_name[self.targets[idx]]
        y = self.targets[idx]
        
        X = Image.open(x_path).convert("RGB")
        X = tf_image_domain(X)
        self.batch_counter += 1
        self.current_domain += 1
        
        return X, y
    
class CustomTransformIllumination:
    def __init__(self, brightness, contrast):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, x):
        # Apply rotation to the image
        image = np.asarray(x)
        noise_image = image.copy()
    
        # brightness = 10 
        # contrast = 1.4  
        noise_image = cv2.addWeighted(noise_image, self.contrast, np.zeros(image.shape, image.dtype), 0, self.brightness) 
        noise_image = np.clip(noise_image, 0, 255).astype(np.uint8)
        
        return noise_image
    
    def __repr__(self):
        return f"{self.__class__.__name__}(brightness={self.brightness}, contrast={self.contrast})"
    
class CustomTransformOcclusion:
    def __init__(self, occlusion_size):
        self.occlusion_size = occlusion_size

    def __call__(self, x):
        # Apply rotation to the image
        image = np.asarray(x)
        noise_image = image.copy()
    
        h, w, _ = image.shape
        noise_image = cv2.rectangle(noise_image, (w, h), (w - (w//self.occlusion_size), h - (h//self.occlusion_size)), (0, 0, 0), cv2.FILLED)
        noise_image = np.clip(noise_image, 0, 255).astype(np.uint8)
        
        return noise_image
    
    def __repr__(self):
        return f"{self.__class__.__name__}(occlusion_size={self.occlusion_size})"

class CustomTransformNoise:
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, x):
        # Apply rotation to the image
        image = np.asarray(x)
        noise_image = image.copy()
    
        normal_noise = np.random.normal(self.mean, self.sigma, image.shape)
        noise_image = image + normal_noise
        noise_image = np.clip(noise_image, 0, 255).astype(np.uint8)
        
        return noise_image
    
    def __repr__(self):
        return f"{self.__class__.__name__}(occlusion_size={self.occlusion_size})"
    
class CustomTransformSaltPepperNoise():
    def __init__(self, salt_prob=0.01, pepper_prob=0.01):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob

    def __call__(self, img):
        img_np = np.array(img)
        h,w = img_np.shape[0], img_np.shape[1]

        salt_mask = np.random.rand(h,w) < self.salt_prob
        img_np[salt_mask] = 255

        pepper_mask = np.random.rand(h,w) < self.pepper_prob
        img_np[pepper_mask] = 0

        return Image.fromarray(img_np.astype(np.uint8))

class CustomTransformPermuted:
    def __init__(self, image_size, seed=None):
        # Generate a random permutation for shuffling
        np.random.seed(seed)
        self.image_size = image_size
        self.permutation = np.random.permutation(image_size[0]*image_size[1])

    def __call__(self, image):
        np_image = np.array(image)
        if len(np_image.shape) == 3 and np_image.shape[2] == 3:
            img_permuted = np.zeros_like(np_image)
            for c in range(3):
                # Flatten the image to a 1D array
                img_flat = np_image[:, :, c].flatten()
                
                # Apply the permutation
                img_permuted_flat = img_flat[self.permutation]
                
                # Reshape back to the original shape
                img_permuted[:, :, c] = img_permuted_flat.reshape(self.image_size)
        else:
            # Flatten the image to a 1D array
            img_flat = np_image.flatten()
            
            # Apply the permutation
            img_permuted_flat = img_flat[self.permutation]
            
            # Reshape back to the original shape
            img_permuted = img_permuted_flat.reshape(self.image_size)
        
        return Image.fromarray(img_permuted)