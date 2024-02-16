from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image
import numpy as np
import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
import torchvision.models as models
import matplotlib.pyplot as plt


#device for GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define seed for reproducing experiments
RANDOM_SEED = 43
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_database_df(root_path, csv_path, batch_size, image_size=(128,128), is_agumentation=False, test_size=None, as_rgb=False):
    """load images from csv and split into train and testing resulting train and test dataloader

    Args:
        root_path (str): root path is located images
        csv_path (str): path of csv file to get images.
        batch_size (int): number of batch in training and test
        image_size (tuple, optional): _description_. Defaults to (128,128).
        is_agumentation (bool, optional): if is True, we use augmentation in dataset. Defaults to False.
        test_size (float, optional): if is not None, you should set up a float number that indicates partication will be split to train. 0.1 indicates 10% of test set. Defaults to None.
        as_rgb (bool, optional): if is True is a colored image. Defaults to False.

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
        train = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, task="Train", as_rgb=as_rgb)
        test = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, task="Test", as_rgb=as_rgb)
        num_class = len(train.cl_name.values())
            
        print(train.cl_name)
            
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
            
        train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler, num_workers=4)
        test_loader = DataLoader(data, batch_size=batch_size, sampler=test_sampler, num_workers=4)
            
        #print(Counter(train_loader.dataset))

    return train_loader, test_loader, num_class

def load_database_federated(root_path, csv_path, batch_size, num_clients, image_size=(128,128), is_agumentation=False, test_size=None, as_rgb=False):
    """load images from csv and split into train and testing resulting train and test dataloader

    Args:
        root_path (str): root path is located images
        csv_path (str): path of csv file to get images.
        batch_size (int): number of batch in training and test
        num_clients (int): number of clients federated network
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
            train_loader_clients.append(train_loader)
        
        for ds_test in test_split:
            test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
            test_loader_clients.append(test_loader)
        
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
            
            train_loader_clients.append(train_loader)
            test_loader_clients.append(test_loader)
            
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

        elif model_name == "efficientnet":
            model = models.efficientnet.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            
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
        
        else:
            print("Ivalid model name, exiting...")
            exit()
            
        model = model.to(device)

        return model

def show_images(dataset_loader, db_name, path_to_save):
    """function that show images from dataloader

    Args:
        dataset_loader (torch.utils.data.Dataloader): images dataloader
        db_name (str): database name
        path_to_save (str): path to save images

    """
    os.makedirs(path_to_save, exist_ok=True)
    batch = next(iter(dataset_loader))
    images, labels = batch
        
    plt.figure(figsize=(11, 11))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(make_grid(images[:32], padding=2, normalize=True), (1, 2, 0)))
    #plt.savefig("./attack-images/preview_train_{}.png".format(db_name), bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join(path_to_save, "preview_train_{}.png".format(db_name)))

def show_all_images(dataset_loader, db_name, path_to_save):
    """function that show images from dataloader

    Args:
        dataset_loader (torch.utils.data.Dataloader): images dataloader
        db_name (str): database name
        path_to_save (str): path to save images

    """
    os.makedirs(path_to_save, exist_ok=True)
    images, labels = dataloader_to_numpy(dataset_loader)

    for i in range(len(images)):    
        plt.figure(figsize=(4, 4))
        plt.axis("off")
        #plt.title("Training Images")
        plt.imshow(np.transpose(make_grid(images[i], padding=0, normalize=True), (1, 2, 0)))
        #plt.savefig("./attack-images/preview_train_{}.png".format(db_name), bbox_inches='tight', pad_inches=0)
        plt.savefig(os.path.join(path_to_save, f"train_{i}_label{labels[i]}.png".format(db_name)))

def numpy_to_dataloader(images, labels, batch_size):
    """convert numpy dataset to dataloader

    Args:
        images (np.ndarray): numpy array images
        labels (np.ndarray): numpy array labels
        batch_size (int): batch size

    Returns:
        loader (torch.utils.data.Dataloader): torch dataloader with images and labels  
    """    
    dataset  = CustomDataset(images, labels)
    
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    
    return loader

def dataloader_to_numpy(dataloader):
    """convert dataloader dataset to numpy array

    Args:
        dataloader (torch.utils.data.Dataloader): pytorch dataloder with images and labels

    Returns:
        images (np.ndarray): numpy array images
        labels (np.array): numpy array labels
    """    
    images, labels = zip(*[dataloader.dataset[i] for i in range(len(dataloader.dataset))])
    images = torch.stack(images).numpy() 
    labels = np.array(labels)
    
    return images, labels 

class CustomDatasetFromCSV(Dataset):
    """Generating custom dataset for importing images from csv
    """    
    def __init__(self, path_root, tf_image, csv_name, as_rgb=False, task=None):
        self.data = pd.read_csv(csv_name)
        self.as_rgb = as_rgb
        if task is not None:
            self.data.query("Task == @task", inplace=True)
        self.tf_image = tf_image
        self.root = path_root
        self.cl_name = {c: i for i, c in enumerate(np.unique(self.data["y"]))}
        self.BARVALUE = "/" if not os.name == "nt" else "\\"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #x_path = os.path.join(self.root, self.data.iloc[idx, 0].split(self.BARVALUE)[-2], self.data.iloc[idx, 0].split(self.BARVALUE)[-1])
        x_path = os.path.join(self.root, self.data.iloc[idx, 0])
        y = self.cl_name[self.data.iloc[idx, 1]]
        
        X = Image.open(x_path).convert("RGB")
        #X = cv2.cvtColor(cv2.imread(x_path), cv2.COLOR_BGR2RGB) if self.as_rgb else cv2.imread(x_path, cv2.IMREAD_GRAYSCALE)
 
        if self.tf_image:
            X = self.tf_image(X)
        
        return X, y

class CustomDataset(Dataset):
    """Generating custom dataset for converting to dataloader
    """  
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        X = self.images[idx]
        y = self.labels[idx]
        
        # if self.transform:
        #     x = Image.fromarray(self.data[idx].transpose(1,2,0))
        #     x = self.transform(x)
        
        return X, y