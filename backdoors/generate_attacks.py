import sys
#sys.path.append("../utils")

import torch
import numpy as np
from utils import utils
import os
import time
from PIL import Image
import matplotlib.pyplot as plt

from art.estimators.classification import PyTorchClassifier
from art.attacks.poisoning import PoisoningAttackBackdoor, PoisoningAttackCleanLabelBackdoor, BadDetGlobalMisclassificationAttack, HiddenTriggerBackdoorPyTorch, SleeperAgentAttack
from art.attacks.poisoning.perturbations import add_pattern_bd, add_single_bd, insert_image
from art.utils import to_categorical


class PoisonWithBadNets():

    def __init__(self, target_size, target_path=None, batch_size=32, poison_percent=0.2):
        self.target_path = target_path
        self.poison_percent = poison_percent
        self.batch_size = batch_size
        self.target_size = target_size
        self.max_val = 0
    
    def posison_dataset(self, x_clean, y_clean, poison_func):
        """function to generate poisoning dataset

        Args:
            x_clean (_type_): _description_
            y_clean (_type_): _description_
            poison_func (_type_): _description_

        Returns:
            DataLoader: dataset with poisoned examples
        """
        x_poison = np.copy(x_clean)
        y_poison = np.copy(y_clean)
        is_poison = np.zeros(np.shape(y_poison)[0])
        nb_samples = np.shape(y_clean)[0]
        num_poison = int(self.poison_percent * nb_samples)

        print(f"Number of samples: {nb_samples}")
        print(f"Number of poisoning samples: {num_poison}")

        indices_to_be_poisoned = np.random.choice(nb_samples, num_poison, replace=False)
        imgs_to_be_poisoned = x_clean[indices_to_be_poisoned]
        labels_to_be_poisoned = y_clean[indices_to_be_poisoned]

        backdoor_attack = PoisoningAttackBackdoor(poison_func)
        poison_images, poison_labels = backdoor_attack.poison(
            imgs_to_be_poisoned, y=labels_to_be_poisoned
        )
        
        print(f"Labels: {poison_labels}")

        x_poison[indices_to_be_poisoned] = poison_images
        y_poison[indices_to_be_poisoned] = poison_labels
        is_poison[indices_to_be_poisoned] = 1

        is_poison = is_poison != 0

        return is_poison, x_poison, y_poison
    
    def poison_func_pattern(self, x):
        print(x.shape)
        images = np.expand_dims(add_pattern_bd(x, pixel_value=self.max_val, channels_first=True), axis=3)
        #images = add_pattern_bd(x, pixel_value=self.max_val, channels_first=False)
        
        if len(images.shape) == 5:
                images = images.squeeze(3)
        
        return images
    
    def poison_func_single(self, x):
        if len(x.shape) == 4:
            x = np.transpose(x, (0, 2, 3, 1))
        elif len(x.shape) == 3:
            x = np.transpose(x, (1, 2, 0))
            
        images = add_single_bd(x, pixel_value=self.max_val)
         
        return np.transpose(images, (0, 3, 1, 2))
            
        
    def poison_func_target(self, x):
        x = utils.normalize_imageNet(x)
        mode = None
        if len(x.shape) == 4:
            if x.shape[1] == 3:
                mode = "RGB"
            else:
                mode = "L"
        elif len(x.shape) == 3:
            if x.shape[0] == 3:
                mode = "RGB"
            else:
                mode = "L"
        else:
            mode = "L"
            
        return insert_image(x, backdoor_path=self.target_path, size=self.target_size, random=True, channels_first=True, mode=mode)
    
    def run_badNets(self, dataloader, func_name):
        
        images, y = utils.dataloader_to_numpy(dataloader)
        
        self.max_val = np.max(images)
        
        posion_func = None
        if func_name == "pattern":
            posion_func = self.poison_func_pattern
        elif func_name == "single": 
            posion_func = self.poison_func_single
        elif func_name == "target":
            posion_func = self.poison_func_target
        else:
            print("func name not found!!!")
            exit(0)
            
        (is_poison_train, x_poisoned_raw, y_poisoned_raw) = self.posison_dataset(
            x_clean=images, y_clean=y, poison_func=posion_func
        )
        
        #utils.show_one_image(torch.tensor(x_poisoned_raw[-1]), torch.tensor(y_poisoned_raw[-1]), "./datasets/poison", "poison_test")
        
        # Shuffle training data
        n_train = np.shape(y_poisoned_raw)[0]
        shuffled_indices = np.arange(n_train)
        np.random.shuffle(shuffled_indices)
        x_train = x_poisoned_raw[shuffled_indices]
        y_train = y_poisoned_raw[shuffled_indices]
        
        dataloader_poisoned = utils.numpy_to_dataloader(images=x_train, labels=y_train, batch_size=self.batch_size, is_transform=True)
        #images, labels = next(iter(dataloader_poisoned))
        
        #utils.show_one_image(images[0], labels[0], "./datasets/poison", "poison_test")
        
        return dataloader_poisoned