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

    def __init__(self, target_size, target_path=None, poison_percent=0.2):
        self.target_path = target_path
        self.poison_percent = poison_percent
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
        nb_class = len(np.unique(y_clean))

        for i in range(nb_class):
            src = i
            tgt = (i + 1) % nb_class
            n_points_in_tgt = int(np.round(np.sum(y_clean == tgt)))
            print(f"Number of posiong {self.poison_percent}")
            if n_points_in_tgt > 0:
                num_poison = int((self.poison_percent * n_points_in_tgt) / (1 - self.poison_percent))
            else:
                num_poison = 0
            src_imgs = np.copy(x_clean[y_clean == src])

            n_points_in_src = np.shape(src_imgs)[0]
            if num_poison:
                indices_to_be_poisoned = np.random.choice(n_points_in_src, num_poison)

                imgs_to_be_poisoned = src_imgs[indices_to_be_poisoned]
                labels_to_be_poisoned = np.ones(num_poison) * tgt
                backdoor_attack = PoisoningAttackBackdoor(poison_func)
                poison_images, poison_labels = backdoor_attack.poison(
                    imgs_to_be_poisoned, y=labels_to_be_poisoned
                )
                #utils.show_one_image(torch.tensor(poison_images[0]), torch.tensor(poison_labels[0]), "./datasets/poison", "poison_test")
                
                x_poison = np.append(x_poison, poison_images, axis=0)
                y_poison = np.append(y_poison, poison_labels, axis=0)
                is_poison = np.append(is_poison, np.ones(num_poison))

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
        
        dataloader_poisoned = utils.numpy_to_dataloader(images=x_train, labels=y_train, batch_size=32, is_transform=False)
        #images, labels = next(iter(dataloader_poisoned))
        
        #utils.show_one_image(images[0], labels[0], "./datasets/poison", "poison_test")
        
        return dataloader_poisoned