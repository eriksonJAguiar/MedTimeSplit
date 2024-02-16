import sys
#sys.path.append("../utils")

import torch
import numpy as np
from utils import utils
import os
import time

from art.estimators.classification import PyTorchClassifier
from art.attacks.poisoning import PoisoningAttackBackdoor, PoisoningAttackCleanLabelBackdoor, BadDetGlobalMisclassificationAttack, HiddenTriggerBackdoorPyTorch, SleeperAgentAttack
from art.attacks.poisoning.perturbations import add_pattern_bd, add_single_bd, insert_image
from art.utils import to_categorical


class PoisonWithBadNets():

    def __init__(self, target_path, poison_percent=0.2, target_size=(5,5)):
        self.target_path = target_path
        self.poison_percent = poison_percent
        self.target_size = target_size
        self.max_val = 0
    
    def posison_dataset(self, x_clean, y_clean, poison_func):
        
        x_poison = np.copy(x_clean)
        y_poison = np.copy(y_clean)
        is_poison = np.zeros(np.shape(y_poison)[0])

        for i in range(10):
            src = i
            tgt = (i + 1) % 10
            n_points_in_tgt = np.round(np.sum(np.argmax(y_clean, axis=0) == tgt))
            num_poison = int((self.poison_percent * n_points_in_tgt) / (1 - self.poison_percent))
            src_imgs = np.copy(x_clean[np.argmax(y_clean, axis=0) == src])

            n_points_in_src = np.shape(src_imgs)[0]
            if num_poison:
                indices_to_be_poisoned = np.random.choice(n_points_in_src, num_poison)

                imgs_to_be_poisoned = src_imgs[indices_to_be_poisoned]
                backdoor_attack = PoisoningAttackBackdoor(poison_func)
                poison_images, poison_labels = backdoor_attack.poison(
                    imgs_to_be_poisoned, y=to_categorical(np.ones(num_poison) * tgt, 10)
                )
                x_poison = np.append(x_poison, poison_images, axis=0)
                y_poison = np.append(y_poison, poison_labels, axis=0)
                is_poison = np.append(is_poison, np.ones(num_poison))

            is_poison = is_poison != 0

            return is_poison, x_poison, y_poison
    
    def poison_func_pattern(self, x):
        return np.expand_dims(add_pattern_bd(x.squeeze(3), pixel_value=self.max_val), axis=3)

    def poison_func_single(self, x):
        return np.expand_dims(add_single_bd(x.squeeze(3), pixel_value=self.max_val), axis=3)
        
    def poison_func_target(self, x):
        return insert_image(x, backdoor_path=self.target_path, size=self.target_size, random=True)
    
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
        
        # Shuffle training data
        n_train = np.shape(y_poisoned_raw)[0]
        shuffled_indices = np.arange(n_train)
        np.random.shuffle(shuffled_indices)
        x_train = x_poisoned_raw[shuffled_indices]
        y_train = y_poisoned_raw[shuffled_indices]
        
        dataloader_poisoned = utils.numpy_to_dataloader(images=x_train, labels=y_train, batch_size=32)
        #utils.show_images(dataloader_poisoned, "Poisoned", "./")
        
        return dataloader_poisoned