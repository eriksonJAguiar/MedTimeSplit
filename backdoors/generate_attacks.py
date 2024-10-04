import numpy as np
from utils import utils
import matplotlib.pyplot as plt

from art.attacks.poisoning import PoisoningAttackBackdoor, PoisoningAttackCleanLabelBackdoor, BadDetGlobalMisclassificationAttack, HiddenTriggerBackdoorPyTorch, SleeperAgentAttack
from art.attacks.poisoning.perturbations import add_pattern_bd, add_single_bd, insert_image
from art.utils import to_categorical


class PoisonWithBadNets():
    """PoisonWithBadNets is a class designed to generate poisoned datasets for testing the robustness of machine learning models against backdoor attacks.
    Attributes:
        target_path (str): Path to the target image used in the backdoor attack.
        poison_percent (float): Percentage of the dataset to be poisoned.
        batch_size (int): Batch size for the dataloader.
        target_size (tuple): Size of the target image.
        max_val (int): Maximum pixel value in the dataset.
    Methods:
        __init__(self, target_size, target_path=None, batch_size=32, poison_percent=0.2):
            Initializes the PoisonWithBadNets class with the given parameters.
        posison_dataset(self, x_clean, y_clean, poison_func):
            Generates a poisoned dataset using the specified poisoning function.
                x_clean (numpy.ndarray): Clean input data.
                y_clean (numpy.ndarray): Clean labels.
                poison_func (function): Function to apply the poisoning.
                tuple: A tuple containing a boolean array indicating poisoned samples, poisoned input data, and poisoned labels.
        poison_func_pattern(self, x):
            Applies a pattern-based backdoor to the input data.
                x (numpy.ndarray): Input data.
                numpy.ndarray: Poisoned input data with the pattern-based backdoor.
        poison_func_single(self, x):
            Applies a single-pixel backdoor to the input data.
                x (numpy.ndarray): Input data.
                numpy.ndarray: Poisoned input data with the single-pixel backdoor.
        poison_func_target(self, x):
            Inserts a target image as a backdoor into the input data.
                x (numpy.ndarray): Input data.
                numpy.ndarray: Poisoned input data with the target image backdoor.
        run_badNets(self, dataloader, func_name):
            Runs the backdoor attack on the dataset using the specified poisoning function.
                dataloader (torch.utils.data.DataLoader): Dataloader for the clean dataset.
                func_name (str): Name of the poisoning function to use ('pattern', 'single', or 'target').
                torch.utils.data.DataLoader: Dataloader for the poisoned dataset."""
    
    def __init__(self, target_size, target_path=None, batch_size=32, poison_percent=0.2):
        """
        Initializes the attack generation parameters.

        Args:
            target_size (int): The size of the target dataset.
            target_path (str, optional): The path to the target dataset. Defaults to None.
            batch_size (int, optional): The size of the batches to be used. Defaults to 32.
            poison_percent (float, optional): The percentage of the dataset to be poisoned. Defaults to 0.2.
        """
        self.target_path = target_path
        self.poison_percent = poison_percent
        self.batch_size = batch_size
        self.target_size = target_size
        self.max_val = 0
    
    def posison_dataset(self, x_clean, y_clean, poison_func, target):
        """
        Poisons a given dataset by applying a specified poisoning function to a subset of the data.
        Args:
            x_clean (np.ndarray): The clean input data.
            y_clean (np.ndarray): The clean labels corresponding to the input data.
            poison_func (callable): The function used to poison the data.
            target (int or np.ndarray): The target label(s) for the poisoned data.
        Returns:
            tuple: A tuple containing:
            - is_poison (np.ndarray): A boolean array indicating which samples have been poisoned.
            - x_poison (np.ndarray): The poisoned input data.
            - y_poison (np.ndarray): The poisoned labels.
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

        x_poison[indices_to_be_poisoned] = poison_images
        y_poison[indices_to_be_poisoned] = target
        is_poison[indices_to_be_poisoned] = 1
        
        print(f"Labels Poison: {y_poison}")
        print(f"Labels Original: {y_clean}")

        is_poison = is_poison != 0

        return is_poison, x_poison, y_poison
    
    def poison_func_pattern(self, x):
        """
        Applies a poisoning pattern to the input data.
        This function adds a specific pattern to the input images to create poisoned data. 
        The pattern is added using the `add_pattern_bd` function, and the resulting images 
        are expanded along a new axis. If the resulting images have 5 dimensions, the 
        singleton dimension is removed.
        Args:
            x (numpy.ndarray): The input images to be poisoned. Expected to be in the 
                       format (batch_size, channels, height, width).
        Returns:
            numpy.ndarray: The poisoned images with the pattern added.
        """
        print(x.shape)
        images = np.expand_dims(add_pattern_bd(x, pixel_value=self.max_val, channels_first=True), axis=3)
        #images = add_pattern_bd(x, pixel_value=self.max_val, channels_first=False)
        
        if len(images.shape) == 5:
                images = images.squeeze(3)
        
        return images
    
    def poison_func_single(self, x):
        """
        Applies a poisoning function to a single input tensor by adding a backdoor.
        This function transposes the input tensor to the appropriate shape, applies a backdoor
        using the `add_single_bd` function, and then transposes the tensor back to its original shape.
        Args:
            x (numpy.ndarray): Input tensor. Expected to be either 3D or 4D.
        Returns:
            numpy.ndarray: The poisoned tensor with the same shape as the input.
        """
        if len(x.shape) == 4:
            x = np.transpose(x, (0, 2, 3, 1))
        elif len(x.shape) == 3:
            x = np.transpose(x, (1, 2, 0))
            
        images = add_single_bd(x, pixel_value=self.max_val)
         
        return np.transpose(images, (0, 3, 1, 2))
            
        
    def poison_func_target(self, x):
        """
        Applies a poisoning function to the input image tensor `x` by inserting a backdoor image.
        Args:
            x (torch.Tensor): The input image tensor. It can have a shape of either (N, C, H, W) or (C, H, W),
                              where N is the batch size, C is the number of channels, H is the height, and W is the width.
        Returns:
            torch.Tensor: The poisoned image tensor with the backdoor image inserted.
        Notes:
            - The input tensor `x` is first normalized using `utils.normalize_imageNet`.
            - The function determines the mode of the image (either "RGB" or "L") based on the shape of `x`.
            - The backdoor image is inserted using the `insert_image` function with the specified `backdoor_path`,
              `target_size`, and other parameters.
        """
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
    
    def run_badNets(self, dataloader, func_name, target=1):
        """
        Executes the BadNets attack on the provided dataloader using the specified poisoning function.
        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader containing the clean dataset.
            func_name (str): The name of the poisoning function to use. Options are "pattern", "single", or "target".
            target (int, optional): The target label for the attack. Defaults to 1.
        Returns:
            torch.utils.data.DataLoader: A dataloader containing the poisoned dataset.
        Raises:
            SystemExit: If the provided func_name is not one of the expected values ("pattern", "single", "target").
        Notes:
            - The function converts the dataloader to numpy arrays for processing.
            - It selects the appropriate poisoning function based on the func_name argument.
            - The dataset is poisoned using the selected function.
            - The poisoned dataset is shuffled before being converted back to a dataloader.
        """
        
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
            x_clean=images, y_clean=y, poison_func=posion_func, target=target
        )
        
        n_train = np.shape(y_poisoned_raw)[0]
        shuffled_indices = np.arange(n_train)
        np.random.shuffle(shuffled_indices)
        x_train = x_poisoned_raw[shuffled_indices]
        y_train = y_poisoned_raw[shuffled_indices]
        
        dataloader_poisoned = utils.numpy_to_dataloader(images=x_train, labels=y_train, batch_size=self.batch_size, is_transform=True)
        
        return dataloader_poisoned