from collections import OrderedDict
from fl_strategy import centralized
from fl_strategy.centralized_lightning import TrainModelLigthning, CustomTimeCallback
from continual_learning import continual
from avalanche.training.supervised.strategy_wrappers_online import OnlineNaive
from avalanche.benchmarks.scenarios.online import split_online_stream
from avalanche.benchmarks import ni_benchmark
from backdoors.generate_attacks import PoisonWithBadNets

import pandas as pd
import lightning as pl

import torch
import flwr
import os

TARGET_PATH = "/home/eriksonaguiar/codes/fl_medical/backdoors/target/alert.png"

def set_parameters(model, parameters):
    """
    Set the parameters of a given model.
    Args:
        model (torch.nn.Module): The model whose parameters are to be set.
        parameters (iterable): An iterable containing the new parameters for the model.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    #state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    state_dict = OrderedDict(
        {
            k: torch.tensor(v, dtype=torch.float32) if v.shape != torch.Size([]) else torch.tensor([0], dtype=torch.float32)
            for k, v in params_dict
        }
    )
    model.load_state_dict(state_dict, strict=True)
    

def get_parameters(model):
    """
    Extracts and returns the parameters of a given model as a list of NumPy arrays.
    Args:
        model (torch.nn.Module): The PyTorch model from which to extract parameters.
    Returns:
        List[np.ndarray]: A list of NumPy arrays containing the parameters of the model.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

class MedicalClient(flwr.client.NumPyClient):
    """
    A custom client class for federated learning in a medical context, extending the Flower NumPyClient.
    Attributes:
        cid (str): Client ID.
        model (torch.nn.Module): The machine learning model.
        model_name (str): The name of the model.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        test_loader (torch.utils.data.DataLoader): DataLoader for testing data.
        lr (float): Learning rate for training.
        epochs (int): Number of epochs for training.
        num_class (int): Number of classes in the dataset.
        metrics_file_name (str): Filename for saving metrics.
        batch_size (int, optional): Batch size for training. Default is 32.
        is_attack (bool, optional): Flag indicating if the client is performing an attack. Default is False.
        poisoning_percent (float, optional): Percentage of data to be poisoned if performing an attack. Default is 0.0.
    Methods:
        get_parameters(config):
            Returns the model parameters.
        fit(parameters, config):
            Trains the model with the given parameters and configuration.
            Returns the updated model parameters, number of training samples, and training metrics.
        evaluate(parameters, config):
            Evaluates the model with the given parameters and configuration.
            Returns the test loss, number of test samples, and test metrics.
    """
    def __init__(self, cid, model, model_name, train_loader, test_loader, lr, epoch, num_class, metrics_file_name, batch_size=32, is_attack=False, poisoning_percent=0.0):
        """
        Initializes the client structure for federated learning.
        Args:
            cid (int): Client ID.
            model (torch.nn.Module): The model to be trained.
            model_name (str): Name of the model.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            test_loader (torch.utils.data.DataLoader): DataLoader for testing data.
            lr (float): Learning rate for the optimizer.
            epoch (int): Number of epochs for training.
            num_class (int): Number of classes in the dataset.
            metrics_file_name (str): File name to save metrics.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            is_attack (bool, optional): Flag to indicate if the client is under attack. Defaults to False.
            poisoning_percent (float, optional): Percentage of data poisoning. Defaults to 0.0.
        """
        self.cid = cid
        self.model = model
        self.model_name = model_name
        self.train_loader = train_loader 
        self.test_loader = test_loader 
        self.lr = lr 
        self.epochs = epoch
        self.num_class = num_class
        self.metrics_file_name = metrics_file_name
        self.is_attack = is_attack
        self.poisoning_percent = poisoning_percent
        self.batch_size = batch_size
        
    def get_parameters(self, config):
        """
        Retrieve the parameters of the model.
        Args:
            config (dict): Configuration dictionary (not used in the current implementation).
        Returns:
            list: Parameters of the model.
        """
        return get_parameters(self.model)
        
    def fit(self, parameters, config):
        """
        Train the model on the client's data.
        Args:
            parameters (list): The parameters to set in the model before training.
            config (dict): Configuration dictionary containing training settings.
        Returns:
            tuple: A tuple containing the model parameters after training, the number of samples used for training, and a dictionary of training metrics.
        The function performs the following steps:
        1. Sets the model parameters.
        2. Prints the client ID and configuration.
        3. Checks if the client is under attack:
            - If not, trains the model using the centralized training method.
            - If under attack, applies a poisoning attack to the training data and then trains the model.
        4. Adds client-specific information to the metrics.
        5. Saves the metrics to a CSV file.
        6. Returns the updated model parameters, the number of training samples, and the metrics.
        """
        set_parameters(self.model, parameters)
        print(f"[Client {self.cid}] fit, config: {config}")
        if not self.is_attack:
            _, metrics, _ = centralized.train(self.model, self.train_loader, epochs=self.epochs, lr=self.lr, num_class=self.num_class)
        else:
            attack = PoisonWithBadNets(target_size=(10, 10), poison_percent=self.poisoning_percent, batch_size=self.batch_size)
            self.train_loader = attack.run_badNets(self.train_loader, "pattern")
            _, metrics, _ = centralized.train(self.model, self.train_loader, epochs=self.epochs, lr=self.lr, num_class=self.num_class)
               
        metrics["client"] = self.cid
        metrics["model"] = self.model_name
        metrics["round"] = config.get("round", 0)
        
        metrics["poisoning"] = float(self.poisoning_percent) 
        
        print(metrics)
        
        if not os.path.exists(f"train_{self.metrics_file_name}"):
            pd.DataFrame([metrics]).to_csv(f"train_{self.metrics_file_name}", header=True, index=False, mode="a")
        else:
            pd.DataFrame([metrics]).to_csv(f"train_{self.metrics_file_name}", header=False, index=False, mode="a")
        
        return self.get_parameters(self.model), len(self.train_loader), metrics

    def evaluate(self, parameters, config):
        """
        Evaluate the model on the test dataset and log the metrics.
        Args:
            parameters (dict): The parameters to set in the model.
            config (dict): Configuration dictionary containing evaluation settings.
        Returns:
            tuple: A tuple containing:
                - float: The test loss.
                - int: The number of samples in the test loader.
                - dict: The evaluation metrics including client ID, round number, model name, and poisoning percentage.
        Notes:
            - The function sets the model parameters, evaluates the model on the test dataset, and logs the metrics.
            - If the metrics file does not exist, it creates a new file with headers; otherwise, it appends to the existing file.
        """
        set_parameters(self.model, parameters)
        print(f"[Client {self.cid}] fit, config: {config}")
        #attack = PoisonWithBadNets(target_size=(10, 10), poison_percent=self.poisoning_percent, batch_size=self.batch_size)
        #self.train_loader = attack.run_badNets(self.train_loader, "pattern")
        #test_loss, metrics_test,  _ = centralized.test(model=self.model,test_loader=self.test_loader, num_class=self.num_class, epochs=self.epochs)
        # if not self.is_attack is None:
        #     test_loss, metrics_test,  _ = centralized.test(model=self.model,test_loader=self.test_loader, num_class=self.num_class, epochs=self.epochs)
        # else:
        #     attack = PoisonWithBadNets(target_size=(20, 20), target_path=TARGET_PATH, poison_percent=self.poisoning_percent)
        #     self.test_loader = attack.run_badNets(self.test_loader, "target")
        #     test_loss, metrics_test,  _ = centralized.test(model=self.model,test_loader=self.test_loader, num_class=self.num_class, epochs=self.epochs)
        test_loss, metrics_test,  _ = centralized.test(model=self.model,test_loader=self.test_loader, num_class=self.num_class, epochs=self.epochs)
        
        metrics_test["client"] = self.cid
        metrics_test["round"] = config.get("round", 0)
        metrics_test["model"] = self.model_name
        
        #if self.is_attack:
        metrics_test["poisoning"] = float(self.poisoning_percent) 
        
        if not os.path.exists(f"test_{self.metrics_file_name}"):
            pd.DataFrame([metrics_test]).to_csv(f"test_{self.metrics_file_name}", header=True, index=False, mode="a")
        else:
            pd.DataFrame([metrics_test]).to_csv(f"test_{self.metrics_file_name}", header=False, index=False, mode="a")
        
        return float(test_loss), len(self.test_loader), metrics_test

class MedicalClientLightning(flwr.client.NumPyClient):
    """Flower client
    """
    def __init__(self, cid, model, model_name, train_loader, test_loader, lr, epoch, num_class, metrics_file_name):
        self.cid = cid
        self.train_loader = train_loader 
        self.test_loader = test_loader 
        self.model = model
        self.model_name = model_name
        self.lr = lr 
        self.epochs = epoch
        self.num_class = num_class
        self.metrics_file_name = metrics_file_name
        self.ligh_model = TrainModelLigthning(model_pretrained=self.model, 
                                              num_class=self.num_class, 
                                              lr=self.lr)
        
    def get_parameters(self, config):
        return get_parameters(self.model)
        
    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        print(f"[Client {self.cid}] fit, config: {config}")
        
        #logger = CSVLogger(save_dir=os.path.join(metrics_save_path,"logs", "hold-out"), name="{}-{}".format(model_name, database_name))
        print(f"Dataset lenght {len(self.train_loader)}")
        if len(self.train_loader) == 0:
            raise ValueError("Dataloader is empty")
        
        trainer = pl.Trainer(
            max_epochs= self.epochs,
            accelerator="gpu",
            devices="auto",
            min_epochs=5,
            log_every_n_steps=10,
            deterministic=False,
            enable_progress_bar=False,
        )
        
        trainer.fit(self.ligh_model, self.train_loader, self.test_loader)
        
        metrics = trainer.logged_metrics
        print(metrics)
        
        results =  {
                "client": self.cid,
                "round" : config.get("round", 0),
                "model": self.model_name,
                "train_acc" : metrics["acc"].item(),
                "train_balanced_acc" : metrics["balanced_acc"].item(),
                "train_f1-score": metrics["f1_score"].item(),
                "train_loss":  metrics["loss"].item(),
                "train_precision":  metrics["precision"].item(),
                "train_recall" :  metrics["recall"].item(),
                "train_auc":  metrics["auc"].item(),
                "train_spc":  metrics["specificity"].item(),
                "train_mcc": metrics["mcc"].item(),
                "train_kappa": metrics["kappa"].item(),
                "val_acc" : metrics["acc"].item(),
                "val_balanced_acc" : metrics["balanced_acc"].item(),
                "val_f1-score": metrics["f1_score"].item(),
                "val_loss":  metrics["loss"].item(),
                "val_precision":  metrics["precision"].item(),
                "val_recall" :  metrics["recall"].item(),
                "val_auc":  metrics["auc"].item(),
                "Val_spc":  metrics["specificity"].item(),
                "Val_mcc": metrics["mcc"].item(),
                "val_kappa": metrics["kappa"].item(),
            }
        
        if not os.path.exists(f"train_{self.metrics_file_name}"):
            pd.DataFrame([results]).to_csv(f"train_{self.metrics_file_name}", header=True, index=False, mode="a")
        else:
            pd.DataFrame([results]).to_csv(f"train_{self.metrics_file_name}", header=False, index=False, mode="a")
        
        return self.get_parameters(self.model), len(self.train_loader), results

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        print(f"[Client {self.cid}] fit, config: {config}")
        
        trainer = pl.Trainer(
            accelerator="gpu",
            devices="auto",
            enable_progress_bar=False,
        )
        
        print(f"Dataset lenght {len(self.test_loader)}")
        if len(self.test_loader) == 0:
            raise ValueError("Dataloader is empty")
    
        results = trainer.test(self.ligh_model, self.test_loader)
        print(results)
        metrics = results[0]
        test_loss = metrics["test_loss"]
        print(metrics)
        
        results =  {
                "client": self.cid,
                "round" : config.get("round", 0),
                "model": self.model_name,
                "test_acc" : metrics["test_acc"].item(),
                "test_balanced_acc" : metrics["test_balanced_acc"].item(),
                "test_f1-score": metrics["test_f1_score"].item(),
                "test_loss":  metrics["test_loss"].item(),
                "test_precision":  metrics["test_precision"].item(),
                "test_recall" :  metrics["test_recall"].item(),
                "test_auc":  metrics["test_auc"].item(),
                "test_spc":  metrics["test_specificity"].item(),
                "test_mcc": metrics["test_mcc"].item(),
                "test_kappa": metrics["test_kappa"].item(),
            }
        
        if not os.path.exists(f"test_{self.metrics_file_name}"):
            pd.DataFrame([results]).to_csv(f"test_{self.metrics_file_name}", header=True, index=False, mode="a")
        else:
            pd.DataFrame([results]).to_csv(f"test_{self.metrics_file_name}", header=False, index=False, mode="a")
        
        return float(test_loss), len(self.test_loader), results
    
class MedicalClientContinous(flwr.client.NumPyClient):
    """
    A Flower client for continuous federated learning in a medical setting.
    Attributes:
        cid (str): Client ID.
        model (torch.nn.Module): The model to be trained and evaluated.
        model_name (str): Name of the model.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        split_method (str): Method used for splitting the data.
        num_domain (int): Number of domains for continual learning.
        lr (float): Learning rate.
        epochs (int): Number of epochs for training.
        num_class (int): Number of classes in the dataset.
        metrics_file_name (str): File name for saving metrics.
        is_attack (bool): Flag indicating if the client is under attack.
        poisoning_percent (float): Percentage of data to be poisoned if under attack.
    Methods:
        __init__(self, cid, model, model_name, train_loader, test_loader, split_method, num_domain, lr, epoch, num_class, metrics_file_name, is_attack=False, poisoning_percent=0.0):
            Initializes the MedicalClientContinous with the given parameters.
        get_parameters(self, config):
            Returns the model parameters.
        fit(self, parameters, config):
            Trains the model on the client's data and returns the updated parameters.
        evaluate(self, parameters, config):
            Evaluates the model on the client's test data and returns the evaluation metrics.
    """
    
    def __init__(self, cid, model, model_name, train_loader, test_loader, split_method, num_domain, lr, epoch, num_class, metrics_file_name, is_attack=False, poisoning_percent=0.0):
        """
        Initializes the client structure with the given parameters.
        Args:
            cid (int): Client ID.
            model (torch.nn.Module): The model to be used.
            model_name (str): The name of the model.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
            test_loader (torch.utils.data.DataLoader): DataLoader for the testing data.
            split_method (str): Method used for splitting the data.
            num_domain (int): Number of domains.
            lr (float): Learning rate.
            epoch (int): Number of epochs.
            num_class (int): Number of classes.
            metrics_file_name (str): Name of the file to store metrics.
            is_attack (bool, optional): Flag indicating if an attack is to be performed. Defaults to False.
            poisoning_percent (float, optional): Percentage of data to be poisoned if attack is True. Defaults to 0.0.
        Attributes:
            cid (int): Client ID.
            model (torch.nn.Module): The model to be used.
            model_name (str): The name of the model.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
            test_loader (torch.utils.data.DataLoader): DataLoader for the testing data.
            split_method (str): Method used for splitting the data.
            num_domain (int): Number of domains.
            lr (float): Learning rate.
            epochs (int): Number of epochs.
            num_class (int): Number of classes.
            metrics_file_name (str): Name of the file to store metrics.
            is_attack (bool): Flag indicating if an attack is to be performed.
            poisoning_percent (float): Percentage of data to be poisoned if attack is True.
            benchmark (ni_benchmark): Benchmark object for training and testing.
        """
        self.cid = cid
        self.model = model
        self.model_name = model_name
        self.train_loader = train_loader 
        self.test_loader = test_loader 
        self.split_method = split_method
        self.num_domain = num_domain
        self.lr = lr 
        self.epochs = epoch
        self.num_class = num_class
        self.metrics_file_name = metrics_file_name
        self.is_attack=is_attack
        self.poisoning_percent=poisoning_percent
        
        if self.is_attack:
            attack = PoisonWithBadNets(target_size=(10, 10), target_path=TARGET_PATH, poison_percent=self.poisoning_percent)
            self.train_loader = attack.run_badNets(self.train_loader, "target")
        
        self.benchmark = ni_benchmark(
            train_dataset=self.train_loader, 
            test_dataset=self.test_loader,
            n_experiences=4, 
            task_labels=False,
            shuffle=True, 
            balance_experiences=True
        )
        
    def get_parameters(self, config):
        """
        Retrieve the parameters of the model.
        Args:
            config (dict): Configuration dictionary (currently unused).
        Returns:
            list: Parameters of the model.
        """
        return get_parameters(self.model)
        
    def fit(self, parameters, config):
        """
        Fit the model with the given parameters and configuration.
        Args:
            parameters (list): The parameters to set in the model.
            config (dict): Configuration settings for the fitting process.
        Returns:
            tuple: A tuple containing the model parameters after fitting, the length of the training loader, and an empty dictionary.
        """
        set_parameters(self.model, parameters)
        print(f"[Client {self.cid}] fit, config: {config}")
    
        train_stream_online = split_online_stream(
            original_stream=self.benchmark.train_stream,
            experience_size=len(self.train_loader)//16,
            drop_last=True
        )
        
        continual.continual_train(
            train_stream_online=train_stream_online,
            model=self.model,
            lr=self.lr,
            num_domains=self.num_domain
        )
        
        return self.get_parameters(self.model), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        """
        Evaluate the model on the test data and log the metrics.
        Args:
            parameters (dict): The parameters to set in the model.
            config (dict): Configuration dictionary containing various settings.
        Returns:
            tuple: A tuple containing:
            - float: The loss value.
            - int: The length of the test loader.
            - dict: The metrics dictionary containing evaluation results.
        The function performs the following steps:
        1. Sets the model parameters.
        2. Prints the client ID and configuration.
        3. Splits the test stream into smaller online streams.
        4. Evaluates the model using continual testing.
        5. Adds client ID and round information to the metrics.
        6. If an attack is present, adds poisoning percentage to the metrics.
        7. Logs the metrics to a CSV file.
        8. Returns the loss, length of the test loader, and the metrics dictionary.
        """
        set_parameters(self.model, parameters)
        print(f"[Client {self.cid}] fit, config: {config}")
        #test_loss, metrics_test,  _ = centralized.test(model=self.model,test_loader=self.test_loader, num_class=self.num_class, epochs=self.epochs)
    
        test_stream_online = split_online_stream(
            original_stream=self.benchmark.test_stream,
            experience_size=len(self.test_loader)//16,
            drop_last=True
        )
    
        metrics_test = continual.continual_test(
            test_stram_online=test_stream_online,
            model=self.model,
            split_method=self.split_method,
            round=config.get("round", 0),
            lr=self.lr,
            cli=self.cid,
            num_domains=self.num_domain
        )
        
        metrics_test["client"] = self.cid
        metrics_test["round"] = config.get("round", 0)
        
        if self.is_attack:
            metrics_test["poisoning"] = float(self.poisoning_percent) 
        
        loss = metrics_test["Loss_Exp_eval"]
        print(metrics_test)
        
        if not os.path.exists(f"test_{self.metrics_file_name}"):
            pd.DataFrame([metrics_test]).to_csv(f"test_{self.metrics_file_name}", header=True, index=False, mode="a")
        else:
            pd.DataFrame([metrics_test]).to_csv(f"test_{self.metrics_file_name}", header=False, index=False, mode="a")
        
        return float(loss), len(self.test_loader), metrics_test