from collections import OrderedDict
from fl_strategy import centralized

import pandas as pd
import lightning as pl

import torch
import flwr
import os

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    #state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    state_dict = OrderedDict(
        {
            k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
            for k, v in params_dict
        }
    )
    model.load_state_dict(state_dict, strict=True)
    

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

class MedicalClient(flwr.client.NumPyClient):
    """Flower client
    """
    def __init__(self, cid, model, train_loader, test_loader, lr, epoch, num_class, metrics_file_name):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader 
        self.test_loader = test_loader 
        self.lr = lr 
        self.epochs = epoch
        self.num_class = num_class
        self.metrics_file_name = metrics_file_name
        
    def get_parameters(self, config):
        return get_parameters(self.model)
        
    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        print(f"[Client {self.cid}] fit, config: {config}")
        loss, metrics, metrics_epochs_train = centralized.train(self.model, self.train_loader, epochs=self.epochs, lr=self.lr, num_class=self.num_class)
        metrics["client"] = self.cid
        metrics["round"] = config.get("round", 0)
        
        if not os.path.exists(f"train_{self.metrics_file_name}"):
            pd.DataFrame([metrics]).to_csv(f"train_{self.metrics_file_name}", header=True, index=False, mode="a")
        else:
            pd.DataFrame([metrics]).to_csv(f"train_{self.metrics_file_name}", header=False, index=False, mode="a")
        
        return self.get_parameters(self.model), len(self.train_loader), metrics

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        print(f"[Client {self.cid}] fit, config: {config}")
        test_loss, metrics_test,  metrics_epochs_test = centralized.test(model=self.model,test_loader=self.test_loader, num_class=self.num_class, epochs=self.epochs)
        metrics_test["client"] = self.cid
        metrics_test["round"] = config.get("round", 0)
        
        if not os.path.exists(f"test_{self.metrics_file_name}"):
            pd.DataFrame([metrics_test]).to_csv(f"test_{self.metrics_file_name}", header=True, index=False, mode="a")
        else:
            pd.DataFrame([metrics_test]).to_csv(f"test_{self.metrics_file_name}", header=False, index=False, mode="a")
        
        return float(test_loss), len(self.test_loader), metrics_test

class MedicalClientLightning(flwr.client.NumPyClient):
    """Flower client
    """
    def __init__(self, cid, model, train_loader, test_loader, lr, epoch, num_class, metrics_file_name):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader 
        self.test_loader = test_loader 
        self.lr = lr 
        self.epochs = epoch
        self.num_class = num_class
        self.metrics_file_name = metrics_file_name
        
    def get_parameters(self, config):
        return get_parameters(self.model)
        
    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        print(f"[Client {self.cid}] fit, config: {config}")
        trainer = pl.Trainer(max_epochs=self.max_epochs, enable_progress_bar=False)
        trainer.fit(self.model, self.train_loader, self.val_loader)
        
        metrics = trainer.logged_metrics
        
        results =  {
                "client": self.cid,
                "round" : config.get("round", 0),
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
            }
        
        if not os.path.exists(f"train_{self.metrics_file_name}"):
            pd.DataFrame([results]).to_csv(f"train_{self.metrics_file_name}", header=True, index=False, mode="a")
        else:
            pd.DataFrame([results]).to_csv(f"train_{self.metrics_file_name}", header=False, index=False, mode="a")
        
        return self.get_parameters(self.model), len(self.train_loader), results

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        print(f"[Client {self.cid}] fit, config: {config}")
        
        trainer = pl.Trainer(enable_progress_bar=False)
        results = trainer.test(self.model, self.test_loader)
        metrics = results[0]
        test_loss = metrics["test_loss"]
        
        results =  {
                "client": self.cid,
                "round" : config.get("round", 0),
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