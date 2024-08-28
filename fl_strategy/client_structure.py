from collections import OrderedDict
from fl_strategy import centralized

import pandas as pd

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