from collections import OrderedDict
from fl_strategy import centralized

import torch
import flwr


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class MedicalClient(flwr.client.NumPyClient):
    """Flower client
    """
    def __init__(self, cid, model, train_loader, test_loader, lr, epoch, num_class):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader 
        self.test_loader = test_loader 
        self.lr = lr 
        self.epochs = epoch
        self.num_class = num_class
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        
    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        print(f"[Client {self.cid}] fit, config: {config}")
        centralized.train(self.model, self.train_loader, epochs=self.epochs, lr=self.lr, num_class=self.num_class)
        return self.get_parameters(self.model), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        print(f"[Client {self.cid}] fit, config: {config}")
        test_loss, metrics_test = centralized.test(model=self.model,test_loader=self.test_loader, num_class=self.num_class, epochs=self.epochs)
        return float(test_loss), len(self.test_loader), metrics_test