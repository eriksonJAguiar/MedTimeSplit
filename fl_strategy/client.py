from collections import OrderedDict
import centralized

import torch
import flwr
import os
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_path = os.path.join("datasets", "MelanomaDB")
csv_path = os.path.join(root_path, "ISIC_2018_dataset.csv")
batch_size = 32
model_name = "resnet50"
lr = 0.001
epochs = 4
    
train_loader, test_loader, num_class = utils.load_database_df(root_path=root_path,
                                                             csv_path=csv_path,
                                                              batch_size=batch_size,
                                                              is_agumentation=True,
                                                             as_rgb=True)

model = utils.make_model_pretrained(model_name=model_name, num_class=num_class)

class MedicalClient(flwr.client.NumPyClient):
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        centralized.train(model, train_loader, epochs=epochs, lr=lr, num_class=num_class)
        return self.get_parameters(config={}), len(train_loader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        test_loss, metrics_test = centralized.test(model=model,test_loader=test_loader, num_class=num_class, epochs=epochs)
        return float(test_loss), len(test_loader), metrics_test


flwr.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=MedicalClient().to_client()
)