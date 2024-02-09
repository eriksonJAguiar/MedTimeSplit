from fl_strategy.client_structure import MedicalClient
from utils import utils
import flwr
import torch
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
client_resources = None
if device.type == "cuda":
    client_resources = {"num_gpus": 1}

root_path = os.path.join("datasets", "MelanomaDB")
csv_path = os.path.join(root_path, "ISIC_2018_dataset.csv")

batch_size = 32
model_name = "resnet50"
lr = 0.001
epochs = 4
num_clients = 2


def client_fn(cid):
    """read client features

    Returns:
        client_features (MedicalClient): a flower client
    """    

    train_paramters = utils.load_database_federated(root_path=root_path,
                                                    csv_path=csv_path,
                                                    num_clients= num_clients,
                                                    batch_size=batch_size,
                                                    is_agumentation=True,
                                                    as_rgb=True)
        
    train_loader = train_paramters["train"]
    test_loader = train_paramters["test"]
    num_class = train_paramters["num_class"]

    model = utils.make_model_pretrained(model_name=model_name, num_class=num_class)
    
    client_features = MedicalClient(cid=cid,
                                    model=model, 
                                    train_loader=train_loader[cid],
                                    test_loader=test_loader[cid], 
                                    lr=lr, 
                                    epoch=epochs,
                                    num_class=num_class)
    
    return client_features.to_client()


flwr.simulation.start_simulation(
    client_fn=client_fn(1),
    num_clients=num_clients,
    config=flwr.server.ServerConfig(num_rounds=3),
    client_resources=client_resources,
)