from fl_strategy.client_structure import MedicalClient
from utils import utils
import flwr
import torch
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_path = os.path.join("datasets", "MelanomaDB")
csv_path = os.path.join(root_path, "ISIC_2018_dataset.csv")

batch_size = 32
model_name = "resnet50"
lr = 0.001
epochs = 1

def client_load(cid, num_clients):
    """read client features

    Returns:
        client_features (MedicalClient): a flower client
    """    

    train_loader, test_loader, num_class = utils.load_database_federated(
                                                    root_path=root_path,
                                                    csv_path=csv_path,
                                                    num_clients= num_clients,
                                                    batch_size=batch_size,
                                                    is_agumentation=True,
                                                    as_rgb=True
                                            )

    model = utils.make_model_pretrained(model_name=model_name, num_class=num_class)
    
    client_features = MedicalClient(cid=1,
                                    model=model, 
                                    train_loader=train_loader[cid],
                                    test_loader=test_loader[cid], 
                                    lr=lr, 
                                    epoch=epochs,
                                    num_class=num_class)
    
    return client_features.to_client()


flwr.client.start_client(
    server_address="[::]:8080",
    client_fn=client_load(0, 2)
)