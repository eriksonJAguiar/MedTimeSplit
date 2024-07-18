from fl_strategy.client_structure import MedicalClient
from utils import utils, partitioning
import pandas as pd
import flwr
import torch
import os
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
client_resources = None
if device.type == "cuda":
    client_resources = {"num_gpus": 1}

root_path = os.path.join("datasets", "MelanomaDB")
csv_path = os.path.join(root_path, "ISIC_2018_dataset.csv")

batch_size = 64
image_size = (256, 256)
model_name = "resnet50"
lr = 0.001
epochs = 50
num_rounds = 10

with open("clients_config/clients_params.json", 'r') as f:
    hyper_params_clients = json.load(f)

num_clients = len(hyper_params_clients.keys())
#num_clients = 2


train_paramters = partitioning.load_database_federated_non_iid(root_path=root_path,
                                                                    csv_path=csv_path,
                                                                    num_clients=num_clients,
                                                                    batch_size=batch_size,
                                                                    as_rgb=True,
                                                                    image_size=image_size,
                                                                    hyperparams_client=hyper_params_clients
                                                                )

results_fl = []

def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    acc = [num_examples * m["val_accuracy"] for num_examples, m in metrics]
    pr = [num_examples * m["val_precision"] for num_examples, m in metrics]
    re = [num_examples * m["val_recall"] for num_examples, m in metrics]
    spc = [num_examples * m["val_specificity"] for num_examples, m in metrics]
    f1 = [num_examples * m["val_f1_score"] for num_examples, m in metrics]
    auc = [num_examples * m["val_auc"] for num_examples, m in metrics]
    
    examples = [num_examples for num_examples, _ in metrics]
    
    results = {
            "val_accuracy": sum(acc)/ sum(examples),
            "val_precision": sum(pr)/sum(examples),
            "val_recall": sum(re)/sum(examples),
            "val_specificity": sum(spc)/sum(examples),
            "val_f1_score": sum(f1)/sum(examples),
            "val_auc": sum(auc)/sum(examples),
            }

    results_fl.append(results)
    
    return results

strategy = flwr.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
)


def client_fn(cid):
    """read client features

    Returns:
        client_features (MedicalClient): a flower client
    """
    train_loader = train_paramters["train"]
    test_loader = train_paramters["test"]
    num_class = train_paramters["num_class"]

    model = utils.make_model_pretrained(model_name=model_name, num_class=num_class)
    
    client_features = MedicalClient(cid=cid,
                                    model=model, 
                                    train_loader=train_loader[int(cid)],
                                    test_loader=test_loader[int(cid)], 
                                    lr=lr, 
                                    epoch=epochs,
                                    num_class=num_class)
    
    return client_features.to_client()


flwr.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=num_clients,
    config=flwr.server.ServerConfig(num_rounds=num_rounds),
    client_resources=client_resources,
    strategy=strategy
)

df_results = pd.DataFrame(results_fl)
df_results.insert(0, "Round", range(len(df_results)))
df_results.to_csv("federated_learning_results.csv", index=False)
print(df_results)