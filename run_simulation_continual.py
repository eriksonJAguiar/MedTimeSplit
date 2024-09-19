from continual_learning.continual import run_continual
from utils.partitioning import load_database_federated_continous, load_database_federated_continousSplit, load_database_federated_continousPermuted
from utils import utils
import pandas as pd
import numpy as np
import os
import json
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

root_path = os.path.join(
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "dataset",
                "MelanomaDB")
csv_path = os.path.join(root_path, "ISIC_2018_dataset.csv")

lr = 0.001
image_size = (224,224)
model_name = "resnet152"
experiences = 4
epoches = 10
batch_size = 32
train_epochs = 10
num_domains = 4
num_clients = 5

with open("clients_config/clients_params.json", 'r') as f:
    hyper_params_clients = json.load(f)

num_clients_params = len(hyper_params_clients.keys())
if num_clients > num_clients_params:
    raise ValueError("Enhance the number of parameter in file params.json")

split_method = {
    "SplitMnist": load_database_federated_continousSplit(root_path=root_path,
                                                   csv_path=csv_path,
                                                   K=num_clients,
                                                   as_rgb=True,
                                                   image_size=image_size,
                                                ),
    "PermutedMnist": load_database_federated_continousPermuted(root_path=root_path,
                                                               csv_path=csv_path,
                                                               K=num_clients,
                                                               as_rgb=True,
                                                               image_size=image_size
                                                            ),
    "MedTimeSplit": load_database_federated_continous(root_path=root_path,
                                                           csv_path=csv_path,
                                                           K=num_clients,
                                                           as_rgb=True,
                                                           image_size=image_size,
                                                           hyperparams_client=hyper_params_clients),
}

for cli in range(num_clients):
    for key, loader_db in split_method.items():
        print(f"key: {key}")
        train, test, num_class = loader_db   
        
        model = utils.make_model_pretrained(model_name=model_name, num_class=num_class)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.001)
        criterion = CrossEntropyLoss()
        
        results_metrics = run_continual(train[cli], 
                                        test[cli], 
                                        model=model,
                                        optimizer=optimizer, 
                                        criterion=criterion, 
                                        train_epochs=train_epochs, 
                                        experiences=num_domains)

        # loader = DataLoader(test[cli], batch_size=16, shuffle=False)
        # for b in range(len(test[cli])//16):
        #     utils.show_images(loader, f"lesion_img_{key}_{cli}", ".", batch_index=b)
    
        print("Final Results:")
        print("Dataframe:")
        domain_data = pd.DataFrame(results_metrics)
        domain_data.insert(1, "SplitMethod", key)
        domain_data.insert(2, "Client", cli)
        print(domain_data)
    
        if os.path.exists("cl_metrics.csv"):
            domain_data.to_csv("cl_metrics.csv", mode="a", header=False, index=False)
        else:
            domain_data.to_csv("cl_metrics.csv", mode="a", header=True, index=False)
           
    