from continual_learning.continual import run_continual
from utils.partitioning import load_database_federated_continous
from utils import utils
import pandas as pd
import os
import json

root_path = os.path.join(
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "dataset",
                "MelanomaDB")
csv_path = os.path.join(root_path, "ISIC_2018_dataset.csv")

lr = 0.001
image_size = (224,224)
model_name = "resnet50"
experiences = 4
epoches = 10
batch_size = 32

with open("clients_config/clients_params.json", 'r') as f:
    hyper_params_clients = json.load(f)

num_clients = len(hyper_params_clients.keys())

train, test, num_class = load_database_federated_continous(root_path=root_path,
                                                     csv_path=csv_path,
                                                     K=num_clients,
                                                     batch_size=batch_size,
                                                     as_rgb=True,
                                                     image_size=image_size,
                                                     hyperparams_client=hyper_params_clients
                                                    )
# train = train_paramters["train"][0]
# test = train_paramters["test"][0]
# num_class = train_paramters["num_class"]

# print(train_paramters["train"])

results_metrics = run_continual(train[0], test[0], num_class, model_name, lr, train_epochs=epoches, experiences=len(train[0]))

#for client in domain_type:
    
    #img, lb = next(iter(train))
    #print(img.shape)
    #print(lb)
    #print(np.expand_dims(img, axis=0).shape)
    #utils.show_one_image(img, lb, ".", f"lesion_img_{domain}")
    #results_metrics = run_continual(train, test, num_class, model_name, lr, train_epochs=epoches, experiences=experiences)
    #print("Final Results:")
    #print(len(results_metrics))
    # print("Dataframe:")
    # domain_data = pd.DataFrame(results_metrics)
    # print(domain)
    # domain_data.insert(1, "Domain", "NONE" if domain is None else domain.upper())
    # print(domain_data)
    
    # if os.path.exists("cl_metrics.csv"):
    #     domain_data.to_csv("cl_metrics.csv", mode="a", header=False, index=False)
    # else:
    #     domain_data.to_csv("cl_metrics.csv", mode="a", header=True, index=False)
           
    