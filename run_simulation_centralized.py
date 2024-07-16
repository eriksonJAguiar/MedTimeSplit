from fl_strategy import centralized
from utils import utils, partitioning
import torch
import os
import time
import json
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
client_resources = None
if device.type == "cuda":
    client_resources = {"num_gpus": 1}

root_path = os.path.join("datasets", "MelanomaDB")
csv_path = os.path.join(root_path, "ISIC_2018_dataset.csv")

batch_size = 64
model_names = ["resnet50", "vgg16", "vgg19", "inceptionv3", "densenet", "efficientnet"]
#model_names = ["inceptionv3"]
lr = 0.001
epochs = 50
iterations = 10
result_file_name = "no_fed_iid_metrics.csv"
result_file_name_epoch = "no_fed_iid_metrics_epoch.csv"

with open("clients_config/clients_params.json", 'r') as f:
    hyper_params_clients = json.load(f)

num_clients = len(hyper_params_clients.keys())

for i in range(num_clients):
    for model_name in model_names:        
        
        # train, test, num_class = utils.load_database_df(
        #     root_path=root_path,
        #     csv_path=csv_path,
        #     batch_size=32, 
        #     image_size=(299, 299) if model_name == "inceptionv3" else (256, 256),
        #     is_agumentation=True,
        #     as_rgb=True,
        # )
        dataset_params = partitioning.load_database_federated_non_iid(
            root_path=root_path,
            csv_path=csv_path,
            batch_size=32, 
            image_size=(299, 299) if model_name == "inceptionv3" else (256, 256),
            num_clients=num_clients,
            hyperparams_client=hyper_params_clients,
            as_rgb=True,
        )
        
        model = utils.make_model_pretrained(model_name=model_name, num_class=dataset_params["num_class"])

        print("=====================================================")
        print(f'========= ID: {i} ======================')
        print(f'========= Model: {model_name} ======================')
        print("=====================================================")
        
        time_start_train  = time.time()
        print("===== Train Phase ==========")
        loss_train, train_metrics, train_metrics_epoch = centralized.train(model=model,
                                                                           train_loader=dataset_params["train"][i],
                                                                           epochs=epochs,
                                                                           lr=lr,
                                                                           num_class=dataset_params["num_class"])
        time_end_train = time.time()
        time_train  = time_end_train - time_start_train

        
        time_start_test  = time.time()
        print("===== Test Phase ==========")
        loss_test, test_metrics, test_metrics_epoch = centralized.test(model=model,
                                                                        test_loader=dataset_params["test"][i],
                                                                        epochs=epochs,
                                                                        num_class=dataset_params["num_class"])
        time_end_test = time.time()
        test_train  = time_end_test - time_start_test
        

        train_metrics["loss"] = loss_train
        test_metrics["val_loss"] = loss_test
        
        train_metrics["train_time"] = time_train
        test_metrics["val_time"] = test_train
        
        #print(train_metrics)
        #print(test_metrics)
        
        print("Dataframe:")
        train_results = pd.DataFrame([train_metrics])
        test_results = pd.DataFrame([test_metrics])
        final_results = pd.concat([train_results, test_results], axis=1)
        final_results.insert(0, "ID", i)
        final_results.insert(1, "Model", model_name)
        print(final_results)
        
        if os.path.exists(result_file_name):
            final_results.to_csv(result_file_name, mode="a", header=False, index=False)
        else:
            final_results.to_csv(result_file_name, mode="a", header=True, index=False)
            
        train_results_epoch = pd.DataFrame(train_metrics_epoch)
        test_results_epoch = pd.DataFrame(test_metrics_epoch)
        final_results_epoch = pd.concat([train_results_epoch, test_results_epoch], axis=1)
        final_results_epoch.insert(0, "ID", i)
        final_results_epoch.insert(1, "Model", model_name)
        
        if os.path.exists(result_file_name_epoch):
            final_results_epoch.to_csv(result_file_name_epoch, mode="a", header=False, index=False)
        else:
            final_results_epoch.to_csv(result_file_name_epoch, mode="a", header=True, index=False)

