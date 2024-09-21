from fl_strategy import centralized
from utils import utils
from fl_strategy.centralized_lightning import PytorchTrainingAndTest
import torch
import os
import time
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
client_resources = None
if device.type == "cuda":
    client_resources = {"num_gpus": 1}

root_path = os.path.join("dataset", "MelanomaDB")
csv_path = os.path.join(root_path, "ISIC_2018_dataset.csv")
# root_path = os.path.join("dataset", "MelanomaDB20")
# csv_path = os.path.join(root_path, "ISIC_2020_dataset.csv")


# p_values = non_iid_test.load_dataset_with_non_iid_params(
#     root_path=root_path,
#     csv_path=csv_path,
#     batch_size=32,
#     image_size=(224,224),
#     num_cli=3
# )

batch_size = 64
#model_names = ["resnet101", "resnet50", "vgg16", "vgg19", "inceptionv3", "densenet", "efficientnetb7"]
#pid - 1254912
model_names = ["resnet152", "resnet50", "vgg16", "vgg19", "alexnet", "efficientnetb7", "nasnetlarge", "inceptionresnet"]
lr = 0.0001
epochs = 100
iterations = 10
result_file_name = "no_fed_metrics"

for i in range(iterations):
    for model_name in model_names:        
        
        train, test, num_class = utils.load_database_df(
            root_path=root_path,
            csv_path=csv_path,
            batch_size=batch_size, 
            image_size=(299, 299) if model_name == "inceptionv3" else (224, 224),
            is_agumentation=True,
            as_rgb=True,
        )    

        print("=====================================================")
        print(f'========= ID: {i} ======================')
        print(f'========= Model: {model_name} ======================')
        print("=====================================================")
        
        model = utils.make_model_pretrained(model_name=model_name, num_class=num_class)
        # trainer = PytorchTrainingAndTest()
        
        # metrics = trainer.run_model(
        #     exp_num=i,
        #     model=model,
        #     model_name=model_name,
        #     database_name="ISIC2020",
        #     train=train,
        #     test=test,
        #     learning_rate=lr,
        #     num_epochs=epochs,
        #     num_class=num_class,
        #     metrics_save_path="no_fed_logs"
        # )
        
        time_start_train  = time.time()
        print("===== Train Phase ==========")
        loss_train, train_metrics, train_metrics_epoch = centralized.train(model=model,
                                                                           train_loader=train,
                                                                           epochs=epochs,
                                                                           lr=lr,
                                                                           num_class=num_class)
        time_end_train = time.time()
        time_train  = time_end_train - time_start_train

        
        time_start_test  = time.time()
        print("===== Test Phase ==========")
        loss_test, test_metrics, test_metrics_epoch = centralized.test(model=model,
                                                                        test_loader=test,
                                                                        epochs=epochs,
                                                                        num_class=num_class)
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
        
        if os.path.exists(f"{result_file_name}.csv"):
            final_results.to_csv(f"{result_file_name}.csv", mode="a", header=False, index=False)
        else:
            final_results.to_csv(f"{result_file_name}.csv", mode="a", header=True, index=False)
            
        # train_results_epoch = pd.DataFrame(train_metrics_epoch)
        # test_results_epoch = pd.DataFrame(test_metrics_epoch)
        # final_results_epoch = pd.concat([train_results_epoch, test_results_epoch], axis=1)
        # final_results_epoch.insert(0, "ID", i)
        # final_results_epoch.insert(1, "Model", model_name)
        
        # if os.path.exists(f"{result_file_name}_epoch.csv"):
        #     final_results_epoch.to_csv(f"{result_file_name}_epoch.csv", mode="a", header=False, index=False)
        # else:
        #     final_results_epoch.to_csv(f"{result_file_name}_epoch.csv", mode="a", header=True, index=False)

