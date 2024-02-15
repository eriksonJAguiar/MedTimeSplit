from fl_strategy import centralized
from utils import utils
import os


def test_centralized():
    
    root_path = os.path.join("datasets", "ISIC2020")
    csv_path = os.path.join(root_path, "ISIC_2020_dataset.csv")
    batch_size = 32
    model_name = "resnet50"
    lr = 0.001
    epochs = 4
    
    # train_loader, test_loader, num_class = utils.load_database_df(root_path=root_path,
    #                                                               csv_path=csv_path,
    #                                                               batch_size=batch_size,
    #                                                               is_agumentation=True,
    #                                                               as_rgb=True)
    
    train_paramters = utils.load_database_federated(root_path=root_path,
                                                    csv_path=csv_path,
                                                    num_clients= 2,
                                                    batch_size=batch_size,
                                                    is_agumentation=True,
                                                    as_rgb=True)
        
    train_loader = train_paramters["train"][1]
    test_loader = train_paramters["test"][1]
    num_class = train_paramters["num_class"]
    
    model = utils.make_model_pretrained(model_name=model_name, num_class=num_class)
    
    train_loss, metrics_train = centralized.train(model=model,
                                                train_loader=train_loader, 
                                                lr=lr,
                                                num_class=num_class,
                                                epochs=epochs)
    
    test_loss, metrics_test = centralized.test(model=model,
                                               test_loader=test_loader, 
                                               num_class=num_class,
                                               epochs=epochs)
    
    print("================= Train ===================================")
    print(f"loss: {train_loss}")
    print(metrics_train)
    
    print("================= Test ===================================")
    print(f"loss: {test_loss}")
    print(metrics_test)              
    

if __name__ == "__main__":
    test_centralized()