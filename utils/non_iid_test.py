import torch
from utils.partitioning import load_database_df, CustomTransformNoise, CustomTransformIllumination, CustomTransformOcclusion
from scipy.stats import ks_2samp
import torchvision
import numpy as np
from torchvision import transforms
import time

#"mean": 0,
#"sigma": 10,
#"brightness": 10,
#"contrast": 1.4,
#"occlusion_size": 4


def __gereate_params_by_distribution(low, high, size, type=int):
    
    # random_values = None
    #np.random.seed(int(time.time()))
    if type is int:
        random_values = np.random.randint(high=high, low=low, size=size)
    else:
        random_values = float(low) + float((high - low)) * np.random.randn(size)
    
    return random_values


def get_parameters_non_iid(num_ids):
    """generate parameter as non-iid data

    Args:
        path (str): database path
        num_ids (int): number of clients or ids

    Returns:
        cl_hyper_param: parameters of clients representing the domains shift
    """
    cl_hyper_param = {}
    
    params = {
            "mean": np.repeat(0, num_ids).astype(int),
            "sigma": __gereate_params_by_distribution(low=10, high=80, size=num_ids, type=int),
            "brightness": __gereate_params_by_distribution(low=10, high=20, size=num_ids, type=int),
            "contrast":  __gereate_params_by_distribution(low=1.2, high=2, size=num_ids, type=float),
            "occlusion_size": __gereate_params_by_distribution(low=10, high=20, size=num_ids, type=int),
    }
    
    for k in range(0, num_ids):
        cl_hyper_param[str(k)] = {
            "mean": params["mean"][k],
            "sigma": params["sigma"][k],
            "brightness": params["brightness"][k],
            "contrast": params["contrast"][k],
            "occlusion_size": params["occlusion_size"][k],
        }
     
    return cl_hyper_param

def __extract_features(dataloader):
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights)
        model.fc = torch.nn.Identity()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        features = []
        with torch.no_grad():
            for imgs, _ in dataloader:
                imgs.to(device)
            features.extend(imgs)
        
        return np.array(features)


def __create_domains(image_size):
    
    hyperparams_client = get_parameters_non_iid(num_ids=1)
    
    print(hyperparams_client)
    
    domain_j = {
            "noise": transforms.Compose([
                        transforms.Resize(image_size),
                        CustomTransformNoise(mean=hyperparams_client["0"]["mean"], sigma=hyperparams_client["0"]["sigma"]),
                        transforms.ToTensor(),
                        #transforms.v2.GaussianNoise(mean=hyperparams_client[str(cl)]["mean"], sigma=hyperparams_client[str(cl)]["sigma"]),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            "illumination": transforms.Compose([
                        transforms.Resize(image_size),
                        CustomTransformIllumination(brightness=hyperparams_client["0"]["brightness"], contrast=hyperparams_client["0"]["contrast"]),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            "occlusion": transforms.Compose([
                        transforms.Resize(image_size),
                        CustomTransformOcclusion(occlusion_size=hyperparams_client["0"]["occlusion_size"]),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        }
    
    return domain_j, hyperparams_client
    

def load_dataset_with_non_iid_params(root_path, csv_path, batch_size, image_size, num_cli):
    
    # domain_j, hyperparams_client = __create_domains((224,224))
    # exit(0)
    
    train_clean, _, _ = load_database_df(
        root_path=root_path, 
        csv_path=csv_path, 
        batch_size=batch_size,
        image_size=image_size,
        as_rgb=True,
        tf_transformer= transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.ToTensor(),
                        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]),
    )

    features_original = __extract_features(train_clean).flatten()
    
    params_optm = {}
    cli = 1
    
    while True:
        domain_j, hyperparams_client = __create_domains(image_size)
        for k, tf in domain_j.items():
            print(f"Domain: {k}")
            train, _, _ = load_database_df(
                root_path=root_path, 
                csv_path=csv_path, 
                batch_size=batch_size,
                image_size=image_size,
                as_rgb=True,
                tf_transformer=tf
            )
            
            next_features = __extract_features(train).flatten()
            ks_statistic, ks_p_value = ks_2samp(features_original, next_features)
            print(f"P-value: {ks_p_value}")
            
            if ks_p_value < 0.005:
                params_optm[str(cli)] = {}
                if k == "noise":
                    params_optm[str(cli)]["mean"] = hyperparams_client["0"]["mean"]
                    params_optm[str(cli)]["sigma"] = hyperparams_client["0"]["sigma"]
                elif k == "illumination":
                    params_optm[str(cli)]["brightness"] = hyperparams_client["0"]["brightness"] 
                    params_optm[str(cli)]["contrast"] = hyperparams_client["0"]["contrast"]
                elif k == "occlusion":
                    params_optm[str(cli)]["occlusion_size"] = hyperparams_client["0"]["occlusion_size"]
            
        if len(params_optm[str(cli)].keys()) == 3:
            cli =+1
            
        if cli > num_cli:
            break  
        
    return params_optm
        