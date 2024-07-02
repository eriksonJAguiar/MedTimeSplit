from continual_learning.continual import run_continual
from utils import utils
import pandas as pd
import numpy as np
import os

root_path = os.path.join(
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "datasets",
                "MelanomaDB")
csv_path = os.path.join(root_path, "ISIC_2018_dataset.csv")

lr = 0.001
model_name = "resnet50"
experiences = 7
epoches = 10

domain_type = [None, "normal", "illumination", "occlusion"]
#domain_type = ["normal", "occlusion"]

for domain in domain_type:
    train, test, num_class = utils.load_database_df(
            root_path=root_path,
            csv_path=csv_path,
            batch_size=32, 
            image_size=(128,128),
            is_agumentation=False,
            as_rgb=True,
            is_stream=True,
            domain_type=domain,
    )
    #img, lb = next(iter(train))
    #print(img.shape)
    #print(lb)
    #print(np.expand_dims(img, axis=0).shape)
    #utils.show_one_image(img, lb, ".", f"lesion_img_{domain}")
    results_metrics = run_continual(train, test, num_class, model_name, lr, train_epochs=epoches, experiences=experiences)
    #print("Final Results:")
    #print(len(results_metrics))
    print("Dataframe:")
    domain_data = pd.DataFrame(results_metrics)
    print(domain)
    domain_data.insert(1, "Domain", "NONE" if domain is None else domain.upper())
    print(domain_data)
    
    if os.path.exists("cl_metrics.csv"):
        domain_data.to_csv("cl_metrics.csv", mode="a", header=False, index=False)
    else:
        domain_data.to_csv("cl_metrics.csv", mode="a", header=True, index=False)
           
    