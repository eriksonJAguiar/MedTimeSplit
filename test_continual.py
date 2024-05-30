from continual_learning.continual import run_continual
from utils import utils
import os
import pandas as pd

root_path = os.path.join(
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "datasets",
                "MelanomaDB")
csv_path = os.path.join(root_path, "ISIC_2018_dataset.csv")

lr = 0.001
train_epochs = 10
model_name = "resnet50"

#domain_type = [None, "normal", "illumination", "occlusion"]
domain_type = ["normal", "occlusion"]

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
    results_metrics  = run_continual(train, test, num_class, model_name, lr, train_epochs=2, experiences=1)
    #print(results_metrics)
    print(pd.DataFrame.from_dict(results_metrics).head())
    