from continual_learning.continual import run_continual
import os

root_path = os.path.join(
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "datasets",
                "MelanomaDB")
csv_path = os.path.join(root_path, "ISIC_2018_dataset.csv")

run_continual(root_path, csv_path)