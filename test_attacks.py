from backdoors.generate_attacks import PoisonWithBadNets
from utils import utils
import os

root_path = os.path.join(
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "datasets",
            "MelanomaDB")
csv_path = os.path.join(root_path, "ISIC_2018_dataset.csv")
batch_size = 32
model_name = "resnet50"
lr = 0.001
epochs = 4


if __name__ == "__main__":
    
    train_loader, test_loader, num_class = utils.load_database_df(root_path=root_path,
                                                                  csv_path=csv_path,
                                                                  batch_size=batch_size,
                                                                  is_agumentation=False,
                                                                  as_rgb=True)
    
    badnets = PoisonWithBadNets(target_path="", poison_percent=0.1)
    posined_dataloder = badnets.run_badNets(train_loader, "pattern")
    utils.show_all_images(posined_dataloder, db_name="Poisoned", path_to_save="./datasets/poison")