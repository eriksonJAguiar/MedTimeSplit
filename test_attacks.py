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
    #clean_label = PoisonWithCleanLabel(target_path="./backdoors/target/alert.png", target_size=(10, 10), poison_percent=0.1, target_label=1)
    #clean_label.run_cleanLabels(train_loader, "target", model_name=model_name, lr=lr)
    badnets = PoisonWithBadNets(target_path="./backdoors/target/alert.png", target_size=(10, 10))
    posined_dataloder = badnets.run_badNets(train_loader, "target")
    #utils.show_all_images(posined_dataloder, db_name="Poisoned", path_to_save="./datasets/poison")
    images, labels = next(iter(posined_dataloder))
    utils.show_one_image(image=images[0], label=labels[0], path_to_save='./', image_name='test_image')