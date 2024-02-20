from fl_strategy.client_structure import MedicalClient
from backdoors.generate_attacks import PoisonWithBadNets
from utils import utils
import flwr
import torch
import os
import argparse

parser = argparse.ArgumentParser(description='')
#parser.add_argument('-dm','--dataset_name', help='databaset name')
#parser.add_argument('-d','--dataset', help='databaset path', required=False)
#parser.add_argument('-dv','--dataset_csv', help='databaset csv file', required=False)

# parser.add_argument('-mn', '--model_name', help="model to training name: resnet50 or resnet18", required=True)
#parser.add_argument('-wp', '--weights_path', help="root of model weigths path", required=True)

# parser.add_argument('-an', '--attack_name', help="Attack name FGSM, PGD, CW or UAP", required=True)
parser.add_argument('-nc', '--num_cli', help="number of clients on the network", required=True)
parser.add_argument('-i', '--cid', help="client number", required=True)
parser.add_argument("-b", "--is_bad", action="store_true", required=False)
#parser.add_argument('-pa', '--path_attack', help="Attack noise", required=True)

args = vars(parser.parse_args())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_path = os.path.join(
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "datasets",
            "MelanomaDB")
csv_path = os.path.join(root_path, "ISIC_2018_dataset.csv")

batch_size = 32
model_name = "resnet50"
lr = 0.001
epochs = 1
num_clients = int(args["num_cli"])
cid = int(args["cid"])
is_bad = args["is_bad"]


train_paramters = utils.load_database_federated(root_path=root_path,
                                                csv_path=csv_path,
                                                num_clients= num_clients,
                                                batch_size=batch_size,
                                                is_agumentation=True,
                                                as_rgb=True)
#adding backdoor into training images
if is_bad:
    badnets = PoisonWithBadNets(target_path="./backdoors/target/alert.png", target_size=(10, 10))
    train_loader = train_paramters["train"][cid]
    train_loader = badnets.run_badNets(train_loader, "pattern")
else:
    train_loader = train_paramters["train"][cid]

test_loader = train_paramters["test"][cid]
num_class = train_paramters["num_class"]

model = utils.make_model_pretrained(model_name=model_name, num_class=num_class)
    
client_features = MedicalClient(cid=cid,
                                model=model, 
                                train_loader=train_loader,
                                test_loader=test_loader, 
                                lr=lr, 
                                epoch=epochs,
                                num_class=num_class)


flwr.client.start_client(
    server_address="[::]:8080",
    client=client_features.to_client(),
)