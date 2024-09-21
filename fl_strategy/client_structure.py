from collections import OrderedDict
from fl_strategy import centralized
from fl_strategy.centralized_lightning import TrainModelLigthning, CustomTimeCallback
from continual_learning import continual
from avalanche.training.supervised.strategy_wrappers_online import OnlineNaive
from avalanche.benchmarks.scenarios.online import split_online_stream
from avalanche.benchmarks import nc_benchmark, ni_benchmark

import pandas as pd
import lightning as pl

import torch
import flwr
import os

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    #state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    state_dict = OrderedDict(
        {
            k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
            for k, v in params_dict
        }
    )
    model.load_state_dict(state_dict, strict=True)
    

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

class MedicalClient(flwr.client.NumPyClient):
    """Flower client
    """
    def __init__(self, cid, model, model_name, train_loader, test_loader, lr, epoch, num_class, metrics_file_name):
        self.cid = cid
        self.model = model
        self.model_name = model_name
        self.train_loader = train_loader 
        self.test_loader = test_loader 
        self.lr = lr 
        self.epochs = epoch
        self.num_class = num_class
        self.metrics_file_name = metrics_file_name
        
    def get_parameters(self, config):
        return get_parameters(self.model)
        
    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        print(f"[Client {self.cid}] fit, config: {config}")
        loss, metrics, _ = centralized.train(self.model, self.train_loader, epochs=self.epochs, lr=self.lr, num_class=self.num_class)
        metrics["client"] = self.cid
        metrics["client"] = self.model_name
        metrics["round"] = config.get("round", 0)
        
        if not os.path.exists(f"train_{self.metrics_file_name}"):
            pd.DataFrame([metrics]).to_csv(f"train_{self.metrics_file_name}", header=True, index=False, mode="a")
        else:
            pd.DataFrame([metrics]).to_csv(f"train_{self.metrics_file_name}", header=False, index=False, mode="a")
        
        return self.get_parameters(self.model), len(self.train_loader), metrics

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        print(f"[Client {self.cid}] fit, config: {config}")
        test_loss, metrics_test,  _ = centralized.test(model=self.model,test_loader=self.test_loader, num_class=self.num_class, epochs=self.epochs)
        metrics_test["client"] = self.cid
        metrics_test["round"] = config.get("round", 0)
        
        if not os.path.exists(f"test_{self.metrics_file_name}"):
            pd.DataFrame([metrics_test]).to_csv(f"test_{self.metrics_file_name}", header=True, index=False, mode="a")
        else:
            pd.DataFrame([metrics_test]).to_csv(f"test_{self.metrics_file_name}", header=False, index=False, mode="a")
        
        return float(test_loss), len(self.test_loader), metrics_test

class MedicalClientLightning(flwr.client.NumPyClient):
    """Flower client
    """
    def __init__(self, cid, model, model_name, train_loader, test_loader, lr, epoch, num_class, metrics_file_name):
        self.cid = cid
        self.train_loader = train_loader 
        self.test_loader = test_loader 
        self.model = model
        self.model_name = model_name
        self.lr = lr 
        self.epochs = epoch
        self.num_class = num_class
        self.metrics_file_name = metrics_file_name
        self.ligh_model = TrainModelLigthning(model_pretrained=self.model, 
                                              num_class=self.num_class, 
                                              lr=self.lr)
        
    def get_parameters(self, config):
        return get_parameters(self.model)
        
    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        print(f"[Client {self.cid}] fit, config: {config}")
        
        #logger = CSVLogger(save_dir=os.path.join(metrics_save_path,"logs", "hold-out"), name="{}-{}".format(model_name, database_name))
        print(f"Dataset lenght {len(self.train_loader)}")
        if len(self.train_loader) == 0:
            raise ValueError("Dataloader is empty")
        
        trainer = pl.Trainer(
            max_epochs= self.epochs,
            accelerator="gpu",
            devices="auto",
            min_epochs=5,
            log_every_n_steps=10,
            deterministic=False,
            enable_progress_bar=False,
        )
        
        trainer.fit(self.ligh_model, self.train_loader, self.test_loader)
        
        metrics = trainer.logged_metrics
        print(metrics)
        
        results =  {
                "client": self.cid,
                "round" : config.get("round", 0),
                "model": self.model_name,
                "train_acc" : metrics["acc"].item(),
                "train_balanced_acc" : metrics["balanced_acc"].item(),
                "train_f1-score": metrics["f1_score"].item(),
                "train_loss":  metrics["loss"].item(),
                "train_precision":  metrics["precision"].item(),
                "train_recall" :  metrics["recall"].item(),
                "train_auc":  metrics["auc"].item(),
                "train_spc":  metrics["specificity"].item(),
                "train_mcc": metrics["mcc"].item(),
                "train_kappa": metrics["kappa"].item(),
                "val_acc" : metrics["acc"].item(),
                "val_balanced_acc" : metrics["balanced_acc"].item(),
                "val_f1-score": metrics["f1_score"].item(),
                "val_loss":  metrics["loss"].item(),
                "val_precision":  metrics["precision"].item(),
                "val_recall" :  metrics["recall"].item(),
                "val_auc":  metrics["auc"].item(),
                "Val_spc":  metrics["specificity"].item(),
                "Val_mcc": metrics["mcc"].item(),
                "val_kappa": metrics["kappa"].item(),
            }
        
        if not os.path.exists(f"train_{self.metrics_file_name}"):
            pd.DataFrame([results]).to_csv(f"train_{self.metrics_file_name}", header=True, index=False, mode="a")
        else:
            pd.DataFrame([results]).to_csv(f"train_{self.metrics_file_name}", header=False, index=False, mode="a")
        
        return self.get_parameters(self.model), len(self.train_loader), results

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        print(f"[Client {self.cid}] fit, config: {config}")
        
        trainer = pl.Trainer(
            accelerator="gpu",
            devices="auto",
            enable_progress_bar=False,
        )
        
        print(f"Dataset lenght {len(self.test_loader)}")
        if len(self.test_loader) == 0:
            raise ValueError("Dataloader is empty")
    
        results = trainer.test(self.ligh_model, self.test_loader)
        print(results)
        metrics = results[0]
        test_loss = metrics["test_loss"]
        print(metrics)
        
        results =  {
                "client": self.cid,
                "round" : config.get("round", 0),
                "model": self.model_name,
                "test_acc" : metrics["test_acc"].item(),
                "test_balanced_acc" : metrics["test_balanced_acc"].item(),
                "test_f1-score": metrics["test_f1_score"].item(),
                "test_loss":  metrics["test_loss"].item(),
                "test_precision":  metrics["test_precision"].item(),
                "test_recall" :  metrics["test_recall"].item(),
                "test_auc":  metrics["test_auc"].item(),
                "test_spc":  metrics["test_specificity"].item(),
                "test_mcc": metrics["test_mcc"].item(),
                "test_kappa": metrics["test_kappa"].item(),
            }
        
        if not os.path.exists(f"test_{self.metrics_file_name}"):
            pd.DataFrame([results]).to_csv(f"test_{self.metrics_file_name}", header=True, index=False, mode="a")
        else:
            pd.DataFrame([results]).to_csv(f"test_{self.metrics_file_name}", header=False, index=False, mode="a")
        
        return float(test_loss), len(self.test_loader), results
    
class MedicalClientContinous(flwr.client.NumPyClient):
    """Flower client
    """
    def __init__(self, cid, model, model_name, train_loader, test_loader, split_method, num_domain, lr, epoch, num_class, metrics_file_name):
        self.cid = cid
        self.model = model
        self.model_name = model_name
        self.train_loader = train_loader 
        self.test_loader = test_loader 
        self.split_method = split_method
        self.num_domain = num_domain
        self.lr = lr 
        self.epochs = epoch
        self.num_class = num_class
        self.metrics_file_name = metrics_file_name
        
        self.benchmark = ni_benchmark(
            train_dataset=train_loader, 
            test_dataset=self.test_loader,
            n_experiences=4, 
            task_labels=False,
            shuffle=True, 
            balance_experiences=True
        )
        
    def get_parameters(self, config):
        return get_parameters(self.model)
        
    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        print(f"[Client {self.cid}] fit, config: {config}")
        #loss, metrics, _ = centralized.train(self.model, self.train_loader, epochs=self.epochs, lr=self.lr, num_class=self.num_class)
        
        # benchmark = ni_benchmark(
        #     train_dataset=self.train, 
        #     test_dataset=self.test,
        #     n_experiences=4, 
        #     task_labels=False,
        #     shuffle=True, 
        #     balance_experiences=True
        # )
    
        train_stream_online = split_online_stream(
            original_stream=self.benchmark.train_stream,
            experience_size=len(self.train_loader)//16,
            drop_last=True
        )
        
        continual.continual_train(
            train_stream_online=train_stream_online,
            model=self.model,
            lr=self.lr,
            num_domains=self.num_domain
        )
        
        
        return self.get_parameters(self.model), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        print(f"[Client {self.cid}] fit, config: {config}")
        #test_loss, metrics_test,  _ = centralized.test(model=self.model,test_loader=self.test_loader, num_class=self.num_class, epochs=self.epochs)
    
        test_stream_online = split_online_stream(
            original_stream=self.benchmark.test_stream,
            experience_size=len(self.test_loader)//16,
            drop_last=True
        )
    
        metrics_test = continual.continual_test(
            test_stram_online=test_stream_online,
            model=self.model,
            split_method=self.split_method,
            round=config.get("round", 0),
            lr=self.lr,
            cli=self.cid,
            num_domains=self.num_domain
        )
        
        metrics_test["client"] = self.cid
        metrics_test["round"] = config.get("round", 0)
        loss = metrics_test["Loss_Exp_eval"]
        print(metrics_test)
        
        if not os.path.exists(f"test_{self.metrics_file_name}"):
            pd.DataFrame([metrics_test]).to_csv(f"test_{self.metrics_file_name}", header=True, index=False, mode="a")
        else:
            pd.DataFrame([metrics_test]).to_csv(f"test_{self.metrics_file_name}", header=False, index=False, mode="a")
        
        return float(loss), len(self.test_loader), metrics_test