import torch
import time
import os
import torch.optim as optim
import lightning as L
from torchmetrics.classification import Accuracy, Recall, Specificity, Precision, F1Score, AUROC, ConfusionMatrix, MatthewsCorrCoef, CohenKappa
from utils.metrics import BalancedAccuracy
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import pandas as pd

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"

device = torch.device(dev)
print("Device: {}".format(device))  

class TrainModelLigthning(L.LightningModule):
    """Module to train a model using Pytorch lightning
    """    
    def __init__(self, model_pretrained, num_class, lr):
        """constructor

        Args:
            model_pretrained (torch.nn.Module): architecture pre-trained on imageNet
            num_class (int): number of classes in the dataset
            lr (float): learning rate for training the model
            is_per_class (bool, optional): If is True metrics are calculated per class. Defaults to False.
        """        
        super().__init__()
        self.model = model_pretrained
        self.lr = lr
        self.num_class = num_class
        self.criterion = torch.nn.CrossEntropyLoss() if self.num_class > 2 else torch.nn.BCEWithLogitsLoss()
        #self.criterion = losses.FocalLoss(alpha=1.0, gamma=5.0, reduction="mean")
        #self.criterion = Loss(loss_type="focal_loss", fl_gamma=5)
        #self.criterion = Loss(loss_type="binary_cross_entropy")
        
        self.train_accuracy = Accuracy(task="binary") if not num_class > 2 else Accuracy(task="multiclass", num_classes=num_class)
        self.val_accuracy = Accuracy(task="binary") if not num_class > 2 else Accuracy(task="multiclass", num_classes=num_class)
        self.test_accuracy = Accuracy(task="binary") if not num_class > 2 else Accuracy(task="multiclass", num_classes=num_class)
        
        self.train_recall = Recall(task="binary")  if not num_class > 2 else Recall(task="multiclass", num_classes=num_class, average='weighted')
        self.val_recall =  Recall(task="binary")  if not num_class > 2 else Recall(task="multiclass", num_classes=num_class, average='weighted')
        self.test_recall =  Recall(task="binary")  if not num_class > 2 else Recall(task="multiclass", num_classes=num_class, average='weighted')  
        
        self.train_specificity = Specificity(task="binary") if not num_class > 2 else Specificity(task="multiclass", num_classes=num_class, average='weighted')
        self.val_specificity = Specificity(task="binary") if not num_class > 2 else Specificity(task="multiclass", num_classes=num_class, average='weighted')
        self.test_specificity = Specificity(task="binary") if not num_class > 2 else Specificity(task="multiclass", num_classes=num_class, average='weighted')
        
        self.train_precision = Precision(task="binary") if not num_class > 2 else Precision(task="multiclass", num_classes=num_class, average="weighted")
        self.val_precision = Precision(task="binary") if not num_class > 2 else Precision(task="multiclass", num_classes=num_class, average="weighted")
        self.test_precision = Precision(task="binary") if not num_class > 2 else Precision(task="multiclass", num_classes=num_class, average="weighted")
        
        self.train_f1 = F1Score(task="binary") if not num_class > 2 else F1Score(task="multiclass", num_classes=num_class, average="weighted")
        self.val_f1 = F1Score(task="binary") if not num_class > 2 else F1Score(task="multiclass", num_classes=num_class, average="weighted")
        self.test_f1 = F1Score(task="binary") if not num_class > 2 else F1Score(task="multiclass", num_classes=num_class, average="weighted")
        
        self.train_auc = AUROC(task="binary") if not num_class> 2 else AUROC(task="multiclass", num_classes=num_class, average="weighted")
        self.val_auc = AUROC(task="binary") if not num_class> 2 else AUROC(task="multiclass", num_classes=num_class, average="weighted")
        self.test_auc = AUROC(task="binary") if not num_class> 2 else AUROC(task="multiclass", num_classes=num_class, average="weighted")
        
        self.train_cm = ConfusionMatrix(task="binary", num_classes=2) if not num_class> 2 else ConfusionMatrix(task="multiclass", num_classes=num_class)
        self.val_cm = ConfusionMatrix(task="binary", num_classes=2) if not num_class> 2 else ConfusionMatrix(task="multiclass", num_classes=num_class)
        
        self.train_balanced_acc = BalancedAccuracy(task="binary") if not num_class> 2 else BalancedAccuracy(task="multiclass", num_classes=num_class)
        self.val_balanced_acc = BalancedAccuracy(task="binary") if not num_class> 2 else BalancedAccuracy(task="multiclass", num_classes=num_class)
        self.test_balanced_acc = BalancedAccuracy(task="binary") if not num_class> 2 else BalancedAccuracy(task="multiclass", num_classes=num_class)
        
        self.train_mcc = MatthewsCorrCoef(task="binary") if not num_class> 2 else MatthewsCorrCoef(task="multiclass", num_classes=num_class)
        self.val_mcc = MatthewsCorrCoef(task="binary") if not num_class> 2 else MatthewsCorrCoef(task="multiclass", num_classes=num_class)
        self.test_mcc = MatthewsCorrCoef(task="binary") if not num_class> 2 else MatthewsCorrCoef(task="multiclass", num_classes=num_class)

        self.train_kappa = CohenKappa(task="binary", ) if not num_class> 2 else CohenKappa(task="multiclass", num_classes=num_class)
        self.val_kappa = CohenKappa(task="binary", ) if not num_class> 2 else CohenKappa(task="multiclass", num_classes=num_class)
        self.test_kappa = CohenKappa(task="binary", ) if not num_class> 2 else CohenKappa(task="multiclass", num_classes=num_class)    
        
    def forward(self, x):
        return self.model(x)
    
    def _shared_step(self, batch):
        features, y_true = batch
        logits = self(features)
        loss = self.criterion(logits, y_true)
        y_pred = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        
        return loss, y_true, y_pred, probs
    
    def _shared_step_binary(self, batch):
        features, y_true = batch
        y_true = y_true.view(-1, 1).float()
        logits = self(features)
        loss = self.criterion(logits, y_true)
        y_pred = (logits > 0.3).float()
        probs = torch.sigmoid(logits)
        
        return loss, y_true, y_pred, probs

    def training_step(self, batch, batch_idx):
        loss, y_true, y_pred, probs = self._shared_step(batch) if self.num_class > 2 else self._shared_step_binary(batch)

        self.model.eval()
        with torch.no_grad():
             _, y_true, y_pred, probs = self._shared_step(batch) if self.num_class > 2 else self._shared_step_binary(batch)
            
        
        self.log('loss', loss, prog_bar=True)
        
        self.train_accuracy.update(y_pred, y_true)
        self.train_precision.update(y_pred, y_true)
        self.train_recall.update(y_pred, y_true)
        self.train_specificity.update(y_pred, y_true)
        self.train_f1.update(y_pred, y_true)
        self.train_auc.update(probs, y_true)
        self.train_balanced_acc.update(y_pred, y_true)
        self.train_mcc.update(probs, y_true)
        self.train_kappa.update(y_pred, y_true)
            
        self.log('acc', self.train_accuracy.compute(), prog_bar=True, on_epoch=True, on_step=False)
        self.log('balanced_acc', self.train_balanced_acc.compute(), prog_bar=True, on_epoch=True, on_step=False)
        self.log('precision', self.train_precision.compute(), prog_bar=True, on_epoch=True, on_step=False)
        self.log('recall', self.train_recall.compute(), prog_bar=True, on_epoch=True, on_step=False)
        self.log('f1_score', self.train_f1.compute(), prog_bar=True, on_epoch=True, on_step=False)
        self.log('specificity', self.train_specificity.compute(), prog_bar=True, on_epoch=True, on_step=False)
        self.log('auc', self.train_auc.compute(), prog_bar=True, on_epoch=True, on_step=False)
        self.log('mcc', self.train_mcc.compute(), prog_bar=True, on_epoch=True, on_step=False)
        self.log('kappa', self.train_kappa.compute(), prog_bar=True, on_epoch=True, on_step=False)
        
        self.model.train()
        
        return loss
   
    def validation_step(self, batch, batch_idx):
        
        loss, y_true, y_pred, probs = self._shared_step(batch) if self.num_class > 2 else self._shared_step_binary(batch)

        self.log('val_loss', loss)
        
        self.val_accuracy(y_pred, y_true)
        self.val_precision(y_pred, y_true)
        self.val_recall(y_pred, y_true)
        self.val_specificity(y_pred, y_true)
        self.val_f1(y_pred, y_true)
        self.val_auc(probs, y_true)
        self.val_balanced_acc(y_pred, y_true)
        self.val_mcc(probs, y_true)
        self.val_kappa(y_pred, y_true)
            
        self.log('val_acc', self.val_accuracy, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_balanced_acc', self.val_balanced_acc, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_precision', self.val_precision, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_recall', self.val_recall, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_f1_score', self.val_f1, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_specificity', self.val_specificity, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_auc', self.val_auc, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_mcc', self.val_mcc, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_kappa', self.val_kappa, on_epoch=True, on_step=False, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_id):
        loss, y_true, y_pred, probs = self._shared_step(batch) if self.num_class > 2 else self._shared_step_binary(batch)
       
        self.test_accuracy(y_pred, y_true)
        self.test_balanced_acc(y_pred, y_true)
        self.test_precision(y_pred, y_true)
        self.test_recall(y_pred, y_true)
        self.test_specificity(y_pred, y_true)
        self.test_f1(y_pred, y_true)
        self.test_auc(probs, y_true)
        self.test_mcc(probs, y_true)
        self.test_kappa(y_pred, y_true)
            
        self.log('test_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test_acc', self.val_accuracy, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test_balanced_acc', self.val_accuracy, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test_precision', self.val_precision, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test_recall', self.val_recall, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test_f1_score', self.val_f1, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test_specificity', self.val_specificity, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test_auc', self.val_auc, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test_mcc', self.val_auc, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test_kappa', self.val_auc, on_epoch=True, on_step=False, prog_bar=True)
        
        return loss
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        #optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        #optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        #miletones = [0.5 * 100, 0.75 * 100]
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=miletones, gamma=0.1)
        #scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=10)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        
        #return [optimizer]
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss', 
                'interval': 'epoch',  
                'frequency': 1,       
                'strict': True   
            }
        }

class CustomTimeCallback(Callback):
    
    def __init__(self, file_train, file_test, root_path) -> None:
        super().__init__()
        # if not os.path.exists("../metrics/time"):
        #     os.mkdir("../metrics/time")
        os.makedirs(os.path.join(root_path, "time"), exist_ok=True)
        
        self.file_train = file_train
        self.file_test = file_test
    
    def on_train_start(self, trainer, pl_module):
        self.start_train = time.time()
    
    def on_train_end(self, trainer, pl_module):
        self.train_end = time.time()
        total = (self.train_end - self.start_train)/60
        
        with open(self.file_train, "a") as f:
            f.write("{}\n".format(total))
    
    def on_validation_start(self, trainer, pl_module):
        self.start_test = time.time()
    
    def on_validation_end(self, trainer, pl_module):
        self.test_end = time.time()
        total = (self.test_end - self.start_test)/60
        
        with open(self.file_test, "a") as f:
            f.write("{}\n".format(total))

class PytorchTrainingAndTest():
    """Training and testing model
    """    
    def run_model(self, exp_num, model, model_name, database_name, train, test, learning_rate, num_epochs, metrics_save_path, num_class=2):
        """execute train and test of the model

        Args:
            exp_num (int): index of experiment executed
            model (torch.nn.Module): model will be trained
            model_name (str): model name
            database_name (str): database name
            train (torch.data.utils.Dataloader): train dataloader
            test (torch.data.utils.Dataloader): test datalaoder
            learning_rate (float): learning rate for training the model
            num_epochs (int): number of epochs for training and testing
            metrics_save_path (str): root path to save metrics
            num_class (int, optional): number of class in the dataset. Defaults to 2.

        Returns:
            metrics_df (pd.Dataframe): dataframe with model's performance
        """        
        #init model using a pytorch lightining call
        ligh_model = TrainModelLigthning(model_pretrained=model, 
                                         num_class=num_class, 
                                         lr=learning_rate)
        
        #define callback for earlystopping
        early_stop_callback = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=8, verbose=True, mode='max')
        
        #define custom callback to calculate the train and test time 
        timer = CustomTimeCallback(os.path.join(metrics_save_path, "time", "train_time_{}-{}.csv".format(model_name, database_name)),
                                   os.path.join(metrics_save_path, "time", "test_time_{}-{}.csv".format(model_name, database_name)), 
                                   metrics_save_path)
            
        #Define callback to save the best model weights
        ckp = ModelCheckpoint(dirpath=os.path.join(metrics_save_path,"logs", "hold-out"), 
                              filename="{}-{}-exp{}".format(model_name, database_name, exp_num), 
                              save_top_k=1, 
                              mode="max", 
                              monitor="val_acc",
                              )
            
        #initate callbacks to execute the training
        callbacks=[ckp, timer]
        #callbacks=[ckp, timer]
        
        #define the function to save the logs
        logger = CSVLogger(save_dir=os.path.join(metrics_save_path,"logs", "hold-out"), name="{}-{}".format(model_name, database_name), version=exp_num)
        
        trainer = L.Trainer(
            max_epochs= num_epochs,
            accelerator="gpu",
            devices="auto",
            min_epochs=5,
            log_every_n_steps=10,
            logger=logger,
            deterministic=False,
            callbacks=callbacks
        )
            
        trainer.fit(
            model=ligh_model,
            train_dataloaders=train,
            val_dataloaders=test
        )
        
        metrics = trainer.logged_metrics
        print(metrics)
            
        
        results =  {
                "exp_num": exp_num,
                "model_name" : model_name,
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
                "val_acc" :  metrics["val_acc"].item(),
                "cal_balanced_acc" : metrics["val_balanced_acc"].item(),
                "val_f1-score":  metrics["val_f1_score"].item(),
                "val_loss":  metrics["val_loss"].item(),
                "val_precision":  metrics["val_precision"].item(),
                "val_recall" :  metrics["val_recall"].item(),
                "val_auc":  metrics["val_auc"].item(),
                "val_spc":  metrics["val_specificity"].item(),
                "val_mcc": metrics["val_mcc"].item(),
                "val_kappa": metrics["val_kappa"].item(),
            }
        results = {k:[v] for k,v in results.items()}
        metrics_df = pd.DataFrame(results)
        
        metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
        
        return metrics_df