"""
Module to train and test models centralized
"""
import torch
from torchmetrics.classification import Accuracy, Recall, Specificity, Precision, F1Score, AUROC, MatthewsCorrCoef, CohenKappa
from utils.metrics import BalancedAccuracy, Bias, Prevalence
from kornia import losses

#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def train(model, train_loader, epochs, lr, num_class):
    """function to train a model

    Args:
        model (torch.nn.Module): model that will be trained
        train_loader (torch.utils.data.Dataloader): train dataloader with image and labels
        epochs (int): number epochs
        lr (float): learning rate
        num_class (int): number of classes in the training
        
    Returns:
        loss (float): loss value in train
        metrics (dict): performance metrics calculate during train
        metrics_epochs_train: perfomance metrics per epoch during train
    """
    #criterion = torch.nn.CrossEntropyLoss() if num_class > 2 else torch.nn.BCEWithLogitsLoss()
    criterion = losses.FocalLoss(alpha=1.0, gamma=5.0, reduction="mean")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    total, loss_val = 0, 0.0
    
    train_accuracy = Accuracy(task="binary").to(device) if not num_class > 2 else Accuracy(task="multiclass", num_classes=num_class, average='weighted').to(device)
    train_recall = Recall(task="binary").to(device)  if not num_class > 2 else Recall(task="multiclass", num_classes=num_class, average='weighted').to(device)
    train_specificity = Specificity(task="binary").to(device) if not num_class > 2 else Specificity(task="multiclass", num_classes=num_class, average='weighted').to(device)
    train_precision = Precision(task="binary").to(device) if not num_class > 2 else Precision(task="multiclass", num_classes=num_class, average="weighted").to(device)
    train_f1 = F1Score(task="binary").to(device) if not num_class > 2 else F1Score(task="multiclass", num_classes=num_class, average="weighted").to(device)
    train_auc = AUROC(task="binary").to(device) if not num_class> 2 else AUROC(task="multiclass", num_classes=num_class, average="weighted").to(device)
    train_balanced_acc = BalancedAccuracy(task="binary").to(device) if not num_class> 2 else BalancedAccuracy(task="multiclass", num_classes=num_class).to(device)
    #train_prevalence = Prevalence(task="binary").to(device) if not num_class> 2 else Prevalence(task="multiclass", num_classes=num_class).to(device)
    #train_bias = Bias(task="binary").to(device) if not num_class> 2 else Bias(task="multiclass", num_classes=num_class).to(device)
    train_mcc = MatthewsCorrCoef(task="binary").to(device) if not num_class> 2 else MatthewsCorrCoef(task="multiclass", num_classes=num_class).to(device)
    train_kappa = CohenKappa(task="binary", ).to(device) if not num_class> 2 else CohenKappa(task="multiclass", num_classes=num_class).to(device)
    
    #writer = SummaryWriter()
    metrics_epochs_train  = []
    
    for e in range(epochs):
        running_loss = 0
        for data in train_loader:
            x, y  = data
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            
            loss_val += loss.item()
            running_loss += loss.item()
            
            y_pred = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1) if num_class > 2 else torch.sigmoid(logits)
            total += y.size(0)
            
            train_accuracy(y_pred, y)
            train_precision(y_pred, y)
            train_recall(y_pred, y)
            train_specificity(y_pred, y)
            train_f1(y_pred, y)
            train_auc(probs, y)
            train_balanced_acc(y_pred, y)
            train_mcc(probs, y)
            train_kappa(y_pred, y)
            
            # writer.add_scalar("loss x epoch", running_loss/len(train_loader), e)
            # writer.add_scalar("accuracy x epoch", train_accuracy.compute(), e)
            # writer.add_scalar("precision x epoch", train_precision.compute(), e)
            # writer.add_scalar("recall x epoch", train_recall.compute(), e)
            # writer.add_scalar("specificity x epoch", train_specificity.compute(), e)
            # writer.add_scalar("f1_score x epoch", train_f1.compute(), e)
            # writer.add_scalar("auc x epoch", train_auc.compute(), e)
            
        epochs_metrics  = {
                           "epoch": e,
                           "loss": running_loss/len(train_loader),
                           "accuracy": train_accuracy.compute().item(),
                           "balanced_acc": train_balanced_acc.compute().item(),
                           "precision": train_precision.compute().item(),
                           "recall": train_precision.compute().item(),
                           "specificity": train_specificity.compute().item(),
                           "f1_score": train_f1.compute().item(),
                           "auc": train_auc.compute().item(),
                           "mcc": train_mcc.compute().item(),
                           "kappa": train_kappa.compute().item(),
                        }
        metrics_epochs_train.append(epochs_metrics)
        
        #writer.close()
    
    metrics = {
               "accuracy": train_accuracy.compute().item(),
               "balanced_acc": train_balanced_acc.compute().item(),
               "precision": train_precision.compute().item(),
               "recall": train_precision.compute().item(),
               "specificity": train_specificity.compute().item(),
               "f1_score": train_f1.compute().item(),
               "auc": train_auc.compute().item(),
               "mcc": train_mcc.compute().item(),
               "kappa": train_kappa.compute().item(),
            }
    
    loss_val = loss_val/total
    
    return loss_val, metrics, metrics_epochs_train

def test(model, test_loader, epochs, num_class):
    """function to test a model

    Args:
        model (torch.nn.Module): model that will be tested
        test_loader (torch.utils.data.Dataloader): test dataloader with image and labels
        epochs (int): number epochs
        num_class (int): number of classes in testing phase
        
    Returns:
        loss (float): loss value in test
        metrics (dict): performance metrics calculate during test
        metrics_epochs_test: perfomance metrics per epoch during test
    """
    #criterion = torch.nn.CrossEntropyLoss() if num_class > 2 else torch.nn.BCEWithLogitsLoss()
    criterion = losses.FocalLoss(alpha=1.0, gamma=2.0, reduction="mean")
    total, loss_val = 0, 0.0
    
    test_accuracy = Accuracy(task="binary").to(device) if not num_class > 2 else Accuracy(task="multiclass", num_classes=num_class, average='weighted').to(device)
    test_recall = Recall(task="binary").to(device)  if not num_class > 2 else Recall(task="multiclass", num_classes=num_class, average='weighted').to(device)
    test_specificity = Specificity(task="binary").to(device) if not num_class > 2 else Specificity(task="multiclass", num_classes=num_class, average='weighted').to(device)
    test_precision = Precision(task="binary").to(device) if not num_class > 2 else Precision(task="multiclass", num_classes=num_class, average="weighted").to(device)
    test_f1 = F1Score(task="binary").to(device) if not num_class > 2 else F1Score(task="multiclass", num_classes=num_class, average="weighted").to(device)
    test_auc = AUROC(task="binary").to(device) if not num_class> 2 else AUROC(task="multiclass", num_classes=num_class, average="weighted").to(device)
    test_balanced_acc = BalancedAccuracy(task="binary").to(device) if not num_class> 2 else BalancedAccuracy(task="multiclass", num_classes=num_class).to(device)
    #test_prevalence = Prevalence(task="binary").to(device) if not num_class> 2 else Prevalence(task="multiclass", num_classes=num_class).to(device)
    #test_bias = Bias(task="binary").to(device) if not num_class> 2 else Bias(task="multiclass", num_classes=num_class).to(device)
    test_mcc = MatthewsCorrCoef(task="binary").to(device) if not num_class> 2 else MatthewsCorrCoef(task="multiclass", num_classes=num_class).to(device)
    test_kappa = CohenKappa(task="binary", ).to(device) if not num_class> 2 else CohenKappa(task="multiclass", num_classes=num_class).to(device)
    
    #writer = SummaryWriter()
    metrics_epochs_test = []
    
    with torch.no_grad():
        for e in range(epochs):
            running_loss = 0
            for data in test_loader:
                x, y  = data
                x, y = x.to(device), y.to(device)
                
                logits = model(x)
                loss = criterion(logits, y)
                
                loss_val += loss.item()
                running_loss += loss_val
                
                y_pred = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1) if num_class > 2 else torch.sigmoid(logits)
                total += y.size(0)
                
                test_accuracy(y_pred, y)
                test_precision(y_pred, y)
                test_recall(y_pred, y)
                test_specificity(y_pred, y)
                test_f1(y_pred, y)
                test_auc(probs, y)
                test_balanced_acc(y_pred, y)
                test_mcc(probs, y)
                test_kappa(y_pred, y)
                
        #         writer.add_scalar("test loss x epoch", running_loss/len(test_loader), e)
        #         writer.add_scalar("test accuracy x epoch", test_accuracy.compute(), e)
        #         writer.add_scalar("test precision x epoch", test_precision.compute(), e)
        #         writer.add_scalar("test recall x epoch", test_recall.compute(), e)
        #         writer.add_scalar("test specificity x epoch", test_specificity.compute(), e)
        #         writer.add_scalar("test f1_score x epoch", test_f1.compute(), e)
        #         writer.add_scalar("test auc x epoch", test_auc.compute(), e)
        
            epoch_metrics = {
                        "val_epoch": e,
                        "val_loss": running_loss/len(test_loader),
                        "val_accuracy": test_accuracy.compute().item(),
                        "val_balanced_acc": test_balanced_acc.compute().item(),
                        "val_precision": test_precision.compute().item(),
                        "val_recall": test_precision.compute().item(),
                        "val_specificity": test_specificity.compute().item(),
                        "val_f1_score": test_f1.compute().item(),
                        "val_auc": test_auc.compute().item(),
                        "val_mcc": test_mcc.compute().item(),
                        "val_kappa": test_kappa.compute().item(),
                    }
                
            metrics_epochs_test.append(epoch_metrics)
        
        # writer.close()
    
    metrics = {
               "val_accuracy": test_accuracy.compute().item(),
               "val_balanced_acc": test_balanced_acc.compute().item(),
               "val_precision": test_precision.compute().item(),
               "val_recall": test_precision.compute().item(),
               "val_specificity": test_specificity.compute().item(),
               "val_f1_score": test_f1.compute().item(),
               "val_auc": test_auc.compute().item(),
               "val_mcc": test_mcc.compute().item(),
               "val_kappa": test_kappa.compute().item(),
            }
    
    loss_val = loss_val/total
    
    return loss_val, metrics, metrics_epochs_test