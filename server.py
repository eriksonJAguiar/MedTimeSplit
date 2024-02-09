import flwr

def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    acc = [num_examples * m["val_accuracy"] for num_examples, m in metrics]
    pr = [num_examples * m["val_precision"] for num_examples, m in metrics]
    re = [num_examples * m["val_recall"] for num_examples, m in metrics]
    spc = [num_examples * m["val_specificity"] for num_examples, m in metrics]
    f1 = [num_examples * m["val_f1_score"] for num_examples, m in metrics]
    auc = [num_examples * m["val_auc"] for num_examples, m in metrics]
    
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {
            "val_accuracy": sum(acc)/ sum(examples),
            "val_precision": sum(pr)/sum(examples),
            "val_recall": sum(re)/sum(examples),
            "val_specificity": sum(spc)/sum(examples),
            "val_f1_score": sum(f1)/sum(examples),
            "val_auc": sum(auc)/sum(examples),
            }

flwr.server.start_server(
    server_address="0.0.0.0:8080",
    config=flwr.server.ServerConfig(num_rounds=3),
    strategy=flwr.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average
    )
)