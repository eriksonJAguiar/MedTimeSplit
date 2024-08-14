import flwr

class CustomFedAvg(flwr.server.strategy.FedAvg):
    def configure_fit(self, server_round, parameters, client_manager):
        # Create fit instructions with the round number included in the config
        config = {"round": server_round}
        fit_ins = flwr.common.FitIns(parameters, config)
        
        # Sample clients and return their fit instructions
        clients = client_manager.sample(num_clients=self.min_fit_clients)
        return [(client, fit_ins) for client in clients]
    
    def configure_evaluate(self, server_round, parameters, client_manager):
        config = {"round": server_round}
        evaluate_ins = flwr.common.EvaluateIns(parameters, config)
        
        # Sample clients and return their evaluate instructions
        clients = client_manager.sample(num_clients=self.min_evaluate_clients)
        return [(client, evaluate_ins) for client in clients]