"""
implementation based on flower repository: https://github.com/adap/flower/blob/main/baselines/niid_bench/niid_bench/strategy.py"
This module defines custom federated learning strategies based on the Flower framework.
Classes:
    CustomFedAvg: Custom implementation of the FedAvg strategy.
    CustomFedProx: Custom implementation of the FedProx strategy.
    FedNovaStrategy: Custom FedAvg strategy with FedNova-based configuration and aggregation.
    ScaffoldStrategy: Custom strategy for SCAFFOLD based on the FedAvg class.
    CustomFedNova: Custom implementation of the FedNova strategy.
    CustomScaffold: Custom implementation of the SCAFFOLD strategy.
CustomFedAvg Methods:
    configure_fit(server_round, parameters, client_manager):
        Configure the fit instructions for the clients.
    configure_evaluate(server_round, parameters, client_manager):
        Configure the evaluate instructions for the clients.
CustomFedProx Methods:
    configure_fit(server_round, parameters, client_manager):
        Configure the fit instructions for the clients.
    configure_evaluate(server_round, parameters, client_manager):
        Configure the evaluate instructions for the clients.
FedNovaStrategy Methods:
    aggregate_fit_custom(server_round, server_params, results, failures):
        Aggregate fit results using a weighted average.
    aggregate_fednova(results):
        Implement custom aggregate function for FedNova.
ScaffoldStrategy Methods:
    aggregate_fit(server_round, results, failures):
        Aggregate fit results using a weighted average.
CustomFedNova Methods:
    configure_fit(server_round, parameters, client_manager):
        Configure the fit instructions for the clients.
    configure_evaluate(server_round, parameters, client_manager):
        Configure the evaluate instructions for the clients.
CustomScaffold Methods:
    configure_fit(server_round, parameters, client_manager):
        Configure the fit instructions for the clients.
    configure_evaluate(server_round, parameters, client_manager):
        Configure the evaluate instructions for the clients.
"""
from functools import reduce
from logging import WARNING

import flwr
import numpy as np

from flwr.common.logger import log
from flwr.common.typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate

from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

class CustomFedAvg(flwr.server.strategy.FedAvg):
    """
    CustomFedAvg is a subclass of the FedAvg strategy from Flower framework.
    It customizes the configuration of fit and evaluate instructions sent to clients.
    Methods
    -------
    configure_fit(server_round, parameters, client_manager)
        Create fit instructions with the round number included in the config,
        sample clients, and return their fit instructions.
    configure_evaluate(server_round, parameters, client_manager)
        Create evaluate instructions with the round number included in the config,
        sample clients, and return their evaluate instructions.
    """
    def configure_fit(self, server_round, parameters, client_manager):
        """
        Configure the fit instructions for the clients in a federated learning setup.
        Args:
            server_round (int): The current round number of the federated learning process.
            parameters (flwr.common.Parameters): The parameters to be sent to the clients.
            client_manager (ClientManager): The client manager responsible for handling client sampling.
        Returns:
            List[Tuple[Client, flwr.common.FitIns]]: A list of tuples where each tuple contains a client and 
            the corresponding fit instructions.
        """
        config = {"round": server_round}
        fit_ins = flwr.common.FitIns(parameters, config)
        
        # Sample clients and return their fit instructions
        clients = client_manager.sample(num_clients=self.min_fit_clients)
        return [(client, fit_ins) for client in clients]
    
    def configure_evaluate(self, server_round, parameters, client_manager):
        """
        Configure the evaluation instructions for the given server round.
        Args:
            server_round (int): The current round of the server.
            parameters (flwr.common.Parameters): The parameters to be evaluated.
            client_manager (ClientManager): The client manager to sample clients from.
        Returns:
            List[Tuple[ClientProxy, flwr.common.EvaluateIns]]: A list of tuples containing 
            the client proxies and their corresponding evaluation instructions.
        """
        config = {"round": server_round}
        evaluate_ins = flwr.common.EvaluateIns(parameters, config)
        
        # Sample clients and return their evaluate instructions
        clients = client_manager.sample(num_clients=self.min_evaluate_clients)
        return [(client, evaluate_ins) for client in clients]

class CustomFedProx(flwr.server.strategy.FedProx):
    """
        Configures the evaluate instructions for the clients for a given round.
        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        parameters : flwr.common.Parameters
            The parameters to be sent to the clients.
        client_manager : flwr.server.client_manager.ClientManager
            The client manager to sample clients from.
        Returns
        -------
        List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.EvaluateIns]]
            A list of tuples containing the client proxy and the evaluate instructions.
    """
    def configure_fit(self, server_round, parameters, client_manager):
        """
        Configure the fit instructions for the clients in a federated learning setup.
        Args:
            server_round (int): The current round number of the federated learning process.
            parameters (flwr.common.Parameters): The parameters to be sent to the clients for training.
            client_manager (ClientManager): The client manager responsible for handling client sampling.
        Returns:
            List[Tuple[Client, flwr.common.FitIns]]: A list of tuples where each tuple contains a client and 
            their corresponding fit instructions.
        """
        # Create fit instructions with the round number included in the config
        config = {"round": server_round}
        fit_ins = flwr.common.FitIns(parameters, config)
        
        # Sample clients and return their fit instructions
        clients = client_manager.sample(num_clients=self.min_fit_clients)
        return [(client, fit_ins) for client in clients]
    
    def configure_evaluate(self, server_round, parameters, client_manager):
        """
        Configure the evaluation instructions for the given server round.
        Args:
            server_round (int): The current round of the server.
            parameters (flwr.common.Parameters): The parameters to be evaluated.
            client_manager (ClientManager): The client manager to sample clients from.
        Returns:
            List[Tuple[Client, flwr.common.EvaluateIns]]: A list of tuples where each tuple contains a client and the evaluation instructions.
        """
        config = {"round": server_round}
        evaluate_ins = flwr.common.EvaluateIns(parameters, config)
        
        # Sample clients and return their evaluate instructions
        clients = client_manager.sample(num_clients=self.min_evaluate_clients)
        return [(client, evaluate_ins) for client in clients]

class FedNovaStrategy(flwr.server.strategy.FedAvg):
    """Custom FedAvg strategy with fednova based configuration and aggregation."""
    
    def aggregate_fit_custom(
        self,
        server_round: int,
        server_params: NDArrays,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        total_samples = sum([fit_res.num_examples for _, fit_res in results])
        c_fact = sum(
            [
                float(fit_res.metrics["a_i"]) * fit_res.num_examples / total_samples
                for _, fit_res in results
            ]
        )
        new_weights_results = [
            (result[0], c_fact * (fit_res.num_examples / total_samples))
            for result, (_, fit_res) in zip(weights_results, results)
        ]

        grad_updates_aggregated = self.aggregate_fednova(new_weights_results)
        # Final parameters = server_params - grad_updates_aggregated
        aggregated = [
            server_param - grad_update
            for server_param, grad_update in zip(server_params, grad_updates_aggregated)
        ]

        parameters_aggregated = ndarrays_to_parameters(aggregated)
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated


    def aggregate_fednova(self, results: List[Tuple[NDArrays, float]]) -> NDArrays:
        """Implement custom aggregate function for FedNova."""
        # Create a list of weights, each multiplied by the weight_factor
        weighted_weights = [
            [layer * factor for layer in weights] for weights, factor in results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime

class ScaffoldStrategy(flwr.server.strategy.FedAvg):
    """Implement custom strategy for SCAFFOLD based on FedAvg class."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        combined_parameters_all_updates = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        len_combined_parameter = len(combined_parameters_all_updates[0])
        num_examples_all_updates = [fit_res.num_examples for _, fit_res in results]
        # Zip parameters and num_examples
        weights_results = [
            (update[: len_combined_parameter // 2], num_examples)
            for update, num_examples in zip(
                combined_parameters_all_updates, num_examples_all_updates
            )
        ]
        # Aggregate parameters
        parameters_aggregated = aggregate(weights_results)

        # Zip client_cv_updates and num_examples
        client_cv_updates_and_num_examples = [
            (update[len_combined_parameter // 2 :], num_examples)
            for update, num_examples in zip(
                combined_parameters_all_updates, num_examples_all_updates
            )
        ]
        aggregated_cv_update = aggregate(client_cv_updates_and_num_examples)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return (
            ndarrays_to_parameters(parameters_aggregated + aggregated_cv_update),
            metrics_aggregated,
        )

class CustomFedNova(FedNovaStrategy):
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

class CustomScaffold(ScaffoldStrategy):
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