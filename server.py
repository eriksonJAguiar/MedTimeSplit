import flwr


flwr.server.start_server(
    server_address="0.0.0.0:8080",
    config=flwr.server.ServerConfig(num_rounds=3),
    strategy=flwr.server.strategy.FedAvg()
)