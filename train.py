import os

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import typer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from entity.params import Config
from models.simple_cnn import SimpleCNN
from utils import get_metrics, load_yaml

MLFLOW_EXPERIMENT_NAME = "MNIST_OCR_Experiment"
CONFIG_PATH = "./config/params.yaml"
CHECKPOINT_INTEVAL = 1
CHECKPOINT_PATH = "ckpt"

def main(host: str, port: int):
    mlflow_tracking_uri = f"http://{host}:{port}" # "http://your_mlflow_tracking_server"
    
    mlflow.end_run()
    
    config: Config = load_yaml.config_by_yaml(CONFIG_PATH)
    print("config:", config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run() as run:
        mlflow.log_param("batch_size", config.batch_size)
        mlflow.log_param("epoch_loop", config.epoch_loop)
        global_step = 0

        for i, epoch in enumerate(range(config.epoch_loop)):
            model.train()
            print(f"epoch: {i}")
            for data, target in tqdm(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                global_step += 1
                mlflow.log_metric("loss", loss.item(), step=global_step)
                mlflow.log_metric("variance", get_metrics.get_variance(loss).item(), step=global_step)
            
            if (i+1) % CHECKPOINT_INTEVAL == 0:
                mlflow.pytorch.log_model(model, f"{CHECKPOINT_PATH}_{i+1}")

        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    typer.run(main)
