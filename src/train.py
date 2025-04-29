import glob
import os

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from entity.params import Config
from models.simple_cnn import SimpleCNN
from utils import load_yaml

MLFLOW_EXPERIMENT_NAME = "MNIST_OCR_Experiment"
CONFIG_PATH = "./resources/config/params.yaml"
DATASET_PATH = "./resources/datasets/"

def train(mlflow_tracking_uri: str):
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
    train_dataset = datasets.MNIST(DATASET_PATH, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run() as run:
        mlflow.log_param("batch_size", config.batch_size)
        mlflow.log_param("epoch_loop", config.epoch_loop)
        step = 0

        for epoch in range(config.epoch_loop):
            model.train()
            print(f"epoch: {epoch}")
            for data, target in tqdm(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                step += 1
                mlflow.log_metric("loss", loss.item(), step=step)
                            
            model.eval()
            image_path_list = glob.glob(os.path.join("output_images", "*.png"))
            for image_path in image_path_list:
                image = Image.open(image_path).convert("L")  # グレースケールに変換
                image = transform(image).unsqueeze(0).to(device)  # バッチ次元追加

                with torch.no_grad():
                    output = model(image)
                    probabilities = F.softmax(output, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()

                print(f"image_name: {os.path.basename(image_path)} Predicted class: {predicted_class}")