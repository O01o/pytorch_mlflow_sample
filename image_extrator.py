import os

import torch
from PIL import Image
from torchvision import datasets, transforms

# 保存ディレクトリ
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# データセットのダウンロードと前処理
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# 保存する画像数
num_images = 12  # 必要な画像数を変更

for i in range(num_images):
    image, label = mnist_dataset[i]  # i番目の画像とラベル
    image = image.squeeze(0)  # [1, 28, 28] -> [28, 28] に変換
    image = Image.fromarray((image.numpy() * 255).astype("uint8"))  # PIL形式に変換
    
    image_path = os.path.join(output_dir, f"mnist_{i}_label_{label}.png")
    image.save(image_path)

    print(f"Saved: {image_path}")
