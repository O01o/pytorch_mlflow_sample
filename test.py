import mlflow.pytorch
import torch
import torch.nn.functional as F
import typer
from PIL import Image
from torchvision import transforms

MODEL_PATH = "model"  # or "ckpt_20", "ckpt_40", ...

def predict(model_uri: str, image_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルの読み込み
    model = mlflow.pytorch.load_model(model_uri).to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    image = Image.open(image_path).convert("L")  # グレースケールに変換
    image = transform(image).unsqueeze(0).to(device)  # バッチ次元追加

    # 推論
    with torch.no_grad():
        output = model()
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    typer.run(predict)
