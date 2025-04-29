# AWS+MLflow+PyTorchではじめる機械学習(PyTorch版)

## 動作環境

| 項目 | 名称 |
| ---- | ---- |
| プログラミング言語 | Python |
| 機械学習フレームワーク | PyTorch |
| 仮想環境 | uv |
| 実験管理 | MLflow |

## 機械学習

| 項目 | 名称 |
| ---- | ---- |
| 学習内容 | 手書き数字分類 |
| 学習データセット | MNIST |
| 学習モデル | CNN |

## 実行手順

1. 本リポジトリをローカルにクローン  
2. uv仮想環境にパッケージを導入
```
uv sync
```
3. MNISTからサンプル画像をいくらか拝借
```
uv run python image_extractor.py
```
4. MLflow Trackerの場所を確認  
ローカルホストまたは別のホストからアクセス  
ネットワークが疎通することを確認  
```
uv init
uv add mlflow
uv sync
uv run mlflow server --host 0.0.0.0 --port 5000
```
5. 学習プログラムを起動
```
uv run python train.py <Trackerのホスト> 5000
```


