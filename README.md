# AWS+MLflow+PyTorchではじめる機械学習(PyTorch版)

## 動作環境

| 項目 | 名称 |
| ---- | ---- |
| プログラミング言語 | Python |
| 機械学習フレームワーク | PyTorch |
| 仮想環境 | Rye |
| 実験管理 | MLflow |

## 機械学習

| 項目 | 名称 |
| ---- | ---- |
| 学習内容 | 手書き数字分類 |
| 学習データセット | MNIST |
| 学習モデル | CNN |

## 実行手順

1. 本リポジトリをローカルにクローン  
2. Rye仮想環境にパッケージを導入
```
rye sync
```
3. MNISTからサンプル画像をいくらか拝借
```
rye run python image_extractor.py
```
4. MLflow Trackerの場所を確認  
ローカルホストまたは別のホストからアクセス  
ネットワークが疎通することを確認  
```
rye init
rye add mlflow
rye sync
rye run mlflow server --host 0.0.0.0 --port 5000
```
5. 学習プログラムを起動
```
rye run python train.py <Trackerのホスト> 5000
```


