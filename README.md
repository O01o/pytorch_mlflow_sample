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

### LINE通知機能に向けた設定

本システムでは、学習完了時にLINE通知する機能を持っている。公式LINEアカウントを作成し、自分の端末にプッシュ通知するための手順は以下の通りである。

1. [LINE Developers](https://developers.line.biz/console/) にアクセス
2. プロバイダーを作成
3. 作成したプロバイダーからチャネルを作成
4. 作成したチャネルに遷移
5. 「チャネル基本設定」から「あなたのユーザーID」を環境変数 **LINE_INTERNAL_USER_ID** にコピー
6. 「Messaging API設定」から「チャネルアクセストークン」を環境変数 **LINE_CHANNEL_ACCESS_TOKEN** にコピー
7. テスト送信
```
uv run python test_line_notify.py <message>
```

### 機械学習プログラムの実行

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
