import typer

import src.train as train


def main(host: str, port: int):
    mlflow_tracking_uri = f"http://{host}:{port}" # "http://your_mlflow_tracking_server"
    train.train(mlflow_tracking_uri)

if __name__ == "__main__":
    typer.run(main)
