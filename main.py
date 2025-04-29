import typer

import src.extensions.line as line
import src.train as train


def main(host: str, port: int):
    mlflow_tracking_uri = f"http://{host}:{port}" # "http://your_mlflow_tracking_server"
    try:
        train.train(mlflow_tracking_uri)
        line.notify(f"{train.MLFLOW_EXPERIMENT_NAME} finished successfully!!!")
    except KeyboardInterrupt:
        line.notify(f"{train.MLFLOW_EXPERIMENT_NAME} is forced to terminate")
    except Exception as e:
        line.notify(f"{train.MLFLOW_EXPERIMENT_NAME} failed...\n{e}")

if __name__ == "__main__":
    typer.run(main)