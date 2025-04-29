import typer

import src.extensions.line as line


def test_line_notify(message):
    line.notify(message)

if __name__ == "__main__":
    typer.run(test_line_notify)