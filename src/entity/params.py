from pydantic import BaseModel


class Config(BaseModel):
    batch_size: int
    epoch_loop: int
