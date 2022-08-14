from dataclasses import dataclass


@dataclass
class Config_class:
    max_epochs: int
    lr: float
    optimizer: str
    alpha: float  # correlation memory
    x_dim: int
    y_dim: int

    lambda_x: float
    lambda_y: float
    sigma_x: float
    sigma_y: float