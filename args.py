from dataclasses import dataclass

@dataclass
class Args:
    c_puct: float
    num_simulations: int
    max_self_play_steps: int
    temperature: float
    dirichlet_epsilon: float
    dirichlet_alpha: float
    batch_size: int
    num_iterations: int
    num_selfPlay_iterations: int
    num_epochs: int
    lr: float
    weight_decay: float
    num_resBlocks: int
    num_hidden: int
    num_parallel_games: int
    num_validation_games: int
