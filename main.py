import os

from args import Args
from alpha_zero.alpha_zero import AlphaZero
from alpha_zero.alpha_zero_opt import AlphaZeroOpt
from nonaga.engine import Engine
from nonaga.vis import Vis
from nonaga.gamestate import Gamestate
from helpers.validation_game import validation_game, validation_game_with_history


args = Args(
    num_iterations=3,
    num_selfPlay_iterations=2,
    num_parallel_games=2,
    num_simulations=10,
    num_epochs=3,
    num_validation_games=10,

    max_self_play_steps=300,

    c_puct=2,
    temperature=1.25,
    dirichlet_epsilon=0.25,
    dirichlet_alpha=0.3,
    
    batch_size=64,
    lr=0.001,
    weight_decay=0.0001,
    num_resBlocks=16,
    num_hidden=256,
)


alpha_zero = AlphaZeroOpt(args)
for iteration in range(args.num_iterations):
    alpha_zero.learn_iteration(iteration)
    checkpoint_files = [
        f for f in os.listdir("checkpoints")
        if os.path.isfile(os.path.join("checkpoints", f))
    ]
    for checkpoint_file in checkpoint_files:
        old_alpha_zero = AlphaZeroOpt(args, checkpoint_file)
        wins, draws, losses = validate(alpha_zero, old_alpha_zero, args.num_validation_games)
        print(f"Wins: {wins}, Draws: {draws}, Losses: {losses} against {checkpoint_file}")