import torch
import numpy as np
from typing import List, Type
from tqdm import trange

from nonaga.hex_unit import Hex
from mcts.node import Node
from nonaga.gamestate import Gamestate
from nonaga.engine import Engine
from nonaga.encoder import Encoder
from nonaga.actions import MoveAction, UpAction, DownAction
from args import Args
from alpha_zero.self_play_game import SelfPlayGame

class ModelOutput:
    def __init__(self, move_policy: torch.Tensor, up_policy: torch.Tensor, down_policy: torch.Tensor, move_value: torch.Tensor, up_value: torch.Tensor, down_value: torch.Tensor):
        self.move_policy = move_policy
        self.up_policy = up_policy
        self.down_policy = down_policy
        self.move_value = move_value
        self.up_value = up_value
        self.down_value = down_value
    
    def get_value_policy(self, turn_state: Gamestate.TurnState) -> torch.Tensor:
        if turn_state == Gamestate.TurnState.move:
            return self.move_policy, self.move_value
        elif turn_state == Gamestate.TurnState.up:
            return self.up_policy, self.up_value
        elif turn_state == Gamestate.TurnState.down:
            return self.down_policy, self.down_value
    
    def softmax(self):
        self.move_policy = torch.softmax(self.move_policy, dim=0)
        self.up_policy = torch.softmax(self.up_policy, dim=0)
        self.down_policy = torch.softmax(self.down_policy, dim=0)
    
    def dirichlet_noise(self, epsilon: float, alpha: float):
        self.move_policy = (1 - epsilon) * self.move_policy + epsilon * np.random.dirichlet([alpha] * Encoder.MOVE_ACTION_SIZE)
        self.up_policy = (1 - epsilon) * self.up_policy + epsilon * np.random.dirichlet([alpha] * Encoder.HEX_ACTION_SIZE)
        self.down_policy = (1 - epsilon) * self.down_policy + epsilon * np.random.dirichlet([alpha] * Encoder.HEX_ACTION_SIZE)

class MCTS_Opt:
    def __init__(self, model, args: Args):
        self.model = model
        self.args = args
    
    @torch.no_grad()
    def search(self, self_play_games: List[SelfPlayGame]) -> List[tuple[torch.Tensor, Type[Node]]]:
        rootable_games = [game for game in self_play_games if game.root is None]
        if len(rootable_games) > 0: 
            rootable_gamestates  = torch.tensor(np.stack([Encoder.encode_state(game.gamestate) for game in rootable_games]), dtype=torch.float32, device=self.model.device)
            model_outputs = self._query_model_in_batch(rootable_gamestates)
            
            for model_output in model_outputs:
                    model_output.softmax()
                    model_output.dirichlet_noise(self.args.dirichlet_epsilon, self.args.dirichlet_alpha)
                
            for i, game in enumerate(rootable_games):
                game.root = Node(game.gamestate, 0, None, -1, self.args)

                policy, value = model_outputs[i].get_value_policy(game.gamestate.next_action)
                policy = self._validate_policy(game.root, policy)
                policy /= np.sum(policy)

                game.root.expand(policy)

        while not all(game.root.visits >= self.args.num_simulations for game in self_play_games):
            simulatable_games = [game for game in self_play_games if game.root.visits < self.args.num_simulations]
            for game in simulatable_games:
                game.expandable_node = None
                node = game.root

                while not node.is_leaf():
                    node = node.select()
                
                reward = Engine(node.gamestate).get_reward()

                if reward is not None:
                    node.backpropagate(reward)
                else:
                    game.expandable_node = node

            expandable_games = [game for game in simulatable_games if game.expandable_node is not None]
            if len(expandable_games) > 0:
                gamestates = torch.tensor(np.stack([Encoder.encode_state(game.gamestate) for game in expandable_games]), dtype=torch.float32, device=self.model.device)
                model_outputs = self._query_model_in_batch(gamestates)

                for model_output in model_outputs:
                    model_output.softmax()
            
            for i, game in enumerate(expandable_games):
                node = game.expandable_node
                policy, value = model_outputs[i].get_value_policy(node.gamestate.next_action)
                policy = self._validate_policy(node, policy)
                policy /= np.sum(policy)

                node.expand(policy)

                node.backpropagate(value.item())
    
    def _query_model_in_batch(self, gamestates: torch.Tensor) -> List[ModelOutput]:
        move_policy, up_policy, down_policy, move_value, up_value, down_value = self.model(gamestates)
        return [ModelOutput(move_policy[i], up_policy[i], down_policy[i], move_value[i], up_value[i], down_value[i]) for i in range(len(gamestates))]
    
    def _validate_policy(self, node: Node, policy: torch.Tensor) -> torch.Tensor:
        if node.gamestate.next_action == Gamestate.TurnState.move:
            return [
                policy[i]
                if policy[i] > 0 and Encoder.MOVE_ACTION_SPACE[i] in Engine(node.gamestate).get_moves()
                else 0
                for i in range(Encoder.MOVE_ACTION_SIZE)
            ]
        elif node.gamestate.next_action == Gamestate.TurnState.up:
            return [
                policy[i]
                if policy[i] > 0 and Encoder.UP_ACTION_SPACE[i] in Engine(node.gamestate).get_ups()
                else 0
                for i in range(Encoder.HEX_ACTION_SIZE)
            ]
        elif node.gamestate.next_action == Gamestate.TurnState.down:
            return [
                policy[i]
                if policy[i] > 0 and Encoder.DOWN_ACTION_SPACE[i] in Engine(node.gamestate).get_downs()
                else 0
                for i in range(Encoder.HEX_ACTION_SIZE)
            ]
    