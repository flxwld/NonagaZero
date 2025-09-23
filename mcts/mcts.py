import torch
import numpy as np
from typing import List
from tqdm import trange

from nonaga.hex_unit import Hex
from mcts.node import Node
from nonaga.gamestate import Gamestate
from nonaga.engine import Engine
from nonaga.encoder import Encoder
from nonaga.actions import MoveAction, UpAction, DownAction
from args import Args


class MCTS:
    def __init__(self, model, args: Args):
        self.model = model
        self.args = args
    
    @torch.no_grad()
    def search(self, gamestate: Gamestate, is_query: bool = False):
        root = Node(gamestate, 0, None, -1, self.args)

        policy, value = self._get_value_policy(root)
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        action_size = Encoder.MOVE_ACTION_SIZE if root.gamestate.next_action == Gamestate.TurnState.move else Encoder.HEX_ACTION_SIZE
        if not is_query:
            policy = (1 - self.args.dirichlet_epsilon) * policy + self.args.dirichlet_epsilon * np.random.dirichlet([self.args.dirichlet_alpha] * action_size)
        policy = self._validate_policy(root, policy)
        policy /= np.sum(policy)
        root.expand(policy)

        for _ in range(self.args.num_simulations):
            node = root

            while not node.is_leaf():
                node = node.select()
            
            reward = Engine(node.gamestate).get_reward()

            if reward is None:
                policy, value = self._get_value_policy(node)
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                policy = self._validate_policy(node, policy)
                policy /= np.sum(policy)

                reward = value.item()

                node.expand(policy)
            
            node.backpropagate(reward)
        
        if root.gamestate.next_action == Gamestate.TurnState.move:
            action_probs = np.zeros(Encoder.MOVE_ACTION_SIZE)
            for child in root.children:
                action_probs[child.action_taken] = child.visits 
        else:
            action_probs = np.zeros(Encoder.HEX_ACTION_SIZE)
            for child in root.children:
                action_probs[child.action_taken] = child.visits
        action_probs /= np.sum(action_probs)
        return action_probs, root
    
    def _get_value_policy(self, node: Node) -> tuple[torch.Tensor, torch.Tensor]:
        move_policy, up_policy, down_policy, move_value, up_value, down_value = self.model(torch.tensor(Encoder.encode_state(node.gamestate), dtype=torch.float32).unsqueeze(0))
        if node.gamestate.next_action == Gamestate.TurnState.move:
            return move_policy, move_value
        elif node.gamestate.next_action == Gamestate.TurnState.up:
            return up_policy, up_value
        elif node.gamestate.next_action == Gamestate.TurnState.down:
            return down_policy, down_value
    
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
    