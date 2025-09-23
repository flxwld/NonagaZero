from math import sqrt
from typing import List, Optional, Type

from nonaga.engine import Engine
from nonaga.gamestate import Gamestate
from nonaga.encoder import Encoder
from args import Args


class Node:
    def __init__(self, gamestate: Gamestate, prior: float, parent: Optional[Type['Node']], action_taken: int, args: Args):
        self.gamestate = gamestate
        self.parent = parent
        self.prior = prior
        self.action_taken = action_taken

        self.args = args

        self.children = []

        self.visits = 0
        self.value = 0
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def select(self) -> Type['Node']:
        best_child = None
        best_score = float('-inf')
        for child in self.children:
            score = self._get_child_score(child)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child
    
    def _get_child_score(self, child: Type['Node']) -> float:
        mean_child_value = child.value / child.visits if child.visits > 0 else 0
        exploration_term = self.args.c_puct * child.prior * sqrt(self.visits) / (child.visits + 1)
        return mean_child_value + exploration_term
    
    def expand(self, policy: List[float]) -> Type['Node']:
        if self.gamestate.next_action == Gamestate.TurnState.move and len(Engine(self.gamestate).get_moves()) == 0:
            child_gamestate = self.gamestate.copy()
            child_gamestate.next_action = Gamestate.TurnState.up
            self.children.append(Node(child_gamestate, 1, self, -1, self.args))
            return
        for action_index, prob in enumerate(policy):
            if not prob > 0: continue
            child_gamestate = self.gamestate.copy()
            if child_gamestate.next_action == Gamestate.TurnState.move:
                move = Encoder.MOVE_ACTION_SPACE[action_index]
                engine = Engine(child_gamestate)
                engine.make_move(move)
                child_gamestate = engine.gamestate
            elif child_gamestate.next_action == Gamestate.TurnState.up:
                up = Encoder.UP_ACTION_SPACE[action_index]
                engine = Engine(child_gamestate)
                engine.make_up(up)
                child_gamestate = engine.gamestate
            elif child_gamestate.next_action == Gamestate.TurnState.down:
                down = Encoder.DOWN_ACTION_SPACE[action_index]
                engine = Engine(child_gamestate)
                engine.make_down(down)
                engine.turn_board()
                child_gamestate = engine.gamestate
            child = Node(child_gamestate, prob, self, action_index, self.args)
            self.children.append(child)
    
    def backpropagate(self, value: float):
        self.visits += 1
        self.value += value

        if self.parent:
            if self.gamestate.next_action == Gamestate.TurnState.move:
                self.parent.backpropagate(-value)
            else:
                self.parent.backpropagate(value)

