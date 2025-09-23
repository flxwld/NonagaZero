from dataclasses import dataclass
from enum import Enum
from typing import List

from .hex_unit import Hex
from .hexboard import HexBoard


@dataclass()
class Gamestate:
    class TurnState(Enum):
        move = 0  
        up = 1  
        down = 2

    def __init__(self, board: HexBoard = None, own_stones: List[Hex] = None, enemy_stones: List[Hex] = None, next_action: TurnState = None, hex_up: Hex = None, last_hex_relocated: Hex = None):
        self.board = board if board is not None else HexBoard()
        self.own_stones = own_stones if own_stones is not None else [Hex(-2, 0), Hex(2, -2), Hex(0, 2)]
        self.enemy_stones = enemy_stones if enemy_stones is not None else [Hex(0, -2), Hex(-2, 2), Hex(2, 0)]
        self.next_action = next_action if next_action is not None else self.TurnState.move
        self.hex_up = hex_up if hex_up is not None else None
        self.last_hex_relocated = last_hex_relocated if last_hex_relocated is not None else None
    
    def copy(self):
        return Gamestate(
            board=self.board.copy(),
            own_stones=self.own_stones.copy(),
            enemy_stones=self.enemy_stones.copy(),
            next_action=self.next_action,
            hex_up=self.hex_up,
            last_hex_relocated=self.last_hex_relocated
        )

