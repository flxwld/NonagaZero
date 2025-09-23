from typing import List

from .hex_unit import Hex


class HexBoard:
    def __init__(self, size: int = 2, board: List[Hex] = None):
        if board is None:
            self.board = [
                Hex(row, col)
                for row in range(-size, size + 1)
                for col in range(-size, size + 1)
                if 
                (row) + (col) >= -size and 
                (row) + (col) <= size
            ]
        else:
            self.board = board
    
    def get_board(self) -> List[Hex]:
        return self.board
    
    def copy(self):
        return HexBoard(board=self.board.copy())

