import torch
from typing import List

from .hex_unit import Hex
from .actions import MoveAction, UpAction, DownAction
from .gamestate import Gamestate

def get_mask(half_size: int):
    return {
        Hex(row, col) 
        for col in range(-half_size, half_size + 1)
        for row in range(-half_size, half_size + 1)
        if row + col >= -half_size and row + col <= half_size
    }

def get_hex_vector(half_size: int, mask: set[Hex]):
    return [
        Hex(row, col)
        for col in range(-half_size, half_size + 1)
        for row in range(-half_size, half_size + 1)
        if Hex(row, col) in mask
    ]

def get_move_action_space(hex_vector: List[Hex]):
    return [
        MoveAction(hex_vector[i], d)
        for d in range(6)
        for i in range(len(hex_vector))
    ]

def get_up_action_space(hex_vector: List[Hex]):
    return [
        UpAction(hex_vector[i])
        for i in range(len(hex_vector))
    ]

def get_down_action_space(hex_vector: List[Hex]):
    return [
        DownAction(hex_vector[i])
        for i in range(len(hex_vector))
    ]

class Encoder:
    SIZE = 19
    HALF_SIZE = SIZE // 2

    MASK = get_mask(HALF_SIZE)

    HEX_VECTOR = get_hex_vector(HALF_SIZE, MASK)
    
    HEX_ACTION_SIZE = len(MASK)
    DIR_ACTION_SIZE = 6
    MOVE_ACTION_SIZE = HEX_ACTION_SIZE * DIR_ACTION_SIZE

    MOVE_ACTION_SPACE = get_move_action_space(HEX_VECTOR)
    UP_ACTION_SPACE = get_up_action_space(HEX_VECTOR)
    DOWN_ACTION_SPACE = get_down_action_space(HEX_VECTOR)

    @staticmethod
    def encode_state(gamestate: Gamestate) -> List[List[List[float]]]:
        board = [
            [
                1.0 if
                Hex(row, col) in gamestate.board.board
                else 0.0
                for col in range(-Encoder.HALF_SIZE, Encoder.HALF_SIZE + 1)
            ]
            for row in range(-Encoder.HALF_SIZE, Encoder.HALF_SIZE + 1)
        ]

        own_stones = [
            [
                1.0 if
                Hex(row, col) in gamestate.own_stones
                else 0.0
                for col in range(-Encoder.HALF_SIZE, Encoder.HALF_SIZE + 1)
            ]
            for row in range(-Encoder.HALF_SIZE, Encoder.HALF_SIZE + 1)
        ]

        enemy_stones = [
            [
                1.0 if
                Hex(row, col) in gamestate.enemy_stones
                else 0.0
                for col in range(-Encoder.HALF_SIZE, Encoder.HALF_SIZE + 1)
            ]
            for row in range(-Encoder.HALF_SIZE, Encoder.HALF_SIZE + 1)
        ]

        return [board, own_stones, enemy_stones]

