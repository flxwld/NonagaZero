from typing import List, Optional

from .gamestate import Gamestate
from .actions import MoveAction, UpAction, DownAction
from .hex_unit import Hex
from .encoder import Encoder

class Engine:
    def __init__(self, gamestate: Optional[Gamestate] = None):
        self.gamestate = gamestate if gamestate is not None else Gamestate()
    
    def reset(self):
        self.gamestate = Gamestate()
    
    def is_terminal(self) -> bool:
        return self.get_reward() is not None
    
    def get_action_space(self):
        if self.gamestate.next_action == Gamestate.TurnState.move:
            return Encoder.MOVE_ACTION_SPACE
        elif self.gamestate.next_action == Gamestate.TurnState.up:
            return Encoder.UP_ACTION_SPACE
        elif self.gamestate.next_action == Gamestate.TurnState.down:
            return Encoder.DOWN_ACTION_SPACE
    
    def make_action(self, action: MoveAction | UpAction | DownAction):
        if self.gamestate.next_action == Gamestate.TurnState.move:
            self.make_move(action)
        elif self.gamestate.next_action == Gamestate.TurnState.up:
            self.make_up(action)
        elif self.gamestate.next_action == Gamestate.TurnState.down:
            self.make_down(action)
    
    def get_next_action(self) -> Gamestate.TurnState:
        return self.gamestate.next_action
    
    def get_gamestate(self) -> Gamestate:
        return self.gamestate
    
    def turn_board(self):
        self.gamestate.own_stones, self.gamestate.enemy_stones = self.gamestate.enemy_stones, self.gamestate.own_stones
    
    def get_reward(self) -> int | None:
        if self._are_connected(self.gamestate.own_stones):
            return 1
        if self._are_connected(self.gamestate.enemy_stones):
            return -1
        return None
    
    def _are_connected(self, stones: List[Hex]) -> bool:
        connections = [
            1 if stones[0] in stones[1].neighbors() else 0,
            1 if stones[1] in stones[2].neighbors() else 0,
            1 if stones[2] in stones[0].neighbors() else 0,
        ]
        return sum(connections) >= 2
    
    def get_moves(self) -> List[MoveAction]:
        moves = []
        for stone in self.gamestate.own_stones:
            for dir in self._get_moves_for_stone(stone):
                moves.append(MoveAction(stone, dir))
        return moves
    
    def make_move(self, move: MoveAction):
        assert self.gamestate.next_action == Gamestate.TurnState.move
        assert move in self.get_moves()
        self.gamestate.own_stones.remove(move.hex)
        self.gamestate.own_stones.append(self._get_move_stone_in_dir(move.hex, move.dir))
        self.gamestate.next_action = Gamestate.TurnState.up

    def _get_moves_for_stone(self, stone: Hex) -> List[int]:
        assert stone in self.gamestate.own_stones
        return [
            dir for dir in range(6) if 
            stone.get_neighbor_in_dir(dir) in self.gamestate.board.get_board() and
            stone.get_neighbor_in_dir(dir) not in self.gamestate.own_stones and
            stone.get_neighbor_in_dir(dir) not in self.gamestate.enemy_stones
        ]
    
    def _get_move_stone_in_dir(self, stone: Hex, dir: int) -> Hex:
        while (
            stone.get_neighbor_in_dir(dir) in self.gamestate.board.get_board() and
            stone.get_neighbor_in_dir(dir) not in self.gamestate.own_stones and
            stone.get_neighbor_in_dir(dir) not in self.gamestate.enemy_stones
        ):
            stone = stone.get_neighbor_in_dir(dir)
        return stone
    
    def get_ups(self) -> List[UpAction]:
        return [
            UpAction(h)
            for h in self.gamestate.board.board
            if
            self._has_two_adjacent_neighbors_not_in_the_board(h) and
            self._does_not_separate_the_board_into_two_parts(h) and
            h not in self.gamestate.own_stones and
            h not in self.gamestate.enemy_stones and
            h is not self.gamestate.last_hex_relocated
        ]

    def make_up(self, up: UpAction):
        assert self.gamestate.next_action == Gamestate.TurnState.up or (self.gamestate.next_action == Gamestate.TurnState.move and len(self.get_moves()) == 0)
        assert up in self.get_ups()
        self.gamestate.board.board.remove(up.hex)
        self.gamestate.hex_up = up.hex
        self.gamestate.next_action = Gamestate.TurnState.down
    
    def _has_two_adjacent_neighbors_not_in_the_board(self, h: Hex) -> bool:
        missing_neighbor_indices =[i for i, neighbor in enumerate(h.neighbors()) if not neighbor in self.gamestate.board.board]
        for i in range(len(missing_neighbor_indices)):
            if (missing_neighbor_indices[i] + 1) % 6 == missing_neighbor_indices[(i+1) % len(missing_neighbor_indices)]:
                return True
        return False
    
    def _has_two_adjacent_neighbors_in_the_board(self, h: Hex) -> bool:
        neighbor_indices = [i for i, neighbor in enumerate(h.neighbors()) if neighbor in self.gamestate.board.board]
        for i in range(len(neighbor_indices)):
            if (neighbor_indices[i] + 1) % 6 == neighbor_indices[(i+1) % len(neighbor_indices)]:
                return True
        return False
    
    def _does_not_separate_the_board_into_two_parts(self, start_h: Hex) -> bool:
        if self._has_two_adjacent_neighbors_in_the_board(start_h): return True
        queue = [n for n in start_h.neighbors() if n in self.gamestate.board.board][:1]
        visited = set()
        while queue:
            h = queue.pop()
            visited.add(h)
            for n in h.neighbors():
                if (
                n in self.gamestate.board.board and 
                n not in visited and
                n not in queue and
                n != start_h
                ):
                    queue.append(n)
        return len(visited) == len(self.gamestate.board.board) - 1
    
    def get_downs(self) -> List[DownAction]:
        candidates = set()
        for h in self.gamestate.board.board:
            for n in h.neighbors():
                if n not in self.gamestate.board.board and n != self.gamestate.hex_up:
                    candidates.add(n)
        candidates = list(candidates)
        return [
            DownAction(h)
            for h in candidates
            if
            self._count_neighbors_in_the_board(h) >= 2
        ]
    
    def _count_neighbors_in_the_board(self, h: Hex) -> int:
        return sum([1 if n in self.gamestate.board.board else 0 for n in h.neighbors()])

    def make_down(self, down: DownAction):
        assert self.gamestate.next_action == Gamestate.TurnState.down
        assert down in self.get_downs()
        self.gamestate.board.board.append(down.hex)
        self.gamestate.hex_up = None
        self.gamestate.last_hex_relocated = down.hex
        self.gamestate.next_action = Gamestate.TurnState.move
    

