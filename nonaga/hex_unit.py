from typing import List


class Hex:
    def __init__(self, q: int, r: int):
        self.q = q
        self.r = r
        self.s = -q - r
    
    def get_coords(self) -> tuple[int, int, int]:
        return self.q, self.r, self.s
    
    def __hash__(self) -> int:
        return hash((self.q, self.r, self.s))
    
    def __eq__(self, other) -> bool:
        return self.q == other.q and self.r == other.r
    
    def __add__(self, other) -> 'Hex':
        return Hex(self.q + other.q, self.r + other.r)
    
    def __sub__(self, other: 'Hex') -> 'Hex':
        return Hex(self.q - other.q, self.r - other.r)
    
    def __str__(self) -> str:
        return f"Hex({self.q}, {self.r})"
    
    def get_neighbor_in_dir(self, dir: int) -> 'Hex':
        dirs = [
            Hex(self.q + 1, self.r),
            Hex(self.q, self.r + 1),
            Hex(self.q - 1, self.r + 1),
            Hex(self.q - 1, self.r),
            Hex(self.q, self.r - 1),
            Hex(self.q + 1, self.r - 1),
        ]
        return Hex(dirs[dir].q, dirs[dir].r)
    
    def neighbors(self) -> List['Hex']:
        return [ self.get_neighbor_in_dir(dir) for dir in range(6) ]