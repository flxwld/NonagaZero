from dataclasses import dataclass

from .hex_unit import Hex


@dataclass(frozen=True)
class MoveAction:
    hex: Hex
    dir: int

    def __str__(self) -> str:
        return f"MoveAction(hex={self.hex}, dir={self.dir})"

    def __eq__(self, other: 'MoveAction') -> bool:
        return self.hex == other.hex and self.dir == other.dir

@dataclass(frozen=True)
class UpAction:
    hex: Hex

    def __str__(self) -> str:
        return f"UpAction(hex={self.hex})"
    
    def __eq__(self, other: 'UpAction') -> bool:
        return self.hex == other.hex

@dataclass(frozen=True)
class DownAction:
    hex: Hex

    def __str__(self) -> str:
        return f"DownAction(hex={self.hex})"

    def __eq__(self, other: 'DownAction') -> bool:
        return self.hex == other.hex