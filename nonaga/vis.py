from math import sqrt
from typing import List
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

from .engine import Engine
from .hex_unit import Hex
from .gamestate import Gamestate


class Vis:
    @staticmethod
    def draw(gamestate: Gamestate):
        fig1, ax1 = plt.subplots()

        centers = [Vis.get_centers(c) for c in gamestate.board.get_board()]
        circles = [plt.Circle((x, y), radius=sqrt(3) / 2) for x, y in centers]
        circles_coll = PatchCollection(circles, match_original=False)
        circles_coll.set_facecolor((0.82, 0.71, 0.55, 0.7))
        ax1.add_collection(circles_coll)

        own_stone_centers = [Vis.get_centers(c) for c in gamestate.own_stones]
        own_stones = [plt.Circle((x, y), radius=0.6) for x, y in own_stone_centers]
        own_stones_coll = PatchCollection(own_stones, match_original=False)
        own_stones_coll.set_facecolor((0.95, 0.95, 0.95))
        ax1.add_collection(own_stones_coll)
        
        enemy_stone_centers = [Vis.get_centers(c) for c in gamestate.enemy_stones]
        enemy_stones = [plt.Circle((x, y), radius=0.6) for x, y in enemy_stone_centers]
        enemy_stones_coll = PatchCollection(enemy_stones, match_original=False)
        enemy_stones_coll.set_facecolor((0.1, 0.1, 0.1))
        ax1.add_collection(enemy_stones_coll)

        ax1.autoscale_view()
        ax1.set_aspect('equal')
        ax1.axis('off')
        plt.show()

    def draw_with_marked_ups(gamestate: Gamestate):
        marked_ups = [up.hex for up in Engine(gamestate).get_ups()]
        fig1, ax1 = plt.subplots()
        marked_circles_coll = Vis._get_collection(marked_ups, (0.2, 0.7, 0.2, 0.85), sqrt(3) / 2)
        ax1.add_collection(marked_circles_coll)

        circles_coll = Vis._get_collection(
            [c for c in gamestate.board.get_board() if c not in marked_ups and c != gamestate.last_hex_relocated],
            (0.82, 0.71, 0.55, 0.7),
            sqrt(3) / 2
        )
        ax1.add_collection(circles_coll)

        last_hex_relocated_coll = Vis._get_collection(
            [gamestate.last_hex_relocated], 
            (0.65, 0.27, 0.17, 0.85), 
            sqrt(3) / 2
        )
        ax1.add_collection(last_hex_relocated_coll)

        own_stones_coll = Vis._get_collection(gamestate.own_stones, (0.95, 0.95, 0.95), 0.6)
        ax1.add_collection(own_stones_coll)

        enemy_stones_coll = Vis._get_collection(gamestate.enemy_stones, (0.1, 0.1, 0.1), 0.6)
        ax1.add_collection(enemy_stones_coll)

        ax1.autoscale_view()
        ax1.set_aspect('equal')
        ax1.axis('off')
        plt.show()
    

    @staticmethod
    def draw_with_marked_downs(gamestate: Gamestate):
        marked_downs = [down.hex for down in Engine(gamestate).get_downs()]
        fig1, ax1 = plt.subplots()
        marked_circles_coll = Vis._get_collection(marked_downs, (0.2, 0.7, 0.2, 0.85), sqrt(3) / 2)
        ax1.add_collection(marked_circles_coll)

        circles_coll = Vis._get_collection([c for c in gamestate.board.get_board() if c not in marked_downs], (0.82, 0.71, 0.55, 0.7), sqrt(3) / 2)
        ax1.add_collection(circles_coll)

        own_stones_coll = Vis._get_collection(gamestate.own_stones, (0.95, 0.95, 0.95), 0.6)
        ax1.add_collection(own_stones_coll)

        enemy_stones_coll = Vis._get_collection(gamestate.enemy_stones, (0.1, 0.1, 0.1), 0.6)
        ax1.add_collection(enemy_stones_coll)

        ax1.autoscale_view()
        ax1.set_aspect('equal')
        ax1.axis('off')
        plt.show()
    
    @staticmethod
    def _get_collection(hexes: List[Hex], color: tuple[float, float, float, float], radius: float) -> PatchCollection:
        centers = [Vis.get_centers(c) for c in hexes]
        circles = [plt.Circle((x, y), radius=radius) for x, y in centers]
        circles_coll = PatchCollection(circles, match_original=False)
        circles_coll.set_facecolor(color)
        circles_coll.set_edgecolor((0.4, 0.26, 0.13, 1.0))
        circles_coll.set_linewidth(2)
        return circles_coll

    
    @staticmethod
    def get_centers(h: Hex) -> List[tuple[float, float]]:
        return sqrt(3) * (h.q + h.r/2.0), 1.5 * h.r