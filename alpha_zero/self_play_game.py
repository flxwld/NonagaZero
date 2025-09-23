from nonaga.engine import Engine

class SelfPlayGame:
    def __init__(self):
        self.engine = Engine()
        self.gamestate = self.engine.gamestate
        self.memory = []
        self.root = None
        self.expandable_node = None