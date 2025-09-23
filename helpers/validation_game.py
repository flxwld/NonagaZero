from nonaga.engine import Engine
from alpha_zero.alpha_zero_opt import AlphaZeroOpt

def validation_game(model1: AlphaZeroOpt, model2: AlphaZeroOpt):
    engine = Engine()
    models = [model1, model2]

    player = 1
    while not engine.get_reward():
        for i in range(3):
            model_action = models[player].query(engine.gamestate)
            engine.make_action(model_action)
        engine.turn_board()
        player = -player

    return -engine.get_reward()

def validation_game_with_history(model1: AlphaZeroOpt, model2: AlphaZeroOpt):
    history = []
    engine = Engine()
    models = [model1, model2]

    player = 1
    while not engine.get_reward():
        for i in range(3):
            model_action = models[player].query(engine.gamestate)
            engine.make_action(model_action)
            history.append(engine.gamestate.copy())
        engine.turn_board()
        player = -player

    return -engine.get_reward(), history

def validate(model1: AlphaZeroOpt, model2: AlphaZeroOpt, games: int):
    wins, draws, losses = 0, 0, 0
    for v in range(games):
        result = validation_game(model1, model2)
        if result == 1:
            wins += 1
        elif result == 0:
            draws += 1
        else:
            losses += 1
    for v in range(games):
        result = validation_game(model2, model1)
        if result == 1:
            losses += 1
        elif result == 0:
            draws += 1
        else:
            wins += 1
    return wins, draws, losses