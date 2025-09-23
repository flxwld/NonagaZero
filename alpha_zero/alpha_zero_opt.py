import torch
from tqdm import trange
import numpy as np
import random
import torch.nn.functional as F

from nonaga.engine import Engine
from nonaga.gamestate import Gamestate
from nonaga.encoder import Encoder
from model.model import AlphaZeroModel
from mcts.mcts_opt import MCTS_Opt
from args import Args
from model.storage import save_checkpoint, load_checkpoint
from alpha_zero.self_play_game import SelfPlayGame


class AlphaZeroOpt:
    def __init__(self, args: Args, filename: str = None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AlphaZeroModel(device=device, num_resBlocks=args.num_resBlocks, num_hidden=args.num_hidden)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.model.to(device)
        if filename is not None:
            load_checkpoint(self.model, self.optimizer, filename)
        self.mcts = MCTS_Opt(self.model, args)
        self.args = args

    def self_play(self):
        move_memory = []
        up_memory = []
        down_memory = []
        player = 1
        games = [SelfPlayGame() for _ in range(self.args.num_parallel_games)]

        for i in trange(self.args.max_self_play_steps, desc="Self-playing games"):
            if len(games) == 0:
                break
            self.mcts.search(games)

            for game in games:
                action_probs = np.zeros(Encoder.MOVE_ACTION_SIZE) if game.gamestate.next_action == Gamestate.TurnState.move else np.zeros(Encoder.HEX_ACTION_SIZE)
                for child in game.root.children:
                    action_probs[child.action_taken] = child.visits
                action_probs /= np.sum(action_probs)

                game.memory.append((game.gamestate.copy(), action_probs, player))

                temperature_action_probs = action_probs ** (1 / self.args.temperature)
                temperature_action_probs /= np.sum(temperature_action_probs)
                action = np.random.choice(game.engine.get_action_space(), p=temperature_action_probs)

                action_index = game.engine.get_action_space().index(action)
                child = next((c for c in game.root.children if c.action_taken == action_index))
                game.root = child

                priors = [c.prior for c in game.root.children]
                priors = np.array(priors)
                priors = (1 - self.args.dirichlet_epsilon) * priors + self.args.dirichlet_epsilon * np.random.dirichlet([self.args.dirichlet_alpha] * len(priors))
                priors /= np.sum(priors)
                for i, child in enumerate(game.root.children):
                    child.prior = priors[i]
                
                game.engine.make_action(action)

                reward = game.engine.get_reward()
                if reward is not None:
                    for h_gamestate, h_action_probs, h_player in game.memory:
                        outcome = 1 if h_player == player else -1
                        if h_gamestate.next_action == Gamestate.TurnState.move:
                            move_memory.append((
                                Encoder.encode_state(h_gamestate),
                                h_action_probs,
                                outcome
                            ))
                        elif h_gamestate.next_action == Gamestate.TurnState.up:
                            up_memory.append((
                                Encoder.encode_state(h_gamestate),
                                h_action_probs,
                                outcome
                            ))
                        elif h_gamestate.next_action == Gamestate.TurnState.down:
                            down_memory.append((
                                Encoder.encode_state(h_gamestate),
                                h_action_probs,
                                outcome
                            ))
                    games.remove(game)
                
                if game.engine.get_next_action() == Gamestate.TurnState.move:
                    player = -player
                    game.engine.turn_board()

        for game in games:
            for h_gamestate, h_action_probs, h_player in game.memory:
                outcome = 0
                if h_gamestate.next_action == Gamestate.TurnState.move:
                    move_memory.append((
                        Encoder.encode_state(h_gamestate),
                        h_action_probs,
                        outcome
                    ))
                elif h_gamestate.next_action == Gamestate.TurnState.up:
                    up_memory.append((
                        Encoder.encode_state(h_gamestate),
                        h_action_probs,
                        outcome
                    ))
                elif h_gamestate.next_action == Gamestate.TurnState.down:
                    down_memory.append((
                        Encoder.encode_state(h_gamestate),
                        h_action_probs,
                        outcome
                    ))
        return move_memory, up_memory, down_memory
    
    def train(self, move_memory, up_memory, down_memory):
        random.shuffle(move_memory)
        epoch_move_policy_loss = 0
        epoch_move_value_loss = 0
        for batchIdx in range(0, len(move_memory), self.args.batch_size):
            sample = move_memory[batchIdx:batchIdx+self.args.batch_size]
            game_state, policy_targets, value_targets = zip(*sample)
            
            game_state, policy_targets, value_targets = np.array(game_state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            game_state = torch.tensor(game_state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
            
            move_policy, up_policy, down_policy, move_value, up_value, down_value = self.model(game_state)
            
            out_policy = move_policy
            out_value = move_value
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_move_policy_loss += policy_loss.item() * len(sample)
            epoch_move_value_loss += value_loss.item() * len(sample)
        
        random.shuffle(up_memory)
        epoch_up_policy_loss = 0
        epoch_up_value_loss = 0
        for batchIdx in range(0, len(up_memory), self.args.batch_size):
            sample = up_memory[batchIdx:batchIdx+self.args.batch_size]
            game_state, policy_targets, value_targets = zip(*sample)
            
            game_state, policy_targets, value_targets = np.array(game_state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            game_state = torch.tensor(game_state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
            
            move_policy, up_policy, down_policy, move_value, up_value, down_value = self.model(game_state)
            
            out_policy = up_policy
            out_value = up_value
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_up_policy_loss += policy_loss.item() * len(sample)
            epoch_up_value_loss += value_loss.item() * len(sample)
        
        random.shuffle(down_memory)
        epoch_down_policy_loss = 0
        epoch_down_value_loss = 0
        for batchIdx in range(0, len(down_memory), self.args.batch_size):
            sample = down_memory[batchIdx:batchIdx+self.args.batch_size]
            game_state, policy_targets, value_targets = zip(*sample)
            
            game_state, policy_targets, value_targets = np.array(game_state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            game_state = torch.tensor(game_state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
            
            move_policy, up_policy, down_policy, move_value, up_value, down_value = self.model(game_state)
            
            out_policy = down_policy
            out_value = down_value
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_down_policy_loss += policy_loss.item() * len(sample)
            epoch_down_value_loss += value_loss.item() * len(sample)
        
        epoch_move_policy_loss /= len(move_memory)
        epoch_move_value_loss /= len(move_memory)
        epoch_up_policy_loss /= len(up_memory)
        epoch_up_value_loss /= len(up_memory)
        epoch_down_policy_loss /= len(down_memory)
        epoch_down_value_loss /= len(down_memory)
        return epoch_move_policy_loss, epoch_move_value_loss, epoch_up_policy_loss, epoch_up_value_loss, epoch_down_policy_loss, epoch_down_value_loss
        
    
    def learn_iteration(self, iteration: int):
        move_memory = []
        up_memory = []
        down_memory = []

        self.model.eval()
        for selfPlay_iteration in trange(self.args.num_selfPlay_iterations, desc="Self-playing"):
            return_move_memory, return_up_memory, return_down_memory = self.self_play()
            move_memory += return_move_memory
            up_memory += return_up_memory
            down_memory += return_down_memory
            
        self.model.train()
        for epoch in trange(self.args.num_epochs, desc="Training"):
            move_policy_loss, move_value_loss, up_policy_loss, up_value_loss, down_policy_loss, down_value_loss = self.train(move_memory, up_memory, down_memory)
            print(f"Move policy loss: {move_policy_loss}, Move value loss: {move_value_loss}")
            print(f"Up policy loss: {up_policy_loss}, Up value loss: {up_value_loss}")
            print(f"Down policy loss: {down_policy_loss}, Down value loss: {down_value_loss}")
        
        save_checkpoint(self.model, self.optimizer, iteration, f"model_{iteration}.pt")
    
    def query(self, gamestate: Gamestate):
        self.model.eval()
        with torch.no_grad():
            action_probs, root = self.mcts.search(gamestate, is_query=True)
            if gamestate.next_action == Gamestate.TurnState.move:
                action = np.argmax(action_probs)
                action = Encoder.MOVE_ACTION_SPACE[action]
                return action
            elif gamestate.next_action == Gamestate.TurnState.up:
                action = np.argmax(action_probs)
                action = Encoder.UP_ACTION_SPACE[action]
                return action
            elif gamestate.next_action == Gamestate.TurnState.down:
                action = np.argmax(action_probs)
                action = Encoder.DOWN_ACTION_SPACE[action]
                return action
