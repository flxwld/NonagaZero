import torch
from tqdm import trange
import numpy as np
import random
import torch.nn.functional as F

from nonaga.engine import Engine
from nonaga.gamestate import Gamestate
from nonaga.encoder import Encoder
from model.model import AlphaZeroModel
from mcts.mcts import MCTS
from args import Args
from model.storage import save_checkpoint, load_checkpoint


class AlphaZero:
    def __init__(self, args: Args, filename: str = None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AlphaZeroModel(device=device, num_resBlocks=args.num_resBlocks, num_hidden=args.num_hidden)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.model.to(device)
        if filename is not None:
            load_checkpoint(self.model, self.optimizer, filename)
        self.mcts = MCTS(self.model, args)
        self.args = args

    def self_play(self):
        memory = []
        player = 1
        engine = Engine()
        
        for i in range(self.args.max_self_play_steps):
            action_probs, root = self.mcts.search(engine.gamestate)

            memory.append((engine.gamestate.copy(), action_probs, player))

            temperature_action_probs = action_probs ** (1 / self.args.temperature)
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(engine.get_action_space(), p=temperature_action_probs)

            engine.make_action(action)

            reward = engine.get_reward()

            if reward is not None:
                return_move_memory = []
                return_up_memory = []
                return_down_memory = []
                for h_gamestate, h_action_probs, h_player in memory:
                    outcome = 1 if h_player == player else -1
                    if h_gamestate.next_action == Gamestate.TurnState.move:
                        return_move_memory.append((
                            Encoder.encode_state(h_gamestate),
                            h_action_probs,
                            outcome
                        ))
                    elif h_gamestate.next_action == Gamestate.TurnState.up:
                        return_up_memory.append((
                            Encoder.encode_state(h_gamestate),
                            h_action_probs,
                            outcome
                        ))
                    elif h_gamestate.next_action == Gamestate.TurnState.down:
                        return_down_memory.append((
                            Encoder.encode_state(h_gamestate),
                            h_action_probs,
                            outcome
                        ))
                return return_move_memory, return_up_memory, return_down_memory
            
            if engine.get_next_action() == Gamestate.TurnState.move:
                player = -player
                engine.turn_board()
        
        return_move_memory = []
        return_up_memory = []
        return_down_memory = []
        for h_gamestate, h_action_probs, h_player in memory:
            outcome = 0
            if h_gamestate.next_action == Gamestate.TurnState.move:
                return_move_memory.append((
                    Encoder.encode_state(h_gamestate),
                    h_action_probs,
                    outcome
                ))
            elif h_gamestate.next_action == Gamestate.TurnState.up:
                return_up_memory.append((
                    Encoder.encode_state(h_gamestate),
                    h_action_probs,
                    outcome
                ))
            elif h_gamestate.next_action == Gamestate.TurnState.down:
                return_down_memory.append((
                    Encoder.encode_state(h_gamestate),
                    h_action_probs,
                    outcome
                ))
        return return_move_memory, return_up_memory, return_down_memory
    
    def train(self, move_memory, up_memory, down_memory):
        random.shuffle(move_memory)
        random.shuffle(up_memory)
        random.shuffle(down_memory)
        for batchIdx in trange(0, len(move_memory), self.args.batch_size, desc="Training on move memory"):
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
        
        for batchIdx in trange(0, len(up_memory), self.args.batch_size, desc="Training on up memory"):
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
        
        for batchIdx in trange(0, len(down_memory), self.args.batch_size, desc="Training on down memory"):
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

    
    def learn(self):
        for iteration in range(self.args.num_iterations):
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
                self.train(move_memory, up_memory, down_memory)
            
            save_checkpoint(self.model, self.optimizer, iteration, f"model_{iteration}.pt")
    
    def query(self, gamestate: Gamestate):
        self.model.eval()
        with torch.no_grad():
            action_probs, root = self.mcts.search(gamestate)
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
