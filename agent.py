import random
import numpy as np
class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.actions = [0, 1, 2, 3]
    def get_q(self, state, action):
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0
        return self.q_table[(state, action)]
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = [
            self.get_q(state, a)
            for a in self.actions
        ]
        return self.actions[np.argmax(q_values)]
    def update(self, state, action,
               reward, next_state):
        current_q = self.get_q(state, action)
        max_future_q = max([
            self.get_q(next_state, a)
            for a in self.actions
        ])
        new_q = current_q + self.alpha * (
            reward +
            self.gamma * max_future_q -
            current_q
        )
        self.q_table[(state, action)] = new_q
    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )