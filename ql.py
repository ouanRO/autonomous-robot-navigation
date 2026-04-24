import numpy as np
import os

class QLearningAgent:
    def __init__(self):
        self.n_dist = 20
        self.n_angle = 36  # segments de 10 degrés
        self.n_speed = 6
        self.n_actions = 4

        self.alpha = 0.5
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995

        self.Q = np.zeros((self.n_dist, self.n_angle, self.n_speed, self.n_actions))

    def discretize(self, dist, angle, speed):
        # Conversion de la distance (paliers de 40px)
        d = min(int(dist / 20), self.n_dist - 1)
        # Conversion de l'angle (-180/180 -> 0/35)
        a = int((angle + 180) / 10) 
        a = max(0, min(a, self.n_angle - 1))
        # Vitesse
        s = min(int(abs(speed)), self.n_speed - 1)
        return (d, a, s)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        if np.all(self.Q[state] == 0):
            return np.random.randint(0, self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (reward + self.gamma * best_next - self.Q[state][action])
        # Décroissance automatique de l'exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        np.save(filename, self.Q)
        print(f"> Table sauvegardée sous : {filename}")

    def load(self, filename):
        if os.path.exists(filename):
            self.Q = np.load(filename)
            print(f"> Table {filename} chargée avec succès.")
            return True
        return False











