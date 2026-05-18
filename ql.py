
import numpy as np
import os


class QLearningAgent:
    def __init__(self):
        self.n_dist = 20
        self.n_angle = 36  # segments de 10 degrés
        self.n_speed = 4
        self.n_actions = 4

        #  ALPHA VARIABLE 
        self.alpha_min = 0.1
        self.alpha_max = 0.9
        self.N = 10000  # nombre d'étapes pour montée de alpha
        self.step_count = 0

        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995

        self.Q = np.zeros(
            (self.n_dist, self.n_angle, self.n_speed, self.n_actions)
        )


    # DISCRETISATION

    def discretize(self, dist, angle, speed):
        d = min(int(dist / 30), self.n_dist - 1)

        a = int((angle + 180) / 10) % 360
        
        a = max(0, min(a, self.n_angle - 1))

        s = min(int(abs(speed)), self.n_speed - 1)

        return (d, a, s)


    # CHOIX ACTION

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)

        if np.all(self.Q[state] == 0):
            return np.random.randint(0, self.n_actions)

        return np.argmax(self.Q[state])


    #  ALPHA VARIABLE

    def get_alpha(self):
        if self.step_count < self.N:
            return self.alpha_min + (self.step_count / self.N) * (self.alpha_max - self.alpha_min)
        else:
            return self.alpha_min


    # UPDATE Q-LEARNING

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.Q[next_state])

        alpha = self.get_alpha()

        self.Q[state][action] += alpha * (
            reward + self.gamma * best_next - self.Q[state][action]
        )

        self.step_count += 1

        # décroissance epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # SAVE / LOAD

    def save(self, filename):
        np.save(filename, self.Q)
        print(f"> Table sauvegardée sous : {filename}")

    def load(self, filename):
        if os.path.exists(filename):
            self.Q = np.load(filename)
            print(f"> Table {filename} chargée avec succès.")
            return True
        return False






# ce que c'est la q valeure et ce que elle represente 
#le principe du  renforcement par apprentissage pour le ql
# explique le principe en generale


