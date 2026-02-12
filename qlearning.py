
import numpy as np              
import matplotlib.pyplot as plt 
import random                  

class GridEnvironment:
    def __init__(self, grid, start, goal, noise=0.0):
        self.grid = grid              # Carte 2D : 0 = libre, 1 = obstacle
        self.start = start            # Position départ
        self.goal = goal              # Position goal
        self.noise = noise            # Niveau de stochasticité
        self.state = start            # État actuel du véhicule
        self.height, self.width = grid.shape  # Dimensions de la grille
        # Actions possibles : droite, gauche, bas, haut
        self.actions = [(0,1), (0,-1), (1,0), (-1,0)]

    def reset(self):
        self.state = self.start       # Remise à zéro
        return self.state

    def step(self, action):
        # Si bruit aléatoire → changer action
        if random.random() < self.noise:
            action = random.randint(0,3)

        move = self.actions[action]    # Déplacement selon action

        # Nouvelle position
        new_y = self.state[0] + move[0]
        new_x = self.state[1] + move[1]

        # Limites de la grille
        if new_y < 0 or new_y >= self.height or new_x < 0 or new_x >= self.width:
            new_y, new_x = self.state

        # Obstacles
        if self.grid[new_y, new_x] == 1:
            new_y, new_x = self.state

        self.state = (new_y, new_x)

        # Récompense : +100 si goal, -1 sinon
        if self.state == self.goal:
            return self.state, 100, True
        else:
            return self.state, -1, False


# CLASSE AGENT Q-LEARNING
class QLearningAgent:
    def __init__(self, height, width, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.alpha = alpha        # Taux apprentissage
        self.gamma = gamma        # Facteur de réduction
        self.epsilon = epsilon    # Taux exploration : probabilité de choisir une action aléatoire
        self.Q = np.zeros((height, width, n_actions))  # Initialisation de la table Q à zéro : dimensions = (hauteur, largeur, nombre d'actions)
# Choix de l'action à effectuer depuis l'état actuel
    def choose_action(self, state): # On récupère les coordonnées de l'état actuel
        y, x = state
        # Exploration aléatoire : choisir une action au hasard avec probabilité epsilon
        if random.random() < self.epsilon:
            return random.randint(0,3) # Choix d'une action aléatoire (0 à 3)
        # Exploitation : meilleure action
        else:
            return np.argmax(self.Q[y, x]) # Renvoie l'indice de l'action avec la plus grande valeur Q
# Mise à jour de la table Q après avoir pris une action et reçu une récompense
    def update(self, state, action, reward, next_state):
        y, x = state # Coordonnées de l'état actuel
        ny, nx = next_state  # Coordonnées de l'état suivant
        best_next = np.max(self.Q[ny, nx])  # Meilleure valeur future
        # Formule Q-learning :  Valeur Q actuelle multiplié par le taux d'apprentissage, plus la valeur actualisée de l'état suivant ,moins la valeur Q actuelle
        self.Q[y, x, action] = self.Q[y, x, action] + self.alpha * (reward + self.gamma * best_next - self.Q[y, x, action])


#  ENTRAÎNEMENT 


def train(env, agent, episodes=500):
    # Boucle principale sur tous les épisodes d'entraînement
    for ep in range(episodes):
        state = env.reset() # Réinitialise l'environnement au début de l'épisode
        done = False # Indique si l'épisode est terminé (objectif atteint)
        steps = 0  # Compteur de pas
        # Boucle pour chaque étape de l'épisode
        while not done and steps < 500:
            action = agent.choose_action(state) # Choisir une action
            next_state, reward, done = env.step(action) # Exécuter l'action et recevoir le nouvel état, la récompense et le done
            agent.update(state, action, reward, next_state) # Mettre à jour la table Q avec la formule du Q-learning
            state = next_state # Passer à l'état suivant pour la prochaine étape
            steps += 1 # Incrémenter le compteur de pas


#  VISUALISATION DE LA TRAJECTOIRE


def visualize_policy(env, agent, start, goal):
    state = env.reset()# On réinitialise l'environnement et on place le véhicule au départ
    done = False
    trajectory = [state]   # Liste qui va stocker toutes les positions traversées par le véhicule
    while not done and len(trajectory) < 500: # On continue tant que le goal n'est pas atteint et que la trajectoire ne dépasse pas 500 pas
        y, x = state # On récupère les coordonnées actuelles du véhicule
        action = np.argmax(agent.Q[y, x])  # Meilleure action connue dans la table Q pour cet état
        next_state, reward, done = env.step(action)  # On applique l'action et on récupère le nouvel état, la récompense et si on a atteint l'objectif
        trajectory.append(next_state)  # On ajoute la nouvelle position à la trajectoire
        state = next_state # On met à jour l'état courant pour l'itération suivante

    # Affichage
    plt.figure("Q-learning 2D")
    plt.imshow(env.grid, cmap='binary')  # Carte
    ys = [p[0] for p in trajectory]
    xs = [p[1] for p in trajectory]
    plt.plot(xs, ys, 'r-', linewidth=2)    # Trajectoire en rouge
    plt.plot(start[1], start[0], 'bo', label="Start")  # Départ bleu
    plt.plot(goal[1], goal[0], 'go', label="Goal")    # Goal vert
    plt.legend()
    plt.title("Trajectoire apprise par Q-learning")
    plt.show()


#  PROGRAMME PRINCIPAL

if __name__ == "__main__":

    # Petite grille pour test
    grid = np.zeros((20,20))    # 0 = libre, 1 = obstacle
    # Ajout d'obstacles pour tester
    grid[5:7, 5:15] = 1
    grid[12:14, 3:17] = 1

    start = (0,0)   # Départ (y,x)
    goal = (19,19)  # Goal (y,x)

    env = GridEnvironment(grid, start, goal, noise=0.0)
    agent = QLearningAgent(height=grid.shape[0], width=grid.shape[1], n_actions=4,
                           alpha=0.1, gamma=0.95, epsilon=0.1)

    train(env, agent, episodes=1000)       # Entraînement
    visualize_policy(env, agent, start, goal)  # Affichage
