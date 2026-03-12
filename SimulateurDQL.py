import torch
import torch.nn as nn
import torch.optim as optim
import random
from env_dql import EnvironnementRobot
from collections import deque
import pygame
import math
import numpy as np

# reseau de neuronne : 
class DQN(nn.Module):
    # couche d'entree 4 neruonnes car on a 4 valeurs dans l'etat x, y, angle, vitesse
    # couche cachee : 64 neurones puis 64 neurones
    # couche de sortie : 5 neurones une pour chaque action possible
    def __init__(self, taille_etat, nb_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(taille_etat, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, nb_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# init
env = EnvironnementRobot()
taille_etat = 4  # x, y, angle, vitesse
nb_actions = 5   # accelerer, freiner, gauche, droite, rien
memoire = deque(maxlen=10000)

reseau = DQN(taille_etat, nb_actions)
optimizer = optim.Adam(reseau.parameters(), lr=0.001)

epsilon = 1.0        # 100% de hasard au debut car il connait rien donc une action au pif est choisie
epsilon_min = 0.01   # on gardera toujours 1% de hasard pour qu'il cherche toujours de nouvelles choses et ne se bloque pas 
epsilon_decay = 0.995 # a la fin de chaque partie on multiplie epsilon pour le reduire et le rendre moin random


nb_episodes = 500 # le robot va essayer 500 fois 

for episode in range(nb_episodes):
    etat = env.reset() # on replace le robot au depart
    done = False # fin
    score_episode = 0
    
    while not done:
        # dhoisir une action qu debut qu hasard par la suite grace au reseau
        if random.random() < epsilon:
            action = random.randint(0, nb_actions - 1) # action au hasard
        else:
            with torch.no_grad():
                etat_tensor = torch.FloatTensor(etat).unsqueeze(0) # on transforme l'etat en tensor pour le passer dans le reseau
                q_values = reseau(etat_tensor) # on recupere les q_values pour chaque action
                action = torch.argmax(q_values).item() # on choisit l'action avec la plus grande q_value
        
        # on envoie l'action a l'env
        nouvel_etat, reward, done = env.step(action)

        # on affiche le robot
        env.dessiner()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        # on sauvegarde l'xp en mémoire pour pas que le reseau oublie ce qu'il a appris dans les episode d'avt
        memoire.append((etat, action, reward, nouvel_etat, done))
        
        # on entraine le reseau 
        # Q cible ​= R + gamma * maxQ(s′,a′)
        # On ne s'entraîne QUE si on a au moins 32 souvenirs en mémoire
        if len(memoire) > 32:
            # on pioche 32 souvenirs au hasard
            batch = random.sample(memoire, 32)
            optimizer.zero_grad()
            
            # 3. on va calculer l'erreur pour chaque souvenir du lot 
            for etat_b, action_b, reward_b, nouvel_etat_b, done_b in batch:
                
                # on transforme l'etat_b en tenseur et on calcule la prediction
                prediction = torch.FloatTensor(etat_b).unsqueeze(0) 
                prediction = reseau(prediction)[0][action_b] # on recupere la q_value de l'action choisie dans le souvenir
                
                # on calcule la cible avec la formule 
                if done_b:
                    cible = torch.tensor(reward_b, dtype=torch.float32)
                else:
                    # on calcule la Q-valeur max du nouvel_etat_b avec le reseau et on applique la formule : 
                    # R + gamma * maxQ(s', a')
                    cible = torch.tensor(reward_b, dtype=torch.float32) + 0.99 * torch.max(reseau(torch.FloatTensor(nouvel_etat_b).unsqueeze(0))).item()                
                    
                # on calcule l'erreur entre la prediction et la cible
                loss = nn.MSELoss()(prediction, cible)
                
                # on accumule les gradients 
                loss.backward()
                
            # 5. on MAJ les poids du cerveau une fois que le lot est fini
            optimizer.step()
        
        # on passe à l'etat suivant
        etat = nouvel_etat
        score_episode += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
    print(f"Épisode {episode} terminé avec un score de {score_episode}")

print("Entraînement terminé !")