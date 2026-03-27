import torch
import torch.nn as nn
import torch.optim as optim
import random
from env_dql import EnvironnementRobot
from collections import deque
import pygame
import math
import numpy as np
import sys 
import os
from RRT import RRT

# on cree un reseau de neurones de 3 couches 
# la premiere recoit les 8 valeurs de l'etat et la derniere envoie une action parmi les 5  

class DQN(nn.Module):
    def __init__(self, taille_etat, nb_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(taille_etat, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, nb_actions)
    # permet d activer certain neuronne et mettre a 0 les valaeur negative
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

# on configure la carte depart, arrivee, obstacle
depart_robot = (50, 50)
arrivee_finale = (700, 500)
liste_obstacles = [
    pygame.Rect(400, 200, 50, 200),
    pygame.Rect(450, 350, 150, 50),
]

# on lance RRT pour trouver un chemin entre le depart et l arrivee en evitant les obstacles
solveur_rrt = RRT(depart_robot, arrivee_finale, 2000, liste_obstacles, 30, None)
chemin_trouve, _ = solveur_rrt.compute_path()

# si RRT n a pas trouve de chemin, on met juste l arrivee pour que le robot puisse quand meme apprendre a y aller
if chemin_trouve is None or len(chemin_trouve) == 0:
    chemin_trouve = [arrivee_finale]
# sinon on enleve le point de depart du chemin trouve pour ne pas le donner en double a l environnement
else:
    chemin_trouve.pop(0)

# on initialise l environnement et le reseau de neurones
env = EnvironnementRobot(depart=depart_robot, waypoints=chemin_trouve)
taille_etat = 8  # 8 entree x, y, direction cos/sin, vitesse, distance cible, angle cible (cos/sin)
nb_actions = 5   
memoire = deque(maxlen=20000) # memoire on garde les derniere 20K actions pour l entrainement

# Double DQN car DQN simple n apprend rien et part juste dans les murs
# on cree un premier reseau qui va apprendre qui est donc le joueur et 
# un deuxieme qui va servir de d'arbitre donc le reseau cible et il va juste copier le cervrau du premier puis il juge les actions 
# du premier pour lui donner une meilleure estimation de la valeur de chaque action
reseau = DQN(taille_etat, nb_actions)
reseau_cible = DQN(taille_etat, nb_actions)
reseau_cible.load_state_dict(reseau.state_dict())
reseau_cible.eval() # on gele le reseau cible il ne s'entraine pas il sert juste de reference 

if os.path.exists("meilleur_modele_dqn.pth"):
    reseau.load_state_dict(torch.load("meilleur_modele_dqn.pth", weights_only=True))
    reseau_cible.load_state_dict(reseau.state_dict())
    print("modele charge")
else:
    print("pas de modele")

reseau.train()
optimizer = optim.Adam(reseau.parameters(), lr=0.001)

epsilon = 1.0        
epsilon_min = 0.05  
epsilon_decay = 0.995 

nb_episodes = 400 
# on garde en memoire le meilleur score pour sauvegarder le modele qui a le mieux performe
meilleur_score = -float('inf') 

# boucle d entrainement sur les episode
for episode in range(nb_episodes):
    # on reset l environnement a chaque episode pour repartir du debut
    etat = env.reset() 
    # on initialise done a False pour la boucle de jeu et le score de l episode a 0 et un compteur de pas pour eviter les episode infini
    done = False 
    score_episode = 0
    compteur_pas = 0 

    while not done:
        compteur_pas += 1
        if compteur_pas > 800: 
            done = True

        if random.random() < epsilon:
            action = random.randint(0, nb_actions - 1) 
        else:
            with torch.no_grad():
                # on convertit l etat en tensor pour le passer dans le reseau
                etat_tensor = torch.FloatTensor(etat).unsqueeze(0) 
                # on recupere les q_values pour chaque action possible et on prend celle qui a la plus haute valeur
                q_values = reseau(etat_tensor) 
                action = torch.argmax(q_values).item() 
        
        # le robot agit dans l environnement avec l action choisie et 
        # on recupere le nouvel etat, la recompense et si l episode est termine
        nouvel_etat, reward, done = env.step(action)
        memoire.append((etat, action, reward, nouvel_etat, done))
        
        # on entraine le reseau que si on a assez de memoire pour faire un batch d entrainement
        if len(memoire) > 128:
            # on prend un echantillon aleatoire de 128 transitions de notre memoire pour l entrainement
            batch = random.sample(memoire, 128)
            # on convertit les listes de transitions en tensors pour le calcul
            etats_batch = torch.FloatTensor(np.array([m[0] for m in batch])) 
            actions_batch = torch.LongTensor([[m[1]] for m in batch])        
            rewards_batch = torch.FloatTensor([m[2] for m in batch])         
            nouvel_etats_batch = torch.FloatTensor(np.array([m[3] for m in batch])) 
            dones_batch = torch.FloatTensor([m[4] for m in batch])    

            # on recupere les q_values pour les actions prise dans le batch
            # on utilise gather pour prendre la q_value correspondant a l action reel prise dans le batch 
            # si il calcule q = action 2 vaut 2 mais que le batch il avait pris l'action 0 qlors il va qd mm prendre l action 0 
            q_valeurs = reseau(etats_batch).gather(1, actions_batch).squeeze(1)
            
            # application du double DQN / calcul mathematique pas de modification direct des neurones cibles
            with torch.no_grad(): 
                # on prend les actions qui ont la plus haute q_value pour le nouvel etat avec le reseau principal
                meill_actions = reseau(nouvel_etats_batch).argmax(1, keepdim=True)
                # on prend les q_values de ces actions avec le reseau cible pour calculer la cible de l entrainement
                # on force de prendre la valeur q du rseau cible pour l action quil a projete avec le nouvelle etat
                max_q_suivantes = reseau_cible(nouvel_etats_batch).gather(1, meill_actions).squeeze(1)

            # on calcule donc la prochain reward reel avec le jugement du reseau cible     
            # les cibles ici sont la recompense plus la valeur de la meilleure action suivante 
            # on multiplie par (1 - done) pour que si l episode est termine on ne prend pas en compte la valeur de la prochaine action 
            cibles = rewards_batch + 0.99 * max_q_suivantes * (1 - dones_batch)
            
            # SmoothL1Loss pour la stabilité au lieu de MSELoss qui est pas assez bon
            # puis on calcul l'erreur entre les q_values predites par le reseau pour les actions prises et les cibles calculees avec le reseau cible
            loss = nn.SmoothL1Loss()(q_valeurs, cibles)
            
            # on fait une passe de retropropagation pour entrainer le reseau
            optimizer.zero_grad() 
            loss.backward()       
            # limitateur de vitesse (du gradient) pour pas qu'il oublie tout ce qu'il a appris a cause d'une grosse erreur
            nn.utils.clip_grad_norm_(reseau.parameters(), 1.0) 
            optimizer.step()      
        
        # on met a jour l etat actuel pour le prochain pas de temps
        etat = nouvel_etat
        # on ajoute la recompense de ce pas de temps au score total de l episode
        score_episode += reward

    # on reduit epsilon pour que le robot explore moins au fil du temps et exploite plus ce qu il a appris
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
    print(f"episode {episode} termine avec un score de {score_episode}")

    # on sauvegarde le modele si on a battu le meilleur score
    if score_episode > meilleur_score:
        meilleur_score = score_episode
        torch.save(reseau.state_dict(), "meilleur_modele_dqn.pth")
        print(f"meilleur modele score: {meilleur_score:.2f}")

    # MAJ du reseau cible toutes les 50 episodes pour que l' arbitre soit a jour avec le joueur
    if episode % 50 == 0:
        reseau_cible.load_state_dict(reseau.state_dict())

print("entreinement termine")
reseau.load_state_dict(torch.load("meilleur_modele_dqn.pth", weights_only=True))

epsilon = 0.0 
# on met le reseau en mode evaluation pour qu'il ne calcule pas les gradients et qu'il puisse juste faire des predictions
reseau.eval()

while True:
    etat = env.reset()
    done = False
    
    while not done:
        with torch.no_grad():
            etat_tensor = torch.FloatTensor(etat).unsqueeze(0)
            q_values = reseau(etat_tensor)
            action = torch.argmax(q_values).item()
            
        nouvel_etat, reward, done = env.step(action)
        env.dessiner()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        etat = nouvel_etat