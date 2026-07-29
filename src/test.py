import torch
import torch.nn as nn
import pygame
import os
import sys
import random
from env_dql import EnvironnementRobot
from RRT import RRT

class DQN(nn.Module):
    def __init__(self, taille_etat, nb_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(taille_etat, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, nb_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

depart_robot = (50, 50)
arrivee_finale = (700, 500)

cartes = [
    [pygame.Rect(400, 200, 50, 200), pygame.Rect(450, 350, 150, 50)],
    # carte arche
    [pygame.Rect(200, 150, 150, 330), pygame.Rect(350, 150, 150, 130), pygame.Rect(500, 150, 150, 330)],
    # carte vide
    [],
    # carte mur central
    [pygame.Rect(350, 100, 100, 400)],
    
    # carte grosse boite centrale
    # laisse un couloir ultra serre de 100 pixels sur les bord
    [pygame.Rect(100, 100, 600, 400)] 
]
liste_obstacles = random.choice(cartes)

solveur_rrt = RRT(depart_robot, arrivee_finale, 2000, liste_obstacles, 30, None)
chemin_trouve, _ = solveur_rrt.compute_path()

if chemin_trouve is None or len(chemin_trouve) == 0:
    chemin_trouve = [arrivee_finale]
else:
    chemin_trouve.pop(0) 

env = EnvironnementRobot(depart=depart_robot, waypoints=chemin_trouve)
env.liste_obstacles = liste_obstacles # met a jour la map physique

taille_etat = 13
nb_actions = 5
reseau = DQN(taille_etat, nb_actions)

nom_fichier_modele = "meilleur_modele_dqn.pth" 

if os.path.exists(nom_fichier_modele):
    reseau.load_state_dict(torch.load(nom_fichier_modele, weights_only=True))
    print("charger")
else:
    print("erreur")
    sys.exit()

# on met le reseau en mode evaluation pour qu'il ne calcule pas les gradients et qu'il puisse juste faire des predictions
reseau.eval()

while True:
    etat = env.reset()
    done = False
    
    while not done:
        env.clock.tick(30)  
        # pas de hasard
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