import pygame
import math
import numpy as np

class EnvironnementRobot:
    def __init__(self):

        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Simulation du robot DQL")
        self.clock = pygame.time.Clock()

        # constante de l'env
        self.largeur = 800
        self.hauteur = 600
        self.depart = (50, 50)
        self.arrivee = (700, 500)
        
        # spe du robot
        self.vitesse_max = 5
        self.vitesse_rotation = 5
        self.acceleration = 0.1
        self.robot_rayon = 30
        
        # obstacles (pour calculer les collisions sans les dessiner)
        self.liste_obstacles = [
            pygame.Rect(200, 150, 130, 330),
            pygame.Rect(500, 150, 130, 330),
            pygame.Rect(350, 150, 130, 130)
        ]
        
        # variables d'état du robot
        self.x = 0
        self.y = 0
        self.angle = 0
        self.vitesse = 0

    def reset(self):
        # remet le robot à sa position de depart et retourne l'etat initial (x, y, angle, vitesse)
        self.x = self.depart[0]
        self.y = self.depart[1]
        self.angle = 0
        self.vitesse = 0    

        etat = (self.x, self.y, self.angle, self.vitesse)
        return etat

    def step(self, action):
        # on l'appele step a chaaque tour pour faire avancer le robot d'une action (0 à 4)
        # 0 = accelerer, 1 = freiner, 2 = tourne a gauche, 3 = tourne a droite, 4 = ne rein faire
        if action == 0:
            self.vitesse = min(self.vitesse + self.acceleration, self.vitesse_max)
        elif action == 1: 
            self.vitesse = max(self.vitesse - self.acceleration, 0)
        elif action == 2:    
            self.angle = (self.angle - self.vitesse_rotation) % 360 # pour avoir un angle entre 0 et 360 et pas 109843 a la fin 
        elif action == 3: 
            self.angle = (self.angle + self.vitesse_rotation) % 360
       
        
        # physique du mouvement 
        rad = math.radians(self.angle)
        self.y -= math.cos(rad) * self.vitesse
        self.x -= math.sin(rad) * self.vitesse
        
     
        reward = 0
        done = False # devient True si on gagne ou si on s'écrase
        
        # on detecte si le robot sort de l'écran (0 < x < largeur, et 0 < y < hauteur)
        
        if self.x < 0 or self.x > self.largeur or self.y < 0 or self.y > self.hauteur:
            done = True
            reward = -100 # grosse punition pack sortie de l'écran
        
        # on detecte les collisions avec les obstacles
        robot_rect = pygame.Rect(self.x - self.robot_rayon, self.y - self.robot_rayon, self.robot_rayon*2, self.robot_rayon*2)
        for obstacle in self.liste_obstacles:
            if robot_rect.colliderect(obstacle):
                done = True
                reward = -100 # grosse punition pack collision

        # on detecte si on est arrivé a destination 
        
        if math.dist((self.x, self.y), self.arrivee) < self.robot_rayon:
            done = True
            reward = 100 # grosse recompense pack victoire
        
        # on ajoute une mini-punition à chaque tour  pour le forcer à trouver le meilleur chemin plus rapidement
        if not done:
            reward = -1 

        # puis on reconstruit le nouvel etat s'
        nouvel_etat = (self.x, self.y, self.angle, self.vitesse)
        
        return nouvel_etat, reward, done
    
    def dessiner(self):
        # fond gris
        self.screen.fill((200, 200, 200))
        
        # dessiner l'arrivee 
        pygame.draw.circle(self.screen, (0, 255, 0), self.arrivee, 20)
        
        # dessiner les obstacles 
        for obstacle in self.liste_obstacles:
            pygame.draw.rect(self.screen, (255, 0, 0), obstacle)
            
        # dessiner le robot 
        pygame.draw.circle(self.screen, (50, 50, 50), (int(self.x), int(self.y)), self.robot_rayon)
        
        # maj l'affichage
        pygame.display.flip()
        self.clock.tick(60) 