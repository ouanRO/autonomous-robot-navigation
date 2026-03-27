import pygame
import math
import numpy as np

class EnvironnementRobot:
    def __init__(self, depart=(50, 50), waypoints=None):
        pygame.init()
        # configuration de la fenetre et des paramètres du robot
        self.largeur, self.hauteur = 800, 600
        self.screen = pygame.display.set_mode((self.largeur, self.hauteur))
        self.clock = pygame.time.Clock()

        # configuration de l'environnement carte, obstacles, etc 
        self.depart = depart
        if waypoints is None:
            waypoints = [(700, 500)]
        self.waypoints_originaux = waypoints.copy()  # on garde une copie pour pouvoir reset l'environnement
        
        self.vitesse_max = 5
        self.vitesse_rotation = 10 
        self.acceleration = 0.5    
        self.robot_rayon = 25
        
        # configuration de l'obastacle en L 
        self.liste_obstacles = [
            pygame.Rect(400, 200, 50, 200),
            pygame.Rect(450, 350, 150, 50),
        ]
        # on reset l'environnement pour initialiser les variables d'etat du robot et la premiere cible
        self.reset()

    # la fonction reset remet le robot a sa position de depart, 
    # remet les waypoints a leur ordre d'origine et choisit le premier waypoint comme cible
    def reset(self):
        self.x, self.y = self.depart
        self.angle = 0
        self.vitesse = 0    
        self.waypoints = self.waypoints_originaux.copy()
        self.arrivee = self.waypoints.pop(0)
        
        # on calcule la distance initiale pour pouvoir donner une recompose par rapport a l'eloignetment ou l'approche de la cible
        self.dist_precedente = math.dist((self.x, self.y), self.arrivee)
        
        # on retourne l'etat initial du robot pour que le reseau puisse commencer a apprendre
        return self._get_etat()

    # la fonction _get_etat calcule les 8 valeurs de l'etat a partir de la position,
    # l'angle, la vitesse du robot et la position de la cible
    def _get_etat(self):
        # calcul de la distance et de l'angle vers la cible
        dx = self.arrivee[0] - self.x
        dy = self.arrivee[1] - self.y
        dist = math.sqrt(dx**2 + dy**2)
        angle_vers_cible = math.atan2(dy, dx)
        angle_robot_rad  = math.radians(self.angle)
        diff_angle = angle_vers_cible - angle_robot_rad
        
        # on normalise les valeurs pour que le reseau puisse apprendre plus facilement (entre 0 et 1 ou -1 et 1)
        # on retourne donc un tableau de 8 valeurs :
        #  position x et y normalisées, direction cos/sin, vitesse normalisee, distance relative, angle relatif (cos/sin)
        return np.array([
            self.x / self.largeur,
            self.y / self.hauteur,
            math.cos(angle_robot_rad), # direction en x 
            math.sin(angle_robot_rad), # direction en y
            self.vitesse / self.vitesse_max,
            dist / math.sqrt(self.largeur**2 + self.hauteur**2), # Distance relative
            math.cos(diff_angle),  # angle relatif vers cible si = 1 il regarde tout droit vers la cible
            math.sin(diff_angle),      
        ], dtype=np.float32)

    # la fonction step applique l'action choisie par le reseau
    # met a jour la position du robot, calcule la recompense et 
    # retourne le nouvel etat, la recompense et si l episode est termine
    # les actions sont : 0 = accelerer, 1 = freiner, 2 = tourner a gauche, 3 = tourner a droite, 4 = ne rien faire
    def step(self, action):
        if action == 0: 
            self.vitesse = min(self.vitesse + self.acceleration, self.vitesse_max)
        elif action == 1: 
            self.vitesse = max(self.vitesse - self.acceleration, 0)
        elif action == 2: 
            self.angle = (self.angle - self.vitesse_rotation) % 360 
        elif action == 3: 
            self.angle = (self.angle + self.vitesse_rotation) % 360
       
        # physique standard pour faire avancer le robot dans la direction de son angle a une vitesse donnee
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.vitesse
        self.y += math.sin(rad) * self.vitesse
        
        reward = -0.1 # petite penalite de temps
        done = False 
        
        # on verifie les collisions avec les murs et les obstacles
        # si collision c'est une penalite et la fin de l episode
        robot_rect = pygame.Rect(self.x-15, self.y-15, 30, 30)
        if (self.x < 0 or self.x > self.largeur or self.y < 0 or self.y > self.hauteur or 
            any(robot_rect.colliderect(obs) for obs in self.liste_obstacles)):
            return self._get_etat(), -50, True 

        # on calcule la distance actuelle vers la cible pour donner  
        # une recompense positive s'il s'en approche et negative s'il s'en eloigne
        dist_actuelle = math.dist((self.x, self.y), self.arrivee)
        reward += (self.dist_precedente - dist_actuelle) * 3.0
        self.dist_precedente = dist_actuelle 

        # on donne une recompense supplementaire s
        # si le robot regarde dans la direction de la cible pour encourager un bon alignement 
        dx = self.arrivee[0] - self.x
        dy = self.arrivee[1] - self.y
        angle_vers_cible = math.atan2(dy, dx)
        alignement = math.cos(angle_vers_cible - math.radians(self.angle))
        reward += max(0, alignement) * 0.2

        # si cible atteinte avec une marge de 1.5 fois le rayon du robot pour que ce soit plus facile
        # on donne une grosse recompense et on passe a la cible suivante
        if dist_actuelle < self.robot_rayon * 1.5:
            if len(self.waypoints) > 0:
                self.arrivee = self.waypoints.pop(0)
                # on calcule la distance precedente pour la nouvelle cible pour que le robot puisse 
                # continuer a recevoir des recompenses d'approche ou d'eloignement
                self.dist_precedente = math.dist((self.x, self.y), self.arrivee)
                reward += 100 
            else:
                reward += 200
                done = True

        return self._get_etat(), reward, done
    
    def dessiner(self):
        self.screen.fill((230, 230, 230))
        pygame.draw.circle(self.screen, (0, 255, 0), (int(self.arrivee[0]), int(self.arrivee[1])), 15)
        for obs in self.liste_obstacles: pygame.draw.rect(self.screen, (255, 0, 0), obs)
        pygame.draw.circle(self.screen, (50, 50, 50), (int(self.x), int(self.y)), self.robot_rayon)
        end_x = self.x + math.cos(math.radians(self.angle)) * 30
        end_y = self.y + math.sin(math.radians(self.angle)) * 30
        pygame.draw.line(self.screen, (255, 255, 255), (self.x, self.y), (end_x, end_y), 3)
        pygame.display.flip()