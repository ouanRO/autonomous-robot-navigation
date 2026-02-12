import pygame
import math
import sys
from RRT import RRT

pygame.init()

largeur, hauteur = 800, 600
fenetre = pygame.display.set_mode((largeur, hauteur))

fond_gris = (200, 200, 200)
robot_couleur = (50, 50, 50) # couleur noire

class robot:
    def __init__(self, x, y, vitesse_max, vitesse_rotation):
        self.vitesse_max = vitesse_max
        self.vitesse_rotation = vitesse_rotation
        self.vitesse = 0  # notre robot a une vitesse de 0 au debut
        self.angle = 0 # l'angle permet de nous guider dans quels direction le robot va tourner
        self.x = x
        self.y = y
        self.robot_rayon = 30
        self.acceleration = 0.1 # chaque fois qu on veut acclerer

    def rotation(self, gauche=False, droite=False):
        if gauche:
            self.angle += self.vitesse_rotation # En reduisant la vitesse de rotation notre angle tend vers la gauche
        elif droite:
            self.angle -= self.vitesse_rotation
    
    def avancer_droit(self):
        self.vitesse = min(self.vitesse + self.acceleration, self.vitesse_max) # limitateur de vitesse pour pas allew a l'infini on est pas dans l'espace
        self.mouvement()
    
    def mouvement(self):
        rad = math.radians(self.angle) # conversion en radian
        # On veut savoir combien le robot avance vers le haut donc  cosinus = adjacent / hypotenuse -> cos(teta) = verticale / vitesse
        # donc verticale = cos(teta) * vitesse
        verticale = math.cos(rad) * self.vitesse
        # mm logique pour l'horizontale sauf qu'ici on a sinus = oppose / Hypotenuse -> sin(teta) = horizontale / vitesse
        # donc  horizontale = sin(teta) * vitesse
        horizontale = math.sin(rad) * self.vitesse

        self.y -= verticale
        self.x -= horizontale


# on initialise le robot ici
robot_test = robot(largeur//2, hauteur//2, 5, 5)

# Boucle qui fait afficher notre fenetre tant qu on le ferme pas
run = True
clock = pygame.time.Clock() # Horloge pour gerer les images par seconde
IPS = 30

liste_obstacles = []
rectangle1 = pygame.Rect(200, 150, 100, 300)
liste_obstacles.append(rectangle1)
rectangle2 = pygame.Rect(500, 150, 100, 300)        
liste_obstacles.append(rectangle2)
rectangle3 = pygame.Rect(350, 150, 100, 100)
liste_obstacles.append(rectangle3)

depart = (50, 50)          # point de départ du robot
arrivee = (700, 500)       # objectif en bas à droite
long_branche = 30          # longueur d'une branche
iterations = 400          # nombre d'essais pour trouver le chemin dans le code de RRT fixée par yoann

# on place le robot au départ
robot_test.x = depart[0]
robot_test.y = depart[1]

solveur = RRT(depart, arrivee, iterations, liste_obstacles, long_branche)
chemin_trouve, arbre_complet = solveur.compute_path() # On récupère la liste de points du chemin trouvé pour l'afficher dans le simulateur

#print(chemin_trouve)

while run:
    clock.tick(IPS)

   
    fenetre.fill(fond_gris)
    for obstacle in liste_obstacles:
        pygame.draw.rect(fenetre, (255, 0, 0), obstacle) # on dessine les obstacles en rouge

    # on dessine les branches de l'arbre de recherche
    for branche in arbre_complet:
        # une branche contient ((x1, y1), (x2, y2))
        pygame.draw.line(fenetre, (150, 150, 150), branche[0], branche[1], 1)
    
    if len(chemin_trouve) > 1: # on dessine que si on a trouvé un chemin

        # pygame.draw.lines(surface, couleur, fermé ?, liste_points, épaisseur)
        pygame.draw.lines(fenetre, (0, 0, 255), False, chemin_trouve, 3)
        
        # on dessiner le point d'arrivée en Vert
        pygame.draw.circle(fenetre, (0, 255, 0), arrivee, 10)

    pygame.draw.circle(fenetre,robot_couleur, (robot_test.x, robot_test.y),robot_test.robot_rayon)
    pygame.display.flip()

    

    for event in pygame.event.get(): # Fermer la fenetre
        if event.type == pygame.QUIT:
            run = False
            break
    
    touche_clavier = pygame.key.get_pressed()

    if touche_clavier[pygame.K_q]:
        robot_test.rotation(gauche=True)
    if touche_clavier[pygame.K_d]:
        robot_test.rotation(droite=True)
    if touche_clavier[pygame.K_z]:
        robot_test.avancer_droit()
pygame.quit()