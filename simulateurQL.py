import pygame
import numpy as np
import math
import sys
import os
import random
from RRTql import RRT
from ql import QLearningAgent

pygame.init()

# CONFIGURATION FENÊTRE 
WIDTH, HEIGHT = 1000, 600 
SIM_WIDTH = 800 # Zone de simulation
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RRT + Q-Learning : Navigation Autonome")
clock = pygame.time.Clock()


# CLASSE ROBOT 

class Robot:
    def __init__(self, x, y):
        self.x_init, self.y_init = x, y
        self.radius = 18
        self.reset()

    def reset(self):
        """Réinitialise l'état du robot pour un nouvel épisode"""
        self.x, self.y = self.x_init, self.y_init
        self.angle = 0
        self.v = 0
        self.acc = 0.5
        self.max_speed = 4

    def move(self, action):
        """Applique les forces physiques selon l'action choisie par l'IA"""
        if action == 0: self.v += self.acc             # Accélérer
        elif action == 1: self.angle -= 15; self.v *= 0.9 # Tourner Gauche + freinage
        elif action == 2: self.angle += 15; self.v *= 0.9 # Tourner Droite + freinage
        elif action == 3: self.v -= self.acc             # Reculer/Freiner
        
        # Physique du mouvement
        self.v = max(0, min(self.v, self.max_speed))
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.v
        self.y += math.sin(rad) * self.v
        
        # Collision avec les bords de la zone
        self.x = max(self.radius, min(self.x, SIM_WIDTH - self.radius))
        self.y = max(self.radius, min(self.y, HEIGHT - self.radius))


# OUTILS MATHÉMATIQUES

def distance(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])

def angle_to(robot_obj, target_pos):
    """Calcule l'erreur d'angle entre l'orientation du robot et la cible"""
    dx, dy = target_pos[0] - robot_obj.x, target_pos[1] - robot_obj.y
    target_angle = math.degrees(math.atan2(dy, dx))
    diff = (target_angle - robot_obj.angle + 180) % 360 - 180
    return diff

def inflate(obstacles, r):
    """Ajoute une marge de sécurité autour des obstacles (Minkowski Sum simplifiée)"""
    return [pygame.Rect(o.x-r, o.y-r, o.width+2*r, o.height+2*r) for o in obstacles]


# INITIALISATION

start, goal = (60, 60), (740, 500)

def generate_random_obstacles(n=3):
    """Génère des obstacles aléatoires pour tester la robustesse"""
    obs_list = []
    for _ in range(n):
        w, h = random.randint(80, 150), random.randint(100, 300)
        x = random.randint(150, SIM_WIDTH - 250)
        y = random.randint(50, HEIGHT - h - 50)
        obs_list.append(pygame.Rect(x, y, w, h))
    return obs_list

obstacles = generate_random_obstacles(3)
inflated = inflate(obstacles, 25) # Obstacles avec zone de sécurité

# 1. Planification globale (RRT)
BRANCH_SIZE = 25 # Change ce chiffre pour tester !
rrt = RRT(start, goal, 2000, inflated, BRANCH_SIZE, None)
res = rrt.compute_path()
path = res[0] if res else sys.exit("RRT failed")

# 2. Agent Apprenant (Q-Learning)
robot = Robot(*start) 
agent = QLearningAgent()

# Chargement de la mémoire si elle existe
if agent.load("best_qtable.npy"):
    agent.epsilon = 0.1 
    print("Intelligence chargée !")
else:
    agent.epsilon = 1.0 # Exploration totale si nouveau
    print("Nouvel apprentissage...")

# Variables de suivi
waypoint_index = 1
best_steps = float('inf')
steps_episode = 0
episodes = 0
robot_track = [] 

# --- POLICES ---
font_title = pygame.font.SysFont("Arial", 22, bold=True)
font_data = pygame.font.SysFont("Courier", 16, bold=True)


# BOUCLE PRINCIPALE (Apprentissage)

running = True
while running:
    screen.fill((230, 230, 230))
    # Panneau latéral (Dashboard)
    pygame.draw.rect(screen, (35, 35, 40), (SIM_WIDTH, 0, 200, HEIGHT))

    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False

    # 1. Perception de l'état
    target = path[waypoint_index]
    dist = distance((robot.x, robot.y), target)
    ang = angle_to(robot, target)
    
    # Discrétisation pour la Q-Table
    state = agent.discretize(dist, ang, robot.v)
    
    # 2. Action
    action = agent.choose_action(state)
    old_dist = dist
    robot.move(action)
    steps_episode += 1
    
    # 3. Observation du nouvel état
    robot_track.append((robot.x, robot.y))
    if len(robot_track) > 250: robot_track.pop(0)
    
    new_dist = distance((robot.x, robot.y), target)
    new_ang = angle_to(robot, target)
    next_state = agent.discretize(new_dist, new_ang, robot.v)
    
    collision = any(o.collidepoint(robot.x, robot.y) for o in inflated)

    #  LOGIQUE DE RÉCOMPENSE  
    reward = 0
    shortcut_detected = False
    dist_to_final_goal = distance((robot.x, robot.y), path[-1])

    if collision:
        reward = -500 # Punition pour choc
        rad = math.radians(robot.angle)
        robot.x -= math.cos(rad) * 15 # Recul forcé
        robot.y -= math.sin(rad) * 15
        robot.v = 0
    
    elif dist_to_final_goal < 30: # BUT ATTEINT
        reward = 30000 
        if steps_episode < best_steps:
            best_steps = steps_episode
            agent.save("best_qtable.npy")
        pygame.time.delay(2000)
        robot.reset(); waypoint_index = 1; steps_episode = 0; episodes += 1; robot_track = []
        
    else:
        # --- MÉCANISME DE RACCOURCI (Look-ahead) ---
        for i in range(min(15, len(path) - waypoint_index)):
            check_idx = waypoint_index + i
            if distance((robot.x, robot.y), path[check_idx]) < 70:
                if check_idx > waypoint_index:
                    # Plus le robot "saute" de points, plus on le récompense
                    reward = 5000 + ((check_idx - waypoint_index) * 2000)
                    waypoint_index = check_idx
                    shortcut_detected = True
                    break

        if not shortcut_detected:
            if new_dist < 35: # Passage au point suivant "normal"
                reward = 100
                waypoint_index = min(waypoint_index + 1, len(path)-1)
            else:
                # Calcul de la punition de fluidité (angle)
                alignment_penalty = (abs(ang) ** 1.2) / 10
                # Progression positive si on se rapproche de la cible
                reward = (old_dist - new_dist) * 40 - alignment_penalty - 2

    # 4. Apprentissage (Mise à jour de la Q-Table)
    agent.update(state, action, reward, next_state)
    current_reward = int(reward)

    # --- AFFICHAGE GRAPHIQUE ---
    for o in obstacles: pygame.draw.rect(screen, (220, 60, 60), o) # Obstacles
    pygame.draw.lines(screen, (60, 60, 200), False, path, 2) # Chemin RRT
    
    if len(robot_track) > 2:
        pygame.draw.lines(screen, (50, 255, 50), False, robot_track, 2) # Tracé réel
    
    # Dessin des Waypoints
    for i, p in enumerate(path):
        if i == len(path) - 1:
            pygame.draw.circle(screen, (0, 255, 0), (int(p[0]), int(p[1])), 10)
        else:
            c = (255, 255, 0) if i == waypoint_index else (80, 80, 160)
            pygame.draw.circle(screen, c, (int(p[0]), int(p[1])), 5 if i == waypoint_index else 3)

    # Dessin Robot
    pygame.draw.circle(screen, (40, 40, 40), (int(robot.x), int(robot.y)), robot.radius)
    
    # fenetre d'affichage
    panel_x = SIM_WIDTH + 15
    
    screen.blit(font_title.render("AFFICHAGE", True, (255, 255, 255)), (panel_x, 20))
    
    rew_col = (100, 255, 100) if current_reward >= 0 else (255, 100, 100)
    screen.blit(font_data.render(f"REWARD: {current_reward}", True, rew_col), (panel_x, 80))
    screen.blit(font_data.render(f"EPS:    {agent.epsilon:.3f}", True, (200, 200, 200)), (panel_x, 110))
    screen.blit(font_data.render(f"TARGET: {waypoint_index}/{len(path)-1}", True, (200, 200, 200)), (panel_x, 140))
    screen.blit(font_data.render(f"SPEED:  {robot.v:.2f}", True, (200, 200, 200)), (panel_x, 170))
    screen.blit(font_data.render(f"ANGLE:  {int(ang)}°", True, (200, 200, 200)), (panel_x, 200))
    screen.blit(font_data.render(f"BEST:   {best_steps if best_steps != float('inf') else '---'}", True, (255, 255, 0)), (panel_x, 240))

    pygame.display.flip()
    clock.tick(60)
pygame.quit()