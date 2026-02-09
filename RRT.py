from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import random

from matplotlib.pyplot import rcParams
np.set_printoptions(precision=3, suppress=True)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
plt.rcParams['font.size'] = 22

class treeNode():
    def __init__(self, X, Y):
        self.X = X #position x
        self.Y = Y  # position y
        self.children = [] #enfants des noeuds
        self.parent = None # parents des noeuds

class RRT():
    def __init__(self, start, goal, iteration, grid, stepSize):
        self.randomTree = treeNode(start[0],start[1]) # position de départ
        self.goal = treeNode(goal[0],goal[1])          # objectif
        self.nearestNode = None                         # plus proche noeud
        self.iteration = min(iteration,400)             # nombre total d'itération (limite a 200)
        self.grid = grid                                # la carte
        self.lengthBranch = stepSize                    # longueur des branches
        self.pathDist = 0                               # longueur total parcouru
        self.nearestDist = 10000                        # distancedu plus proche noeud (intialiser a un trs gros nombre)
        self.numberWaypoints = 0                        # nombre de point de repère
        self.waypoints = []                             #liste des points de repères

    def addChild(self, x,y):
        # Ajoute un nouveau noeud enfant au noeud le plus proche
        # si le noeud correspond au but il connecte directement au but
        if( x == self.goal.X):
            self.nearestNode.children.append(self.goal)
            self.goal.parent = self.nearestNode
        else:
            tampNode = treeNode(x,y)
            self.nearestNode.children.append(tampNode)
            tampNode.parent = self.nearestNode

    def sampleAPoint(self):
        # Génère un point aléatoire dans la grille pour l’exploration
        x = random.randint(1, grid.shape[1])
        y = random.randint(1, grid.shape[0])
        point = np.array([x,y])
        return point
    
    def steerToPoint(self, start,end):
        # Calcule un nouveau point dans la direction de end depuis star
    # à une distance égale à la longueur d’une branche 
        offset = self.lengthBranch*self.unitVector(start,end)
        point = np.array([start.X + offset[0], start.Y + offset[1]])
        if point[0] >= grid.shape[1]:
            point[0] = grid.shape[1]-1
        if point[1] >= grid.shape[0]:
            point[1] = grid.shape[0]-1
        return point
    
    def isInObstacle(self,start,end):

        #Vérifie si la branche entre deux points traverse un obstacle dans la grille

        # Version avec vérification d'osbtacle sur chemin d'une branche
        u_hat = self.unitVector(start,end) #u_hat : vecteur u normaliser (û)
        testPoint = np.array([0.0,0.0])
        for i in range(self.lengthBranch):
            testPoint[0] = start.X+ i*u_hat[0]
            testPoint[1] = start.Y+ i*u_hat[1]
            if self.grid[round(testPoint[1]),round(testPoint[0])] == 1:
                return True
        return False
        
        '''
        return (self.grid[round(end[1]),round(end[0])] == 1) // ne verifie pas si il y'a un obstacle sur le chemin d'une branche 
        '''
        
        
    
    def unitVector(self,start,end):
        # Calcule et retourne le vecteur unitaire normalisé entre deux points
        vector = np.array([end[0]-start.X, end[1]-start.Y])
        u_hat = vector/np.linalg.norm(vector)
        return u_hat
    
    def findNearest(self,root,point):
        # Recherche récursive dans l’arbre le noeud le plus proche d’un point donné
        if not root:
            return
        dist = self.distance(root,point)
        if dist <= self.nearestDist:
            self.nearestNode = root
            self.nearestDist = dist
        for i in root.children:
            self.findNearest(i,point)


    def distance(self,noeud1,point):
         # Calcule la distance euclidienne entre un noeud et un point
        dist = np.sqrt((noeud1.X - point[0])**2 + (noeud1.Y - point[1])**2)
        return dist
    
    def goalFound(self,point):
        # Indique si un point est suffisamment proche du but pour considérer que le but est atteint
        if self.distance(self.goal,point) <= self.lengthBranch:
            return True

    def resetNearestValues(self):
        # Réinitialise les variables utilisées pour trouver le noeud le plus proche avant une nouvelle recherche
        self.nearestNode = None
        self.nearestDist = 10000
    
    def retraceRRTPath(self,goal):
        # Remonte récursivement depuis le noeud but jusqu’à la racine
    #  en enregistrant le chemin et calculant la distance totale
        if goal.X == self.randomTree.X:
            return
        self.numberWaypoints += 1
        currentPoint = np.array([goal.X, goal.Y])
        self.waypoints.insert(0,currentPoint)
        self.pathDist += self.lengthBranch
        self.retraceRRTPath(goal.parent)

grid = np.load("rrt_map_500x1000.npy")
start = np.array([100.0 ,100.0])
goal = np.array([700,250])
numIterations = 400
stepSize = 50
goalRegion = plt.Circle((goal[0],goal[1]), stepSize, color='b',fill = False)

fig = plt.figure("RRT")
plt.imshow(grid,cmap='binary')
plt.plot(start[0],start[1],'ro')
plt.plot(goal[0],goal[1],'bo')
ax = fig.gca()
ax.add_patch(goalRegion)
plt.xlabel('X-axis ${m}$')
plt.ylabel('Y-axis ${m}$')

rrt = RRT(start,goal,numIterations,grid,stepSize)

for i in range(rrt.iteration):
    rrt.resetNearestValues()
    print("Iteration ",i)
    point = rrt.sampleAPoint()
    rrt.findNearest(rrt.randomTree,point)
    new = rrt.steerToPoint(rrt.nearestNode,point)
    bool = rrt.isInObstacle(rrt.nearestNode,new)
    if(bool == False):
        rrt.addChild(new[0],new[1])
        plt.pause(0.10)
        plt.plot([rrt.nearestNode.X, new[0]], [rrt.nearestNode.Y, new[1]],'go', linestyle='--')
        if (rrt.goalFound(new)):
            rrt.addChild(goal[0],goal[1])
            print("goal found")
            break

         

rrt.retraceRRTPath(rrt.goal)
rrt.waypoints.insert(0,start)
print("Nombre de repères: ",rrt.numberWaypoints)
print("distance du chemin : ",rrt.pathDist)
print("point de repère: ",rrt.waypoints)
for i in range(len(rrt.waypoints)-1):
    plt.plot([rrt.waypoints[i][0],rrt.waypoints[i+1][0]], [rrt.waypoints[i][1],rrt.waypoints[i+1][1]], 'ro',linestyle="--")
    plt.pause(0.10)
plt.show() 

