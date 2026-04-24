import numpy as np
import random
import pygame


class treeNode():

    def __init__(self,X,Y,teta=None):

        self.X = X
        self.Y = Y
        self.teta = teta

        self.children = []
        self.parent = None


class RRT():

    def __init__(self,start,goal,iteration,grid,stepSize,teta):

        self.randomTree = treeNode(start[0],start[1])
        self.goal = treeNode(goal[0],goal[1])

        self.iteration = iteration

        self.grid = grid

        self.lengthBranch = stepSize

        self.angleBranch = np.radians(teta) if teta is not None else None

        self.nearestNode = None
        self.nearestDist = 10000

        self.pathDist = 0
        self.waypoints = []
        self.numberWaypoints = 0

        self.toutes_les_branches = []


    def addChild(self,x,y,teta=None):

        if abs(x-self.goal.X)<2 and abs(y-self.goal.Y)<2:

            self.toutes_les_branches.append(((self.nearestNode.X,self.nearestNode.Y),(self.goal.X,self.goal.Y)))

            self.nearestNode.children.append(self.goal)

            self.goal.parent = self.nearestNode
            self.goal.teta = teta

        else:

            newNode = treeNode(x,y,teta)

            self.toutes_les_branches.append(((self.nearestNode.X,self.nearestNode.Y),(x,y)))

            self.nearestNode.children.append(newNode)

            newNode.parent = self.nearestNode


    def sampleAPoint(self):

        # goal bias 10%
        if random.random() < 0.1:
            return np.array([self.goal.X,self.goal.Y])

        x = random.randint(1,800)
        y = random.randint(1,600)

        return np.array([x,y])


    def unitVector(self,start,end):

        vector = np.array([end[0]-start.X,end[1]-start.Y])

        norm = np.linalg.norm(vector)

        if norm == 0:
            return np.array([0,0])

        return vector/norm


    def steerToPoint(self,start,end):

        direction = self.unitVector(start,end)

        point = np.array([
            start.X + direction[0]*self.lengthBranch,
            start.Y + direction[1]*self.lengthBranch
        ])

        teta = np.arctan2(point[1]-start.Y,point[0]-start.X)

        if point[0] >= 800: point[0] = 799
        if point[1] >= 600: point[1] = 599
        if point[0] < 0: point[0] = 0
        if point[1] < 0: point[1] = 0

        return point,teta


    def isInObstacle(self,start,end):

        direction = self.unitVector(start,end)

        for i in range(self.lengthBranch):

            x = start.X + i*direction[0]
            y = start.Y + i*direction[1]

            for rect in self.grid:

                if rect.collidepoint(x,y):
                    return True

        return False


    def distance(self,node,point):

        return np.sqrt((node.X-point[0])**2 + (node.Y-point[1])**2)


    def findNearest(self,root,point):

        if root is None:
            return

        dist = self.distance(root,point)

        if dist <= self.nearestDist:

            self.nearestNode = root
            self.nearestDist = dist

        for child in root.children:

            self.findNearest(child,point)


    def resetNearestValues(self):

        self.nearestNode = None
        self.nearestDist = 10000


    def goalFound(self,point):

        return self.distance(self.goal,point) <= self.lengthBranch


    def retraceRRTPath(self,goal):

        if goal.X == self.randomTree.X and goal.Y == self.randomTree.Y:
            return

        self.numberWaypoints += 1

        current = np.array([goal.X,goal.Y])

        self.waypoints.insert(0,current)

        self.pathDist += self.lengthBranch

        self.retraceRRTPath(goal.parent)


    def compute_path(self):

        self.randomTree = treeNode(self.randomTree.X,self.randomTree.Y)

        self.waypoints = []
        self.toutes_les_branches = []
        self.pathDist = 0

        for i in range(self.iteration):

            self.resetNearestValues()

            point = self.sampleAPoint()

            self.findNearest(self.randomTree,point)

            new,teta = self.steerToPoint(self.nearestNode,point)

            if not self.isInObstacle(self.nearestNode,new):

                self.addChild(new[0],new[1],teta)

                if self.goalFound(new):

                    self.addChild(self.goal.X,self.goal.Y,teta)

                    self.retraceRRTPath(self.goal)

                    path = []

                    path.append((self.randomTree.X,self.randomTree.Y))

                    for p in self.waypoints:

                        path.append((p[0],p[1]))

                    return path,self.toutes_les_branches

        print("Aucun chemin trouvé")

        return None