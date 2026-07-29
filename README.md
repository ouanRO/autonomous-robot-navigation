# 🤖 Reinforcement Learning for Autonomous Mobile Robot Navigation

## 📖 Overview

This project explores the use of **Reinforcement Learning (RL)** for autonomous mobile robot navigation in a simulated environment.

The objective is to enable a mobile robot to autonomously navigate from a **starting position (Point A)** to a **target position (Point B)** while avoiding obstacles.

The project combines **classical path planning** with **machine learning** by integrating the **Rapidly-exploring Random Tree (RRT)** algorithm and comparing two Reinforcement Learning approaches:

- **Q-Learning**
- **Deep Q-Network (DQN)**

The main goal is to evaluate the advantages and limitations of both algorithms in terms of learning efficiency, navigation performance, and convergence.

---

## 🎯 Objectives

- Design a simulated environment for autonomous navigation.
- Generate collision-free trajectories using the RRT algorithm.
- Train an autonomous agent using Q-Learning.
- Replace the tabular approach with a Deep Q-Network (DQN).
- Compare the performance of both learning methods.
- Analyze the strengths and limitations of each approach.

---

## 🏗 Project Architecture

```
                 Simulated Environment
                         │
                         ▼
                 Obstacle Generation
                         │
                         ▼
          RRT Path Planning Algorithm
                         │
                         ▼
           Reinforcement Learning Agent
              ├──────────────┤
              │              │
              ▼              ▼
         Q-Learning        DQN
              │              │
              └──────┬───────┘
                     ▼
             Robot Navigation
                     │
                     ▼
          Performance Evaluation
```

---

## ⚙️ Technologies

### Programming Language

- Python

### Libraries

- NumPy
- Pygame
- Matplotlib

### Artificial Intelligence

- Reinforcement Learning
- Q-Learning
- Deep Q-Network (DQN)

### Robotics

- Autonomous Navigation
- Path Planning
- Rapidly-exploring Random Tree (RRT)

---

## 🧠 Methodology

### 1. Environment

A two-dimensional simulated environment is created with static obstacles.

The robot starts from an initial position and must safely reach the goal while avoiding collisions.

---

### 2. Path Planning

The **Rapidly-exploring Random Tree (RRT)** algorithm generates feasible paths through the environment.

These paths are used to guide the navigation process.

---

### 3. Reinforcement Learning

Two different learning strategies are implemented.

#### Q-Learning

The robot learns an optimal navigation policy using a Q-table that stores state-action values.

#### Deep Q-Network (DQN)

The Q-table is replaced by a neural network capable of approximating the optimal action-value function.

This allows better scalability for larger state spaces.

---

### 4. Performance Comparison

Both methods are evaluated according to several criteria:

- Navigation success rate
- Number of collisions
- Training convergence
- Total accumulated reward
- Path length
- Computational cost

---

## 📊 Results

The following elements will be added after the experiments:

- Learning curves
- Reward evolution
- Success rate comparison
- Navigation trajectories
- Performance tables

---
