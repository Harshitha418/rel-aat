# Smart Classroom Energy Saver using Reinforcement Learning

## Objective

This project uses Reinforcement Learning to reduce classroom energy consumption while maintaining comfort.

## Problem Statement

Classrooms often waste electricity by keeping lights and fans ON unnecessarily.

The RL agent learns an optimal policy to:
- reduce energy wastage
- maintain comfort
- automate appliance control

## RL Methodology

### Algorithm
Q-Learning

### State
- Occupancy
- Temperature
- Light level
- Device state

### Actions
- Turn OFF devices
- Turn ON lights
- Turn ON fan/AC
- Turn ON all devices

### Reward
Positive reward:
- energy saving
- comfort maintenance

Negative reward:
- unnecessary energy usage

## Exploration Strategy
Epsilon-greedy exploration

## Results
- Reward convergence achieved
- Energy usage reduced
- Policies successfully learned