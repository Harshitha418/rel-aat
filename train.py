import pickle
import matplotlib.pyplot as plt
from environment import ClassroomEnvironment
from agent import QLearningAgent
env = ClassroomEnvironment()
agent = QLearningAgent()
episodes = 500
rewards = []
average_energy_per_episode = []
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    episode_energy = 0
    for step in range(20):
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        energy_used = action * 2
        episode_energy += energy_used
        agent.update(
            state,
            action,
            reward,
            next_state
        )
        state = next_state
        total_reward += reward
    average_energy = episode_energy / 20
    average_energy_per_episode.append(average_energy)
    agent.decay_epsilon()
    rewards.append(total_reward)
    print(
        f"Episode {episode+1}, "
        f"Reward = {total_reward}, "
        f"Epsilon = {agent.epsilon:.3f}"
    )
    if episode == 100:
        with open("policies/policy_v1.pkl", "wb") as f:
            pickle.dump(agent.q_table, f)
with open("policies/policy_v2.pkl", "wb") as f:
    pickle.dump(agent.q_table, f)

# -------------------------------
# Reward Convergence Graph
# -------------------------------

plt.figure(figsize=(10, 5))
plt.plot(
    rewards,
    linewidth=2
)
plt.xlabel("Episodes", fontsize=12)
plt.ylabel("Total Reward", fontsize=12)
plt.title(
    "Training Convergence",
    fontsize=14,
    fontweight='bold'
)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "graphs/convergence.png",
    dpi=300
)

# -------------------------------
# Energy Consumption Graph
# -------------------------------

plt.figure(figsize=(10, 5))
plt.plot(
    average_energy_per_episode,
    linewidth=2
)
plt.xlabel("Episodes", fontsize=12)
plt.ylabel("Average Energy Usage", fontsize=12)
plt.title(
    "Average Energy Consumption",
    fontsize=14,
    fontweight='bold'
)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "graphs/energy.png",
    dpi=300
)
plt.show()