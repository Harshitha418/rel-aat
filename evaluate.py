import pickle

from environment import ClassroomEnvironment

# Load trained policy
with open("policies/policy_v2.pkl", "rb") as f:
    q_table = pickle.load(f)

env = ClassroomEnvironment()

state = env.reset()

print("\nSMART CLASSROOM ENERGY SAVER\n")
action_meaning = {
    0: "All Devices OFF",
    1: "Lights ON",
    2: "Fan/AC ON",
    3: "All Devices ON"
}

for step in range(10):

    q_values = []

    for action in [0, 1, 2, 3]:

        q = q_table.get((state, action), 0)

        q_values.append(q)

    best_action = q_values.index(max(q_values))

    next_state, reward, done = env.step(best_action)

    print(f"Step {step+1}")
    print(f"State: {state}")
    print(f"Chosen Action: {action_meaning[best_action]}")
    print(f"Reward: {reward}")
    print("------------------------")

    state = next_state