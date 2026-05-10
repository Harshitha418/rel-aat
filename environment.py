import random
class ClassroomEnvironment:

    def __init__(self):
        self.reset()

    def reset(self):
        self.occupancy = random.randint(0, 1)
        self.temperature = random.randint(0, 2)
        self.light = random.randint(0, 1)
        self.device_state = 0
        return self.get_state()
    
    def get_state(self):
        return (
            self.occupancy,
            self.temperature,
            self.light,
            self.device_state
        )

    def step(self, action):
        self.device_state = action
        energy_used = action * 2
        if self.occupancy == 0 and action != 0:
            reward = -10
        
        elif self.occupancy == 1 and action == 0:
            reward = -5
        
        else:
            reward = 5 - energy_used
        
        self.occupancy = random.randint(0, 1)
        self.temperature = random.randint(0, 2)
        self.light = random.randint(0, 1)
        next_state = self.get_state()
        done = False
        return next_state, reward, done
