import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, alpha, gamma, epsilon):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.next_waypoint = None
        self.ql = QLearner((None, 'forward', 'left', 'right'), alpha, gamma, epsilon)
        
        #stats
        self.total_reward = 0
        self.trial_array = [[0 for x in range(101)] for x in range(2)]
        self.n_trial = 0
        self.turn = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.next_waypoint = None
        self.state = None

        #stats
        self.total_reward = 0
        self.n_trial += 1
        self.turn = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        #select what algorithm to run
        #basic = True #run code for section 1 & 2
        basic = False #run Q-Learning algorithm

        if basic:
            # TODO: Select action according to your policy
            action = random.choice((None, 'forward', 'left', 'right'))

            # TODO: Update state
            self.state = (inputs, self.next_waypoint)

            # TODO: Select action according to your policy
            go = False
            action = self.next_waypoint

            if self.next_waypoint == 'forward':
                if inputs['light'] == 'green':
                    go = True
            elif self.next_waypoint == 'right':
                if inputs['light'] == 'red' and inputs['oncoming'] != 'left' or inputs['left'] != 'forward':
                    go = True
            elif self.next_waypoint == 'left':
                if inputs['light'] == 'green' and inputs['oncoming'] != 'forward':
                    go = True

            if not go:
                action = None

            # Execute action and get reward
            reward = self.env.act(self, action)
            self.total_reward += reward
            
        else:
            # TODO: Update state
            inputs = inputs.items()
            self.state = (inputs[0], inputs[1], inputs[3], self.next_waypoint)

            # TODO: Select action according to your policy
            
            #QLearner
            action = self.ql.select_action(self.state)

            # Execute action and get reward
            reward = self.env.act(self, action)
            self.total_reward += reward

            # TODO: Learn policy based on state, action, reward
            new_inputs = self.env.sense(self)
            new_inputs = new_inputs.items()
            
            new_state = (new_inputs[0], new_inputs[1], new_inputs[3], self.next_waypoint)
            self.ql.learn(self.state, new_state, action, reward)

        self.trial_array[0][self.n_trial] = self.turn
        self.trial_array[1][self.n_trial] = self.total_reward
        self.turn += 1
            
        print "Total Reward:"+str(self.total_reward)
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

class QLearner():
    def __init__(self, actions, alpha=0.5, gamma=0.2, epsilon=0.07):
        self.q = {}
        self.all_actions = actions
        self.alpha = alpha #reward multiplier
        self.gamma = gamma #delayed reward discount multiplier
        self.epsilon = epsilon #amount of randomness for action

        print epsilon
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.all_actions) #select random action
        else:
            # select the best available action
            q = [self.get_q(state, a) for a in self.all_actions]
            max_q = max(q)

            if q.count(max_q) > 1:
                best = [i for i in range(len(self.all_actions)) if q[i] == max_q]
                i = random.choice(best)
            else:
                i = q.index(max_q)
                
            action = self.all_actions[i]

        return action

    def learn_q(self, state, action, reward, value):
        old_v = self.q.get((state, action), None)

        if old_v == None:
            new_v = reward
        else:
            new_v = old_v + self.alpha * (value - old_v)

        self.set_q(state, action, new_v) #update table

    def learn(self, state, new_state, action, reward):
        q = [self.get_q(new_state, a) for a in self.all_actions]
        delayed_reward = int(max(q))
        
        self.learn_q(state, action, reward, reward - self.gamma * delayed_reward)

    #get table value with key
    def get_q(self, state, action):
        return self.q.get((state, action), 0.0)

    #set table value at key
    def set_q(self, state, action, q):
        self.q[(state, action)] = q
        

def run():
    """Run the agent for a finite number of trials."""
    import numpy as np

    #intuition values 1st selected
    #alpha = 0.5
    #gamma = 0.7
    #epsilon = 0.05

    #optimal values found
    alpha = 0.5
    gamma = 0.2
    epsilon = 0.07

    #some "bad" values just to test how good our optimal is
    #alpha = 0.8
    #gamma = 0.6
    #epsilon = 0.2

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent, alpha, gamma, epsilon)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit

    mean = np.mean(a.trial_array[0][75:100])
    print "Average Steps: "+str(mean)
    
    #find_optimal()
    

def find_optimal():
    import numpy as np
    optimal_array = [[99 for x in range(101)] for x in range(2)]
    
    #1st run
    #alpha_a = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)#, 0.8, 0.9, 1.0)
    #gamma_a = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)#, 0.8, 0.9, 1.0)
    #epsilon_a = (0.02, 0.05, 0.07, 0.1, 0.13, 0.15, 0.18, 0.2)

    #2nd run
    #alpha_a = (0.3, 0.4, 0.5, 0.6, 0.7)
    #gamma_a = (0.05, 0.07, 0.1, 0.2, 0.3)
    #epsilon_a = (0.02, 0.05, 0.07, 0.1, 0.13)

    #3rd run
    #alpha_a = (0.4, 0.5, 0.6)
    #gamma_a = (0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5)
    #epsilon_a = (0.05, 0.07, 0.1)

    #4th run
    alpha_a = (0.4, 0.5)
    gamma_a = (0.07, 0.1, 0.2, 0.3)
    epsilon_a = (0.05, 0.07)

    total_iti = len(alpha_a) * len(gamma_a) * len(epsilon_a)

    optimal_array = [[0 for x in range(total_iti)] for x in range(2)]

    trail_n = 100
    opt_m = 0

    n = 0

    for i in range(len(alpha_a)):
        for j in range(len(gamma_a)):
            for k in range(len(epsilon_a)):
                alpha = alpha_a[i]
                gamma = gamma_a[j]
                epsilon = epsilon_a[k]

                # Set up environment and agent
                e = Environment()  # create environment (also adds some dummy traffic)
                a = e.create_agent(LearningAgent, alpha, gamma, epsilon)  # create agent
                e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

                # Now simulate it
                sim = Simulator(e, update_delay=0.0001)  # reduce update_delay to speed up simulation
                sim.run(n_trials=trail_n)  # press Esc or close pygame window to quit

                mean = np.mean(a.trial_array[0][trail_n/2:trail_n])

                optimal_array[0][n] = (alpha, gamma, epsilon)
                optimal_array[1][n] = round(mean, 2)

                n += 1

                print "Alpah: "+str(alpha)
                print "Gamma: "+str(gamma)
                print "Epsilon: "+str(epsilon)

                print a.trial_array[0]
                print a.trial_array[1]

    print optimal_array[0]
    print optimal_array[1]

    optimal_avg_steps = min(optimal_array[1])

    for m in range(len(optimal_array[1])):
        if optimal_array[1][m] == optimal_avg_steps:
            opt_m = m
            break

    print "Total combinations: "+str(total_iti)
    print "Optimal Combination of Alpha, Gamma and Epsilon: "+str(optimal_array[0][opt_m])
    print "Optimal Average Steps: "+str(optimal_avg_steps)

if __name__ == '__main__':
    run()
