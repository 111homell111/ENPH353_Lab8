import random
import pickle
import csv

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

        self.best_reward = float('-inf')  # Track the highest reward
        self.best_q = {}  # Store the Q-values for the best policy

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        
        # TODO: Implement loading Q values from pickle file
        try:
            with open(filename + ".pickle", "rb") as f:
                self.q = pickle.load(f)
            print("Loaded file: {}".format(filename + ".pickle"))
        except FileNotFoundError:
            print("File not found. Starting with an empty Q-table.")
            #self.q = {}

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.
        with open(filename + ".pickle", "wb") as f:
            pickle.dump(self.q, f)

        print("Wrote to file: {}".format(filename+".pickle"))

        # Save to a CSV file
        with open(filename + ".csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write headers
            writer.writerow(['State', 'Action', 'Q-value'])
            
            # Write Q-values as rows in the CSV file
            for (state, action), q_value in self.q.items():
                writer.writerow([state, action, q_value])

        print("Also wrote to file: {}".format(filename + ".csv"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action


        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            action = random.choice(self.actions)
            q = self.getQ(state, action)
        else:
            # Exploit: choose the action with the highest Q value
            q_values = [self.getQ(state, a) for a in self.actions]
            max_q = max(q_values)
            
            # Get all actions with the max Q-value (in case of ties)
            best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
            
            # Randomly select among the best actions if there are ties
            action = random.choice(best_actions)
            q = max_q
        
        if return_q:
            return action, q
        else:
            return action

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # Get current Q-value for (state1, action1)
        current_q = self.getQ(state1, action1)
        
        # Get the max Q-value for the next state (state2)
        future_q = max([self.getQ(state2, a) for a in self.actions])
        
        # Bellman update equation: 
        # Q(s1, a1) += alpha * [reward + gamma * max(Q(s2)) - Q(s1, a1)]
        new_q = current_q + self.alpha * (reward + self.gamma * future_q - current_q)
        
        # Update the Q-value for (state1, action1)
        self.q[(state1, action1)] = new_q

        if reward > self.best_reward:
            self.best_reward = reward
            self.best_q = self.q.copy()  # Store the Q-table for the best policy
            self.saveQ('best_policy')
            print(f"New best policy saved with reward: {self.best_reward}")

    def loadBestPolicy(self):
        '''
        Load the best policy from a saved file.
        '''
        try:
            with open("best_policy.pickle", "rb") as f:
                self.best_q = pickle.load(f)
            self.q = self.best_q.copy()  # Load best policy into current Q-table
            print("Loaded best policy with reward: {}".format(self.best_reward))
        except FileNotFoundError:
            print("No best policy saved yet.")


