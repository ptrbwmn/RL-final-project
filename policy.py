import numpy as np

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q.copy()
        self.epsilon = epsilon
        self.state_count = np.zeros((Q.shape[0])).astype(np.float32)
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        #self.state_count[obs]+=1
        #epsilon = self.epsilon * 0.9**(self.state_count[obs])
        epsilon = self.epsilon
        num_actions = self.Q.shape[1]
        greedy = np.random.choice(2,1,p=[epsilon, 1-epsilon])
        if greedy:
            max_actions = np.max(self.Q[obs])
            max_action_idc = np.where(self.Q[obs]==max_actions)
            action = np.random.choice(max_action_idc[0])
        else:
            action = np.random.choice(num_actions,1,p=np.ones(num_actions)/num_actions)[0]
        return action

class EpsilonGreedyPolicy_Double_Q(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q1, Q2, epsilon):
        self.Q1 = Q1.copy()
        self.Q2 = Q2.copy()
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        epsilon = self.epsilon
        num_actions = self.Q1.shape[1]
        greedy = np.random.choice(2,1,p=[epsilon, 1-epsilon])
        if greedy:
            sum_Q1_Q2 = self.Q1 + self.Q2
            max_actions = np.max(sum_Q1_Q2[obs])
            max_action_idc = np.where(sum_Q1_Q2[obs]==max_actions)
            action = np.random.choice(max_action_idc[0])
        else:
            action = np.random.choice(num_actions,1,p=np.ones(num_actions)/num_actions)[0]
        return action