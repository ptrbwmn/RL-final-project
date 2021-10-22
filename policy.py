import numpy as np

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon, eps_decay = 0.):
        self.Q = Q.copy()
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.state_count = np.zeros((Q.shape[0])).astype(np.float32)
        self.sa_count = np.zeros(Q.shape).astype(np.float32)

    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        if self.eps_decay > 0:
            self.state_count[obs]+=1
            epsilon = self.epsilon / self.state_count[obs]**self.eps_decay
        else:
            epsilon = self.epsilon
        num_actions = self.Q.shape[1]
        greedy = np.random.choice([False, True], p=[epsilon, 1-epsilon])
        if greedy:
            max_actions = np.max(self.Q[obs])
            max_action_idc = np.where(self.Q[obs]==max_actions)
            action = np.random.choice(max_action_idc[0])
        else:
            action = np.random.choice(np.arange(num_actions))
        self.sa_count[obs,action]+=1
        return action

class EpsilonGreedyPolicy_Double_Q(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q1, Q2, epsilon_0, eps_decay = 0.):
        self.Q1 = Q1.copy()
        self.Q2 = Q2.copy()
        self.epsilon_0 = epsilon_0
        self.eps_decay = eps_decay
        self.state_count = np.zeros((Q1.shape[0])).astype(np.float32)
        self.sa_count1 = np.zeros(Q1.shape).astype(np.float32)
        self.sa_count2 = np.zeros(Q1.shape).astype(np.float32)
        
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        if self.eps_decay > 0:
            self.state_count[obs]+=1
            epsilon = self.epsilon_0 / self.state_count[obs]**self.eps_decay
        else:
            epsilon = self.epsilon_0
        num_actions = self.Q1.shape[1]
        greedy = np.random.choice([False, True], p=[epsilon, 1-epsilon])
        if greedy:
            sum_Q1_Q2 = self.Q1 + self.Q2
            max_actions = np.max(sum_Q1_Q2[obs])
            max_action_idc = np.where(sum_Q1_Q2[obs]==max_actions)
            action = np.random.choice(max_action_idc[0])
        else:
            action = action = np.random.choice(np.arange(num_actions))
        #self.sa_count[obs,action]+=1; handled in double_q_learning function
        return action