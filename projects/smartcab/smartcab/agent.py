import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from random import randint

from random import randrange, uniform

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor
       
        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.count = 1
        #self.testing = False
        #valid_actions = [None, 'forward', 'left', 'right']
        #self.valid_inputs = {'light': TrafficLight.valid_states, 'oncoming': valid_actions, 'left': valid_actions, 'right': valid_actions}
        self.valid_actions=[None, 'forward', 'left', 'right']


    def __init_q_table(self):
        self.q_table = {}


    def reset(self, destination=None, testing=False):
        """ The res`et function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        if(self.learning ==True):
            #for not optimised
            #self.epsilon = math.fabs(self.epsilon - 0.05  ) # Random exploration factor
            #self.alpha = self.alpha      # Learning factor
            #for optimised version
            self.epsilon = self.alpha ** self.count
            self.count += 1
       

        if(testing ==True):
            self.epsilon = 0   # Random exploration factor
            self.alpha = 0       # Learning factor
        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        
        # NOTE : you are not allowed to engineer features outside of the inputs available.
        # Because the aim of this project is to teach Reinforcement Learning, we have placed 
        # constraints in order for you to learn how to adjust epsilon and alpha, and thus learn about the balance between exploration and exploitation.
        # With the hand-engineered features, this learning process gets entirely negated.
        
        # Set 'state' as a tuple of relevant data for the agent        
        #state = None


        state=(waypoint,inputs['light'], inputs['oncoming'], inputs['left'],inputs['right'])
        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state

        #maxQ = None
        all_actions_inCurrentState=self.Q[state]
        maxq_action= None
        maxq= 0.0
        lstOfTiedMaxActions=[]
        for acts , qval in all_actions_inCurrentState.iteritems() :
            if qval >maxq :
                maxq= qval
                maxq_action=acts
            else: 
                 pass

        # once we have found maxQ , we need to find tie up actions having same max Q
        for acts , qval in all_actions_inCurrentState.iteritems() :
            if qval  == maxq :
               
                lstOfTiedMaxActions=lstOfTiedMaxActions+[(acts,qval)]
            else: 
                 pass 

        #return maxq_action from list of maxQ actions randomly
        return random.choice(lstOfTiedMaxActions)[0]


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        if (self.learning== True):
            if state in self.Q:
                pass
            else:
                self.Q[state]={None:0.0, 'forward':0.0, 'left':0.0, 'right':0.0}
        else:
            pass

        return


    def updateQ(self, state , action , new_q):
      
        if (self.learning== True):
                self.Q[state][action] = new_q
        
        return

    def getQ(self, state , action):
       
        return self.Q[state][action]
       


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        #action = None
        #learning =True
        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        # Otherwise, choose an action with the highest Q-value for the current state
        # Be sure that when choosing an action with highest Q-value that you randomly select between actions that "tie".

        # for not learning 
        if(self.learning!= True):
            action = self.getRandomAction()
            return action
        else:
            # for  learning
            # uniform gives you a floating-point value
            frand = uniform(0, 1)
            if frand < self.epsilon:
                action =  self.getRandomAction() 
                return action
                
            else:
                if state in self.Q:
                    return self.get_maxQ(state)[1]
                else:
                    self.createQ(state)
                    return self.getRandomAction()


    def getRandomAction(self):
        
        a= randint(0, 3)
        print(a)
        return self.valid_actions[a]


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')

         # for not learning 
        if(self.learning== True):
            #new_q = old_q*(1 - self.alpha) + self.alpha*(reward + self.gamma * max_state2_q)
            old_q=self.getQ(state, action)
            new_q = old_q*(1 - self.alpha) + self.alpha*(reward )
            self.updateQ(state , action , new_q)
        else:
            pass

        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(verbose=True,num_dummies= 100 ,grid_size=(8,6)  )
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent,learning= False , epsilon=1.0, alpha=0.99 )
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent , enforce_deadline= True )

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env,size=None, update_delay=0.01 , display= True , log_metrics=True, optimized= False )
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(tolerance=0.005, n_test=10)
    #sim.run()

if __name__ == '__main__':
    for i in range(1):
        run()
    
