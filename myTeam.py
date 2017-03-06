# myTeam.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
from __future__ import division
from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game


#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first='DummyAgent', second='DummyAgent', **kwargs):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.

    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

weights = util.Counter()

class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    # ### The below code is going to be called by the game ### #

    def __init__(self, *args, **kwargs):
        '''
        Initialize agent
        '''
        CaptureAgent.__init__(self, *args, **kwargs)

        # Initialize all weights
        self.weights = util.Counter()

        self.learning_rate = .2
        self.exploration_rate = .05
        self.discount_factor = .99
        self.training = True

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        # Your initialization code goes here, if you need any.
        self.action_list = [None]
        self.initial_food = self.getFood(gameState).count()
        self.initial_defending_food = self.getFoodYouAreDefending(gameState).count()

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """

        # Observe the current state, and update our weights based on the current state and any reward we might get
        #

        previousState = self.getPreviousObservation()
        previousAction = self.getPreviousAction()
        reward = self.getReward(gameState)
        self.update(previousState, previousAction, gameState, reward)

        if random.random() < self.exploration_rate:
            action = random.choice(gameState.getLegalActions(self.index))
        else:
            action = self.getPolicy(gameState)

        print 'Features', self.getFeatures(gameState, action)
        print
        print 'Weights', self.getWeights()
        print
        print

        return action

    # ### The below code just belongs to us ### #

    def getReward(self, gameState):
        if self.getPreviousObservation() is None:
            return 0

        reward = 0
        previousState = self.getPreviousObservation()

        # Find out if we got a pac dot. If we did, add 10 points.
        previousFoodNum = self.getFood(previousState).count()
        foodNum = self.getFood(gameState).count()

        reward += 10 * (previousFoodNum - foodNum)

        return reward

    def update(self, gameState, action, nextState, reward):
        """
        Updates our values. Called on getAction
        """

        # Don't do anything on first iteration
        if self.getPreviousObservation() is None:
            return

        correction = reward + self.discount_factor * self.getValue(nextState) - self.getQValue(gameState, action)
        features = self.getFeatures(gameState, action)

        for weight_name in self.getWeights():
            self.getWeights()[weight_name] += self.learning_rate * correction * features[weight_name]

    def getQValue(self, gameState, action):
        """
        Returns the value we think the given action will give us.
        """

        features = self.getFeatures(gameState, action)

        return sum(features[feature] * self.getWeights()[feature] for feature in features)

    def getValue(self, gameState):
        """
        Returns the value we're giving the current state.
        """
        return max([self.getQValue(gameState, action) for action in gameState.getLegalActions(self.index)] or [0])

    def getPolicy(self, gameState):
        """
        Returns the best action to take in this state.
        """
        legalActions = list(gameState.getLegalActions(self.index))
        # Shuffle to break ties randomly
        random.shuffle(legalActions)
        return max(legalActions or [None], key=lambda action: self.getQValue(gameState, action))

    def getFeatures(self, gameState, action):
        '''
        Returns a list of features.
        IMPORTANT: These should all return values between 0 and 1. For all new features, try to normalize them.
        '''

        # TODO: Incorporate "action" into these features (as right now they only take into account the state)
        # TODO: Add more features / figure out whihc features are important
        features = util.Counter()
        x, y = gameState.getAgentPosition(self.index)
        dx, dy = Actions.directionToVector(action) if action is not None else (0, 0)
        next_x, next_y = x + dx, y + dy

        walls = gameState.getWalls()
        mazeSize = walls.width + walls.height

        closestFood = min(self.getMazeDistance((next_x, next_y), food) for food in self.getFood(gameState).asList())

        features['num_food'] = self.getFood(gameState).count() / self.initial_food
        features['num_defending_food'] = self.getFoodYouAreDefending(gameState).count() / self.initial_defending_food
        features['bias'] = 1.0
        features['score'] = self.getScore(gameState)
        features['closest_food'] = closestFood / mazeSize

        # Distance away from opponents
        agentDistances = gameState.getAgentDistances()

        for agent_num in self.getOpponents(gameState):
            features['opponent_' + str(agent_num) + '_distance'] = agentDistances[agent_num] / mazeSize

        return features

    def getPreviousAction(self):
        return self.action_list[-1]

    def getWeights(self):
        if self.training:
            return weights
        else:
            return self.weights
