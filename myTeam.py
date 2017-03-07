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

    return [eval(first)(firstIndex, **kwargs), eval(second)(secondIndex, **kwargs)]

##########
# Agents #
##########


weights = util.Counter()
weights['closest_food_aStar'] = -1


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

        # Initialize all weights
        self.gameNumber = 0
        self.weights = util.Counter()
        self.numTraining = kwargs.pop('numTraining', 0)
        self.training_exploration_rate = .5
        self.testing_exploration_rate = .05

        self.learning_rate = .2
        self.exploration_rate = .8
        self.discount_factor = .99

        # Manually put in weights here
        if self.numTraining == 0:
            self.weights.update({'score': 0.6809099995971538, 'num_defending_food': 23.6565508664964, 'opponent_0_distance': 2.0699359632136902, 'num_food': 23.633853866509785, 'bias': 115.8643705168336, 'opponent_2_distance': 1.9917190914963816, 'closest_food_aStar': -1.9670769570603142})

        CaptureAgent.__init__(self, *args, **kwargs)

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
        self.gameNumber += 1

        # Your initialization code goes here, if you need any.
        self.action_list = [Directions.STOP]
        self.initial_food = self.getFood(gameState).count()
        self.initial_defending_food = self.getFoodYouAreDefending(gameState).count()

        self.exploration_rate = self.training_exploration_rate if self.isTraining() else self.testing_exploration_rate

        # Copy global weights to local once we stop training
        # if self.gameNumber == self.numTraining + 1:
        #     self.weights = weights
        self.weights['closest_food_aStar'] = -1

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

        if self.isTraining():
            print 'Features', self.getFeatures(gameState, action)
            print
            print 'Weights', self.weights
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

        for weight_name in features:
            self.weights[weight_name] += self.learning_rate * correction * features[weight_name]

    def getQValue(self, gameState, action):
        """
        Returns the value we think the given action will give us.
        """

        features = self.getFeatures(gameState, action)

        return sum(features[feature] * self.weights[feature] for feature in features)

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
        nextGameState = gameState.generateSuccessor(self.index, action)

        walls = gameState.getWalls()
        mazeSize = walls.width * walls.height

        features['num_food'] = self.getFood(gameState).count() / self.initial_food
        features['num_defending_food'] = self.getFoodYouAreDefending(gameState).count() / self.initial_defending_food
        features['bias'] = 1.0
        features['score'] = self.getScore(gameState)
        features['closest_food_aStar'] = len(self.aStarSearch(nextGameState, self.getFood(gameState).asList())) / mazeSize

        # Distance away from opponents
        agentDistances = gameState.getAgentDistances()

        for agent_num in self.getOpponents(gameState):
            features['opponent_' + str(agent_num) + '_distance'] = agentDistances[agent_num] / mazeSize

        return features

    def getPreviousAction(self):
        return self.action_list[-1]

    def isTraining(self):
        return self.gameNumber <= self.numTraining


    # ## A Star Search ## #

    def aStarSearch(self, gameState, goalPositions, agentIndex=None):
        """
        Finds the distance between the agent with the given index and its nearest goalPosition
        """
        if agentIndex is None:
            agentIndex = self.index

        start = gameState.getAgentPosition(agentIndex)

        # If we can't see the agent, return None
        if start is None:
            return None

        # Values are stored a 4-tuples, (State, Position, Path, TotalCost)

        currentPosition, currentState, currentPath, currentTotal = start, gameState, [], 0
        # Priority queue uses the maze distance between the entered point and its closest goal position to decide which comes first
        queue = util.PriorityQueueWithFunction(lambda entry: currentTotal + min(self.getMazeDistance(entry[0], endPoint) for endPoint in goalPositions))

        # Keeps track of visited positions
        visited = set([currentPosition])

        while currentPosition not in goalPositions:
            # print currentPosition
            # print goalPositions
            # print
            possibleActions = currentState.getLegalActions(agentIndex)
            successorStates = [(currentState.generateSuccessor(agentIndex, action), action) for action in possibleActions]

            for state, action in successorStates:
                position = state.getAgentPosition(self.index)
                if position not in visited:
                    visited.add(position)
                    queue.push((position, state, currentPath + [action], currentTotal + 1))
            currentPosition, currentState, currentPath, currentTotal = queue.pop()

        # Check heuristic for consistency #2
        return currentPath
