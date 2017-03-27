from __future__ import division
import random
import util
import copy
import collections
import math
import tensorflow as tf
from captureAgents import CaptureAgent
from game import Directions, Actions
from numpy import array, float32


#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first='DummyAgent', second='DefenseAgent', **kwargs):
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

winning_weights = [array([[ 3.17576432,  0.22922029, -3.35294485, -1.10840964],
       [ 1.04950893,  1.52935839, -2.19146419, -0.40707612],
       [ 2.30970669,  1.17531323, -3.27446604, -1.43581951],
       [-0.38588887, -1.93801641,  0.51416117,  0.59434551]], dtype=float32), array([[  2.26813102e+00,   2.37682555e-03,  -2.67385650e+00,
         -9.63723719e-01]], dtype=float32), array([[ 0.74997079,  0.61594456,  0.54025191,  0.8074066 ],
       [-0.91277975,  1.72260773, -1.28915024,  0.57378536],
       [ 1.98607254, -1.43602443,  1.8726244 , -2.08518648],
       [ 0.2687358 , -0.43212101,  0.93886507, -0.121199  ]], dtype=float32), array([[-1.6464802 ,  1.20602429, -0.77098143,  1.63822222]], dtype=float32), array([[-11.24650192],
       [ 12.57112789],
       [-13.45398903],
       [ 10.75681114]], dtype=float32), array([[ 12.52836514]], dtype=float32)]

# winning_weights = [array([[ 0.66780835, -4.76155901,  2.36532497,  1.87167597],
#        [ 0.33141476, -2.27728558,  2.23192334,  0.15596049],
#        [ 1.12257552, -3.18644667,  2.07127476,  0.71196592],
#        [ 0.13034312, -0.26017287, -0.37825623, -0.25716582]], dtype=float32), array([[-0.07399458, -3.39847708,  2.14233255,  0.25014722]], dtype=float32), array([[ 2.44381118,  2.06468701, -0.22663018, -0.75896955],
#        [ 0.55912024, -0.20414966, -0.20584878,  0.11649705],
#        [ 0.37813476,  0.75363678, -0.29809994, -1.89197373],
#        [ 1.83026278,  4.00979853,  2.10268021,  0.60413957]], dtype=float32), array([[ 1.78436148,  3.00808477,  0.43807781, -0.19622369]], dtype=float32), array([[ 2.1510303 ],
#        [ 0.59337908],
#        [-2.07018137],
#        [ 2.23792672]], dtype=float32), array([[ 1.45820272]], dtype=float32)]



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
        self.numTraining = kwargs.pop('numTraining', 0)
        self.training_exploration_rate = .7
        self.testing_exploration_rate = 0

        self.discount_factor = .95

        # Manually put in weights here

        self.inputs = tf.placeholder(shape=[1,4], dtype=tf.float32)

        self.weights1 = tf.Variable(tf.truncated_normal([4,4]))

        self.bias1 = tf.Variable(tf.zeros(shape=[1,4]))

        self.weights2 = tf.Variable(tf.truncated_normal([4,4]))

        self.bias2 = tf.Variable(tf.zeros(shape=[1, 4]))

        self.weights3 = tf.Variable(tf.truncated_normal([4,1]))

        self.bias3 = tf.Variable(tf.zeros([1,1]))

        self.layer1 = tf.tanh(tf.matmul(self.inputs, self.weights1) + self.bias1)

        self.layer2 = tf.tanh(tf.matmul(self.layer1, self.weights2) + self.bias2)

        self.layer3 = tf.nn.relu(tf.matmul(self.layer2, self.weights3) + self.bias3)

        self.output_layer = self.layer3

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

        self.nextQ = tf.placeholder(shape=[1,1], dtype=tf.float32)

        self.loss = tf.reduce_sum(tf.square((self.nextQ - self.output_layer)))

        self.trainer = self.optimizer.minimize(self.loss)

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

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


        if winning_weights is not None:
            weights1, bias1, weights2, bias2, weights3, bias3 = winning_weights
            self.sess.run([self.weights1.assign(weights1),
                           self.bias1.assign(bias1),
                           self.weights2.assign(weights2),
                           self.bias2.assign(bias2),
                           self.weights3.assign(weights3),
                           self.bias3.assign(bias3)])

    def chooseAction(self, gameState):
        global winning_weights
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
            print repr(self.weights())
            print
            print

        winning_weights = self.weights()

        return action

    # ### The below code just belongs to us ### #

    def getReward(self, gameState):
        if self.getPreviousObservation() is None:
            return 0

        reward = 0
        previousState = self.getPreviousObservation()



        # Find out if we got a pac dot. If we did, add 10 points.
        previousFood = self.getFood(previousState).asList()
        myPosition = gameState.getAgentPosition(self.index)
        currentFood = self.getFood(gameState).asList()

        if myPosition in previousFood and myPosition not in currentFood:
            reward += 10

        return reward

    def weights(self):
        return (self.sess.run([self.weights1, self.bias1, self.weights2, self.bias2, self.weights3, self.bias3]))


# Here is where we implement the 3-layer network: 2 with sigmoid activations, and 1 straight linear with no activation

    def update(self, gameState, action, nextState, reward):
        """
        Updates our values. Called on getAction
        """

        # Don't do anything on first iteration
        if self.getPreviousObservation() is None:
            return

        nextQ = reward + self.discount_factor * self.getValue(nextState)

        correction = reward + self.discount_factor * self.getValue(nextState) - self.getQValue(gameState, action)
        features = self.getFeatures(gameState, action)

        self.sess.run(self.trainer, feed_dict={self.inputs: [features],
                                                 self.nextQ: nextQ})

    def getQValue(self, gameState, action):
        """
        Returns the value we think the given action will give us.
        """

        features = self.getFeatures(gameState, action)

        return self.sess.run(self.output_layer, feed_dict={self.inputs: [features]})

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
        features = []
        position = gameState.getAgentPosition(self.index)
        nextGameState = gameState.generateSuccessor(self.index, action)
        nextPosition = nextGameState.getAgentPosition(self.index)
        food = self.getFood(gameState)
        capsules = self.getCapsules(gameState)

        walls = gameState.getWalls()
        wallsList = walls.asList()
        mazeSize = walls.width * walls.height

        enemyIndices = self.getOpponents(gameState)

        attackablePacmen = [gameState.getAgentPosition(i) for i in enemyIndices if self.isPacman(gameState, i) and self.isGhost(gameState, self.index) and not self.isScared(gameState, self.index)]
        scaredGhostLocations = [gameState.getAgentPosition(i) for i in self.getOpponents(gameState) if self.isScared(gameState, i) and self.isGhost(gameState, i)]
        goalPositions = set(food.asList() + attackablePacmen + capsules + scaredGhostLocations)

        enemyGhostLocations = [gameState.getAgentPosition(i) for i in enemyIndices if self.isGhost(gameState, i) and not self.isScared(gameState, i)]
        # enemyGhostLocations.extend(self.validSurroundingPositionsTo(gameState, enemyGhostLocations, wallsList))
        # enemyGhostLocations.extend(self.validSurroundingPositionsTo(gameState, enemyGhostLocations, wallsList))

        features.append( food.count() / self.initial_food)
        features.append( self.getFoodYouAreDefending(gameState).count() / self.initial_defending_food)
        closestGhost = len(self.aStarSearch(nextPosition, nextGameState, list(enemyGhostLocations))) / mazeSize if enemyGhostLocations else 1
        features.append(closestGhost)
        # features['score'] = self.getScore(gameState)

        avoidPositions = set(enemyGhostLocations)

        # If we're being chased, avoid positions that we can't get out of easily
        if self.isBeingChased(gameState):
            network, source = self.getFlowNetwork(gameState, startingPositions=self.getMiddlePositions(gameState), defenseOnly=False)
            closePositions = [goalPosition for goalPosition in goalPositions if util.manhattanDistance(position, goalPosition) <= 5]

            # Split into groups for faster movement, so we're not finding the max flow of every dot
            groups = set()
            while closePositions:
                changed = True
                newGroup = [closePositions.pop()]
                while changed:
                    changed = False
                    for groupPosition in newGroup:
                        for closePosition in closePositions:
                            manhattanDistance = util.manhattanDistance(groupPosition, closePosition)
                            if (manhattanDistance <= 1 or  # They're right next to each other
                               (manhattanDistance == 2 and (groupPosition[0] + closePosition[0] // 2, groupPosition[1] + closePosition[1] // 2) not in wallsList)):  # They're across from each other, or katty corner. This doesn't work correctly when they're on the other side of a wall from each other, which rarely if ever happens
                                changed = True
                                closePositions.remove(closePosition)
                                newGroup.append(closePosition)
                groups.add(tuple(newGroup))

            # Only find max flow of dots 5 or less away from us. If max flow is 1 or less, don't go there
            for group in groups:
                if len(self.aStarSearch(position, gameState, [group[0]])) <= 5:
                    maxFlow = network.MaxFlow(source, group[0])[0]
                    network.reset()
                    if maxFlow <= 1:
                        maxFlowFromMe = network.MaxFlow(position, group[0])[0]
                        network.reset()
                        if maxFlowFromMe <= 1:
                            avoidPositions.update(group)

        aStar_food_path = self.aStarSearch(nextPosition, nextGameState, list(goalPositions), avoidPositions=avoidPositions)

        features.append((len(aStar_food_path) if aStar_food_path is not None else mazeSize) / mazeSize)

        return features

    def getPreviousAction(self):
        return self.action_list[-1]

    def isTraining(self):
        return self.gameNumber <= self.numTraining

    def isGhost(self, gameState, index):
        """
        Returns true ONLY if we can see the agent and it's definitely a ghost
        """
        position = gameState.getAgentPosition(index)
        if position is None:
            return False
        return not (gameState.isOnRedTeam(index) ^ (position[0] < gameState.getWalls().width / 2))

    def isScared(self, gameState, index):
        """
        Says whether or not the given agent is scared
        """
        isScared = bool(gameState.data.agentStates[index].scaredTimer)
        return isScared


    def isPacman(self, gameState, index):
        """
        Returns true ONLY if we can see the agent and it's definitely a pacman
        """
        position = gameState.getAgentPosition(index)
        if position is None:
            return False
        return not (gameState.isOnRedTeam(index) ^ (position[0] >= gameState.getWalls().width / 2))

    def isBeingChased(self, gameState):
        """
        If we are pacman and the enemy ghost is within aStar distance less than 5
        """
        if (not self.isPacman(gameState, self.index)) or any(self.isScared(gameState, index) for index in self.getOpponents(gameState)):
            return False
        else:
            myPosition = gameState.getAgentPosition(self.index)

            agentLocations = [gameState.getAgentPosition(i) for i in self.getOpponents(gameState)]
            agentDistances = [len(self.aStarSearch(myPosition, gameState, [agentLocation])) for agentLocation in agentLocations if agentLocation is not None]
            return min(agentDistances or [float('inf')]) <= 7

    def validSurroundingPositionsTo(self, gameState, positions, walls):
        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
        actionVectors = [Actions.directionToVector(action) for action in actions]
        # Change action vectors to integers so they work correctly with indexing
        actionVectors = [tuple(int(number) for number in vector) for vector in actionVectors]

        possibleNextPositions = [((position[0] + vector[0], position[1] + vector[1])) for vector in actionVectors for position in positions]
        validNextPositions = [position for position in possibleNextPositions if position not in walls]

        return validNextPositions

    # ## A Star Search ## #

    def aStarSearch(self, startPosition, gameState, goalPositions, avoidPositions=[], returnPosition=False):
        """
        Finds the distance between the agent with the given index and its nearest goalPosition
        """
        walls = gameState.getWalls()
        width = walls.width
        height = walls.height
        walls = walls.asList()

        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        actionVectors = [Actions.directionToVector(action) for action in actions]
        # Change action vectors to integers so they work correctly with indexing
        actionVectors = [tuple(int(number) for number in vector) for vector in actionVectors]

        # Values are stored a 3-tuples, (Position, Path, TotalCost)

        currentPosition, currentPath, currentTotal = startPosition, [], 0
        # Priority queue uses the maze distance between the entered point and its closest goal position to decide which comes first
        queue = util.PriorityQueueWithFunction(lambda entry: entry[2] +   # Total cost so far
                                               width * height if entry[0] in avoidPositions else 0 +  # Avoid enemy locations like the plague
                                               min(util.manhattanDistance(entry[0], endPosition) for endPosition in goalPositions))

        # Keeps track of visited positions
        visited = set([currentPosition])

        while currentPosition not in goalPositions:

            possiblePositions = [((currentPosition[0] + vector[0], currentPosition[1] + vector[1]), action) for vector, action in zip(actionVectors, actions)]
            legalPositions = [(position, action) for position, action in possiblePositions if position not in walls]

            for position, action in legalPositions:
                if position not in visited:
                    visited.add(position)
                    queue.push((position, currentPath + [action], currentTotal + 1))

            # This shouldn't ever happen...But just in case...
            if len(queue.heap) == 0:
                return None
            else:
                currentPosition, currentPath, currentTotal = queue.pop()

        if returnPosition:
            return currentPath, currentPosition
        else:
            return currentPath

    def positionIsHome(self, position, gameWidth):
        isHome = not (self.red ^ (position[0] < gameWidth / 2))
        return isHome

    def getFlowNetwork(self, gameState, startingPositions=None, endingPositions=None, defenseOnly=True):
        '''
        Returns the flow network.
        If starting positions are provided, also returns the source node
        If ending positions are provided, also returns the sink node
        Note: Always returns tuple
        '''
        source = (-1, -1)
        sink = (-2, -2)

        walls = gameState.getWalls()
        wallPositions = walls.asList()
        possiblePositions = [(x, y) for x in range(walls.width) for y in range(walls.height) if (x, y) not in wallPositions and (not defenseOnly or self.positionIsHome((x, y), walls.width))]

        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        actionVectors = [Actions.directionToVector(action) for action in actions]
        # Change vectors from float to int
        actionVectors = [tuple(int(number) for number in vector) for vector in actionVectors]

        # Make source and sink

        network = FlowNetwork()

        # Add all vertices
        for position in possiblePositions:
            network.AddVertex(position)
        network.AddVertex(source)
        network.AddVertex(sink)

        # Add normal edges
        edges = EdgeDict()
        for position in possiblePositions:
            for vector in actionVectors:
                newPosition = (position[0] + vector[0], position[1] + vector[1])
                if newPosition in possiblePositions:
                    edges[(position, newPosition)] = 1

        # Add edges attached to source
        for position in startingPositions or []:
            edges[(source, position)] = float('inf')

        for position in endingPositions or []:
            edges[(position, sink)] = float('inf')

        for edge in edges:
            network.AddEdge(edge[0], edge[1], edges[edge])

        retval = (network,)

        if startingPositions is not None:
            retval = retval + (source,)
        if endingPositions is not None:
            retval = tuple(retval) + (sink,)

        return retval

    def findBottleneckWithMostPacdots(self, gameState):

        startingPositions = self.getMiddlePositions(gameState)
        endingPositions = self.getFoodYouAreDefending(gameState).asList()
        network, source = self.getFlowNetwork(gameState, startingPositions=startingPositions)

        bottleneckCounter = collections.Counter()

        for dot in endingPositions:
            bottlenecks = network.FindBottlenecks(source, dot)
            if len(bottlenecks) == 1:
                bottleneckCounter[bottlenecks[0]] += 1
            network.reset()

        maxBottleneck = max(bottleneckCounter or [None], key=lambda vertex: bottleneckCounter[vertex])
        return maxBottleneck, bottleneckCounter[maxBottleneck]

    def getMiddlePositions(self, gameState):

        # Find the positions closest to the moiddle line so we can start there
        walls = gameState.getWalls()
        wallPositions = walls.asList()
        possiblePositions = [(x, y) for x in range(walls.width) for y in range(walls.height) if (x, y) not in wallPositions and self.positionIsHome((x, y), walls.width)]
        startX = walls.width / 2 - 1 if self.red else walls.width / 2
        startingPositions = [position for position in possiblePositions if position[0] == startX]
        return startingPositions


# ### Implementation of Ford-Fulkerson algorithm, taken from https://github.com/bigbighd604/Python/blob/master/graph/Ford-Fulkerson.py and heavily modified


class Edge(object):
    def __init__(self, u, v, w):
        self.source = u
        self.target = v
        self.capacity = w

    def __repr__(self):
        return "%s->%s:%s" % (self.source, self.target, self.capacity)

    def __eq__(self, other):
        return self.source == other.source and self.target == other.target


class FlowNetwork(object):
    def __init__(self):
        self.adj = {}
        self.flow = {}

    def AddVertex(self, vertex):
        self.adj[vertex] = []

    def GetEdges(self, v):
        return self.adj[v]

    def AddEdge(self, u, v, w=0):
        if u == v:
            raise ValueError("u == v")
        edge = Edge(u, v, w)
        redge = Edge(v, u, w)
        edge.redge = redge
        redge.redge = edge
        self.adj[u].append(edge)
        self.adj[v].append(redge)
        # Intialize all flows to zero
        self.flow[edge] = 0
        self.flow[redge] = 0

    def FindPath(self, source, target):

        currentVertex, currentPath, currentTotal = source, [], 0
        # Priority queue uses the maze distance between the entered point and its closest goal position to decide which comes first
        queue = util.PriorityQueueWithFunction(lambda entry: entry[2] + util.manhattanDistance(entry[0], target))

        visited = set()

        # Keeps track of visited positions
        while currentVertex != target:

            possibleVertices = [(edge.target, edge) for edge in self.GetEdges(currentVertex)]

            for vertex, edge in possibleVertices:
                residual = edge.capacity - self.flow[edge]
                if residual > 0 and not (edge, residual) in currentPath and (edge, residual) not in visited:
                    visited.add((edge, residual))
                    queue.push((vertex, currentPath + [(edge, residual)], currentTotal + 1))

            if queue.isEmpty():
                return None
            else:
                currentVertex, currentPath, currentTotal = queue.pop()

        return currentPath

    def FindBottlenecks(self, source, target):
        maxflow, leadingEdges = self.MaxFlow(source, target)
        paths = leadingEdges.values()

        bottlenecks = []
        for path in paths:
            for edge, residual in path:
                # Save the flows so we don't mess up the operation between path findings
                if self.FindPath(source, edge.target) is None:
                    bottlenecks.append(edge.source)
                    break
        assert len(bottlenecks) == maxflow
        return bottlenecks

    def MaxFlow(self, source, target):
        # This keeps track of paths that go to our destination
        leadingEdges = {}
        path = self.FindPath(source, target)
        while path:
            leadingEdges[path[0]] = path
            flow = min(res for edge, res in path)
            for edge, res in path:
                self.flow[edge] += flow
                self.flow[edge.redge] -= flow

            path = self.FindPath(source, target)
        maxflow = sum([self.flow[edge] for edge in self.GetEdges(source)])
        return maxflow, leadingEdges

    def reset(self):
        for edge in self.flow:
            self.flow[edge] = 0


class EdgeDict(dict):
    '''
    Keeps a list of undirected edges. Doesn't matter what order you add them in.
    '''
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getitem__(self, key):
        return dict.__getitem__(self, tuple(sorted(key)))

    def __setitem__(self, key, val):
        return dict.__setitem__(self, tuple(sorted(key)), val)

    def __contains__(self, key):
        return dict.__contains__(self, tuple(sorted(key)))

    def getAdjacentPositions(self, key):
        edgesContainingKey = [edge for edge in self if key in edge]
        adjacentPositions = [[position for position in edge if position != key][0] for edge in edgesContainingKey]
        return adjacentPositions



class DefenseAgent(DummyAgent):
    def __init__(self, *args, **kwargs):
        self.defenseMode = False
        self.GoToSpot = None
        DummyAgent.__init__(self, *args, **kwargs)

    def registerInitialState(self, gameState):

        DummyAgent.registerInitialState(self, gameState)
        self.checkForBottleneck(gameState)

    def chooseAction(self, gameState):
        # If we were scared and aren't anymore, re-check for bottleneck
        if self.getPreviousObservation():
            if self.isScared(self.getPreviousObservation(), self.index) and not self.isScared(gameState, self.index):
                self.checkForBottleneck(gameState)

        if self.defenseMode and not self.isScared(gameState, self.index):
            position = gameState.getAgentPosition(self.index)
            opponentPositions = [gameState.getAgentPosition(i) for i in self.getOpponents(gameState) if self.isPacman(gameState, i)]
            if opponentPositions:
                pathToOpponents = self.aStarSearch(position, gameState, opponentPositions)
                if (len(pathToOpponents) % 2 == 1 and not  # We want the path length to be odd
                   (len(self.aStarSearch(self.GoToSpot, gameState, [position])) < len(self.aStarSearch(self.GoToSpot, gameState, opponentPositions)))):  # We want to be closer to our spot than they are
                    return pathToOpponents[0]

            pathToSpot = self.aStarSearch(position, gameState, opponentPositions or [self.GoToSpot]) or [Directions.STOP]
            # Paths an odd distance away have a better chance of working
            return pathToSpot[0]
        return DummyAgent.chooseAction(self, gameState)

    def checkForBottleneck(self, gameState):
        bottleneckPosition, numDots = self.findBottleneckWithMostPacdots(gameState)
        if numDots >= 2:
            self.defenseMode = True
            self.GoToSpot = bottleneckPosition
        else:
            self.defenseMode = False
            self.goToSpot = None
