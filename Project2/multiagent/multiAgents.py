# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        # Generate the successor game state after taking the action
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Calculate the score from the successor state
        score = successorGameState.getScore()

        # Consider the distance to the closest food
        foodList = newFood.asList()
        if foodList:
            foodDistances = [manhattanDistance(newPos, food) for food in foodList]
            closestFoodDistance = min(foodDistances)
            score += 1.0 / closestFoodDistance  # Increase score based on closeness to food

        # Consider ghost distances and their scared state
        for ghostState in newGhostStates:
            ghostDistance = manhattanDistance(newPos, ghostState.getPosition())
            if ghostState.scaredTimer == 0 and ghostDistance < 2:
                score -= 10  # Decrease score significantly if close to a dangerous ghost

        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """

        def minimax(agent, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            if agent == 0:  # Pacman's turn (maximizer)
                return max(minimax(1, depth, gameState.generateSuccessor(agent, action)) for action in
                           gameState.getLegalActions(agent))
            else:  # Ghosts' turn (minimizer)
                next_agent = agent + 1
                if next_agent >= gameState.getNumAgents():
                    next_agent = 0
                    depth -= 1
                return min(minimax(next_agent, depth, gameState.generateSuccessor(agent, action)) for action in
                           gameState.getLegalActions(agent))

        # Start from Pacman, the max player
        maximum = float("-inf")
        best_action = None
        for action in gameState.getLegalActions(0):
            value = minimax(1, self.depth, gameState.generateSuccessor(0, action))
            if value > maximum:
                maximum = value
                best_action = action
        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the action using self.depth and self.evaluationFunction
        """
        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            numAgents = gameState.getNumAgents()
            if agentIndex == 0:  # Pacman's turn (maximizer)
                value = float("-inf")
                for action in gameState.getLegalActions(agentIndex):
                    value = max(value, alphaBeta(1, depth, gameState.generateSuccessor(agentIndex, action), alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else:  # Ghosts' turn (minimizer)
                nextAgent = agentIndex + 1
                if nextAgent >= numAgents:
                    nextAgent = 0
                    depth -= 1
                value = float("inf")
                for action in gameState.getLegalActions(agentIndex):
                    value = min(value, alphaBeta(nextAgent, depth, gameState.generateSuccessor(agentIndex, action), alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

        alpha = float("-inf")
        beta = float("inf")
        bestAction = None
        for action in gameState.getLegalActions(0):
            value = alphaBeta(1, self.depth, gameState.generateSuccessor(0, action), alpha, beta)
            if value > alpha:
                alpha = value
                bestAction = action
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        def expectimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            # Calculate the next agent index and depth
            numAgents = gameState.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth - 1 if nextAgent == 0 else depth

            # Get legal actions
            actions = gameState.getLegalActions(agentIndex)

            if not actions:
                return self.evaluationFunction(gameState)

            # Expectimax calculations
            if agentIndex == 0:  # Pacman's turn (maximizer)
                return max(expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in actions)
            else:  # Ghosts' turn (expectimizer)
                return sum(expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in actions) / len(actions)

        # Start from Pacman, the maximizer, at the current depth level
        bestAction = max(gameState.getLegalActions(0), key=lambda action: expectimax(1, self.depth, gameState.generateSuccessor(0, action)))

        return bestAction


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # Setup information to be used as arguments in evaluation function
    pacman_position = currentGameState.getPacmanPosition()
    ghost_positions = currentGameState.getGhostPositions()

    food_list = currentGameState.getFood().asList()
    food_count = len(food_list)
    capsule_count = len(currentGameState.getCapsules())
    closest_food = 1

    game_score = currentGameState.getScore()

    # Find distances from pacman to all food
    food_distances = [manhattanDistance(pacman_position, food_position) for food_position in food_list]

    # Set value for closest food if there is still food left
    if food_count > 0:
        closest_food = min(food_distances)

    # Find distances from pacman to ghost(s)
    for ghost_position in ghost_positions:
        ghost_distance = manhattanDistance(pacman_position, ghost_position)

        # If ghost is too close to pacman, prioritize escaping instead of eating the closest food
        # by resetting the value for closest distance to food
        if ghost_distance < 2:
            closest_food = 99999

    features = [1.0 / closest_food,game_score, food_count,capsule_count]

    weights = [10,  200,-100, -10]

    # Linear combination of features
    return sum([feature * weight for feature, weight in zip(features, weights)])

# Abbreviation
better = betterEvaluationFunction
