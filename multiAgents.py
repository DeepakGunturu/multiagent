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

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        distFromGhost = 0
        nearGhost = "Not near"
        res =  -(currentGameState.getScore()-successorGameState.getScore())
        idx = 0
        tmp = len(newFood.asList())

        # Going through the ghost positions
        for i in successorGameState.getGhostPositions():

            # Checks if the ghost is near Pacman
            if newScaredTimes[idx] == 0:
              if manhattanDistance(i, newPos) < 3:
                nearGhost = "Near"
            idx += 1
        
        # If the action is "Stop", the penalty is stored in the result
        if action == "Stop":
            res -= 500
        
        # If the ghost is near, return the result
        if nearGhost == "Near":
            return res
    
        # Case for when the ghost is not near
        else:
          if tmp > 0:
              distance, closestFood = min([(manhattanDistance(newPos, food), food) for food in newFood.asList()])
              tmp2 = distance//0.5
              if distance == 0:
                  distFromGhost = tmp2
                
              else:
                  distFromGhost = -tmp2
          
          # Add to the scoring function if the current food pellets is more than the successor
          if currentGameState.getNumFood() > successorGameState.getNumFood():
              res += 500
          
          # Adding distance from the ghost to the scoring value
          res += distFromGhost

        return res

def scoreEvaluationFunction(currentGameState):
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
    # The minimax agent evaluates the game state and returns the max or min 
    def minimaxValue(self, gameState, dep, nextAgent):

        # If the depth of the tree is reached or if the game is won or if the game is lost, the 
        if dep == self.depth:
            return self.evaluationFunction(gameState)
        
        if gameState.isWin():
            return self.evaluationFunction(gameState)

        if gameState.isLose():
            return self.evaluationFunction(gameState)

        # If the counter of the next agent is 0, then the maximum value is returned from the function
        if nextAgent == 0:
            return self.alpha(gameState,dep)
        
        # If the counter of the next agent is not 0, then the minimum value is returned from the function
        else:
            return self.beta(gameState,dep,nextAgent)

    # Maximum agent function
    def alpha(self, gameState, dep):
        maxVal = float("-inf")
        for action in gameState.getLegalActions():
            if maxVal < self.minimaxValue(gameState.generateSuccessor(0, action), dep, 1):
                maxVal = self.minimaxValue(gameState.generateSuccessor(0, action), dep, 1)

        return maxVal

    # Minimum agent function
    def beta(self, gameState, dep, nextAgent):
        minVal = float("inf")
        for action in gameState.getLegalActions(nextAgent):
            if nextAgent+1 == gameState.getNumAgents():
                if minVal > self.minimaxValue(gameState.generateSuccessor(nextAgent, action), dep+1, 0):
                    minVal = self.minimaxValue(gameState.generateSuccessor(nextAgent, action), dep+1, 0)
            else:
                if minVal > self.minimaxValue(gameState.generateSuccessor(nextAgent, action), dep, nextAgent+1):
                    minVal = self.minimaxValue(gameState.generateSuccessor(nextAgent, action), dep, nextAgent+1)
        return minVal
        
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(nextAgent):
            Returns a list of legal actions for an agent
            nextAgent=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(nextAgent, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        # Goes through the possible actions in the current game state and returns the action based on the value (either maximum or min)
        maxVal = float("-inf")
        maxAction = None
        for action in gameState.getLegalActions(0):
            if self.minimaxValue(gameState.generateSuccessor(0, action), 0, 1) > maxVal:
                maxVal =  self.minimaxValue(gameState.generateSuccessor(0, action), 0, 1)
                maxAction = action
        return maxAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    # Minimax agent with alpha beta pruning 
    def minimaxValue(self, gameState, dep, nextAgent, alp, be):
        
        # If the depth is reached
        if dep == self.depth:
            return self.evaluationFunction(gameState)
        
        # If the game is won or lost 
        if gameState.isWin():
          return self.evaluationFunction(gameState)

        if gameState.isLose():
          return self.evaluationFunction(gameState)
        
        # No agents explored
        if nextAgent == 0:
            return self.maxAgent(gameState,dep,alp,be)
        
        # If an agent is being explored
        else:
            return self.minAgent(gameState,dep,nextAgent,alp,be)

    # Agent that returns that maximum values
    def maxAgent(self, gameState, dep, alp, be):
        maxV = float("-inf")
        for action in gameState.getLegalActions(0):   
            if maxV < self.minimaxValue(gameState.generateSuccessor(0, action), dep, 1, alp, be):
              maxV = self.minimaxValue(gameState.generateSuccessor(0, action), dep, 1, alp, be)
            
            else:
              maxV += 0

            if maxV > be:
                return maxV

            if alp < maxV:
              alp = maxV

        return maxV

    # Agent that returns thet minimum values
    def minAgent(self, gameState, dep, nextAgent, alp, be):
        minV = float("inf")
        for action in gameState.getLegalActions(nextAgent):
            if nextAgent+1 == gameState.getNumAgents():
                if minV > self.minimaxValue(gameState.generateSuccessor(nextAgent, action), dep+1, 0, alp, be):
                  minV = self.minimaxValue(gameState.generateSuccessor(nextAgent, action), dep+1, 0, alp, be)

            else:
                if minV > self.minimaxValue(gameState.generateSuccessor(nextAgent, action), dep, nextAgent+1, alp, be):
                  minV = self.minimaxValue(gameState.generateSuccessor(nextAgent, action), dep, nextAgent+1, alp, be)
            
            if minV < alp:
                return minV
        
            if be < minV:
              be += 0

            else:
              be = minV

        return minV

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        maxValue = float("-inf")
        alp = float("-inf")
        be = float("inf")
        maxAction = ""

        # Going through the possible actions of the agent and selecting the maximum values of the agent 
        for action in gameState.getLegalActions(0):
            nextValue = self.minimaxValue(gameState.generateSuccessor(0, action), 0, 1, alp, be)
            if nextValue > maxValue:
                maxValue = nextValue
                maxAction = action
            if alp > maxValue:
              alp += 0
            else:
              alp = maxValue

        return maxAction


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
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction