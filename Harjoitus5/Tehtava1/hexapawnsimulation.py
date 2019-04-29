# -*- coding: utf-8 -*-
import sys
import time
import queue
import random
import copy
import hexapawnstates
import hexapawn
import numpy as np
import argparse as ap

def progress(count, total):
    """
    Tämä funktio tulostaa edistymispalkin simuloidessa peliä eri oppimisnopeuksien arvoilla
    """
    bar_len = 30
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    print('[{}] {}{} \r'.format(bar, percents, '%'), end="")
        
def main():
    parser = ap.ArgumentParser()
    parser.add_argument("-a", "--alpha", help="Value of learning rate", required="True")
    args = vars(parser.parse_args())
    alpha = float(args["alpha"])
    importantstates = [[2,2,2,1,0,0,0,1,1],[2,2,2,0,1,0,1,0,1],[2,0,2,2,1,0,0,0,1],[0,2,2,1,2,0,0,0,1],[0,2,2,0,1,0,0,0,1],[0,2,2,0,1,0,1,0,0],[0,2,0,2,1,1,0,0,0]]
    countlist = []
    times = 100
    for t in range(times):
        count = 0
        progress(t+1, times)
        allstates = hexapawnstates.main()
        game = hexapawn.HexaPawn()
        Q = game.initializeStateActionMap(allstates)
        playerscore, computerscore = 0, 0
        perfectgame = False
        gameover = False
        while True:
            count += 1
            board = [2,2,2,0,0,0,1,1,1]
            gameisgoing = True
            mark = 1
            while gameisgoing:
                if mark == 2:
                    state = board
                    action, Qvalues = game.bestAction(Q, state, mark)
                    board = action
                    stateaction = ((tuple(state),tuple(action)))
                    if game.terminalState(board, game.legalMoves(3-mark, board)):
                        computerscore += 1
                        reward = 5
                        Q[stateaction] = game.QLearning(Q, stateaction, alpha, reward)
                        if state != game.mirrorBoard(state) or action != game.mirrorBoard(action):
                            stateaction = ((tuple(game.mirrorBoard(state)),tuple(game.mirrorBoard(action))))
                            Q[stateaction] = game.QLearning(Q, stateaction, alpha, reward)        
                        gameisgoing = False 
                    mark = 1
                else:
                    legalmoves = game.legalMoves(mark, board)
                    legalpositions = []
                    for i in range(len(legalmoves)):
                        legalpositions.append(legalmoves[i][0])
                    nextmove = random.choice(legalmoves)
                    board = game.generateNewBoard(board, nextmove, mark)
                    if game.terminalState(board, game.legalMoves(3-mark, board)):
                        reward = -5
                        playerscore += 1
                        Q[stateaction] = game.QLearning(Q, stateaction, alpha, reward)
                        if state != game.mirrorBoard(state) or action != game.mirrorBoard(action):
                            stateaction = ((tuple(game.mirrorBoard(state)),tuple(game.mirrorBoard(action))))
                            Q[stateaction] = game.QLearning(Q, stateaction, alpha, reward)
                        gameisgoing = False
                    mark = 2
            if count==500:
                countlist.append(float('Inf'))
                break
            perfectlysolved, _, __, ___ = game.perfectGameChecker(Q, importantstates)
            if perfectlysolved != perfectgame:
                countlist.append(count)
                gameover = True
            perfectgame = perfectlysolved
            if gameover:
                break
    average = sum(countlist)/float(times)
    if average == float('Inf'):
        average = 'infinity'
    print("\nWith learning rate {} was played in average {} games after computer was perfect".format(alpha, average))
    
if __name__ == '__main__':
    main()