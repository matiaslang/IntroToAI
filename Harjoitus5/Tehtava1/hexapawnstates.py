# -*- coding: utf-8 -*-
import time
import queue
import random
import copy

class HexaPawn:
    """
    Etsitään kaikki mahdolliset tilat pelille Hexapawn
    """
    def legalMoves(self, mark, board):
        """
        Tämä funktio ratkaisee lailliset siirrot pelilaudalle
        """
        newboard = copy.deepcopy(board)
        if mark == 2:
            newboard.reverse()
        values = [-4, -3, -2]
        legalmoves = []
        positions = [i for i, e in enumerate(newboard) if e == mark]
        for position in positions:
            for x in values:
                if 0 <= position + x < 6:
                    if x == -2 and (position==8 or position==5):
                        continue
                    if x == -4 and (position==6 or position==3):
                        continue
                    if x == -3 and ((newboard[position] == 2 and newboard[position+x] == 1) or (newboard[position] == 1 and newboard[position+x] == 2)):
                        continue
                    if x == -2 and newboard[position+x] == 0:
                        continue
                    if x == -4 and newboard[position+x] == 0:
                        continue
                    if newboard[position] == newboard[position+x]:
                        continue
                    if mark == 2:
                        legalmoves.append([8-position, 8-(position+x)])
                    else:
                        legalmoves.append([position, position+x])
        return legalmoves
		
    def generateBoard(self, board, legalmoves, move, mark):
        """
        Tämä funktio generoi uuden pelilaudan annetun siirron perusteella
        """
        newboard = copy.deepcopy(board)
        newboard[legalmoves[move][1]] = newboard[legalmoves[move][0]]
        newboard[legalmoves[move][0]] = 0
        return newboard
		
    def terminalState(self, board):
        """
        Tämä funktio tarkastaa, onko kyseessä päätetila
        """
        if board[0] == 1 or board[1] == 1 or board[2] == 1:
            return True
        if board[6] == 2 or board[7] == 2 or board[8] == 2:
            return True
			
		
def main():
    game = HexaPawn()
    order = [0,1,2,3,4,5,6,7,8]
    board = [2,2,2,0,0,0,1,1,1]
    openlist = [board]
    closedlist = []
    marklist = [1]
    while len(openlist) != 0:
        mark = marklist[0]
        marklist.pop(0)
        current = openlist[0]
        openlist.pop(0)
        closedlist.append(current)
        if game.terminalState(current):
            continue
        legalmoves = game.legalMoves(mark, current)
        if legalmoves == []:
            continue		
        for i in range(len(legalmoves)):
            nextboard = game.generateBoard(current, legalmoves, i, mark)
            openlist.append(nextboard)
            if mark ==1:
                marklist.append(2)
            else:
                marklist.append(1)
    return closedlist
    
	
if __name__ == '__main__':
    main()