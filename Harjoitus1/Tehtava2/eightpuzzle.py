# -*- coding: utf-8 -*-
import time
import queue
import random

class EightPuzzle:
    def __init__(self):
        """
		Alustetaan 3x3 lauta listana ([1,2,3,4,5,6,7,8,0]).
        """
        self.targetstate = list(range(1, 9))
        self.targetstate.append(0)

    def printState(self, state):
        """
        Tämä funktio printtaa listan matriisi muotoisena vasemmasta yläkulmasta oikeaan alakulmaan.
        """
        for (index, value) in enumerate(state):
            print("{}".format(value),end='')
            if index==2 or index==5 or index==8:
                print("")
        print("")

    def legalMoves(self, position):
        """
        Tämä funktio ratkaisee laillisten siirtojen paikan matriisissa.
        """
        values = [1, -1, 3, -3]
        legalmoves = []
        for x in values:
            if 0 <= position + x < 9:
                if x == 1 and (position==2 or position==5 or position==8):
                    continue
                if x == -1 and (position==0 or position==3 or position==6):
                    continue
                legalmoves.append(x)
        return legalmoves

    def findNextStates(self, state):
        """
        Tämä funktio ratkaisee seuraavat mahdolliset siirrot nykyisestä tilasta.
        """
        legalmoves = {}
        for position in range(9):
            legalmoves[position] = self.legalMoves(position)
        emptyspace = state.index(0)
        moves = legalmoves[emptyspace]
        nextstates = []
        for move in moves:
            state_x = state[:]
            (state_x[emptyspace + move], state_x[emptyspace]) = (state_x[emptyspace], state_x[emptyspace + move])
            nextstates.append(state_x)
        return nextstates

    def drawState(self, state):
        """
        Tämä funktio valitsee seuraavan tilan sattumanvaraisesti, jota tarvitaan sekoittaessa peliä.
        """
        nextstates = self.findNextStates(state)
        randomstate = random.choice(nextstates)
        return randomstate

    def initialState(self, shuffles=200):
        """
        Tämä funktio selvittää alkutilan pelille sekoittamalla x kertaa alkutilasta.
        """
        initialstate = (self.targetstate)[:]
        for i in range(shuffles):
            initialstate = self.drawState(initialstate)
        return initialstate

    def gameSolved(self, state):
        """
        Tämä funktio tarkistaa, onko peli ratkaistu.
        """
        return state == self.targetstate
		
    def calculateCityBlock(self, state):
        """
        Tämä funktio laskee jokaiselle laatalle horisontaaliset ja vertikaaliset liikkeet siirtääkseen numero oikealle paikalle ja palauttaa city-block-etäisyyden.
        """
        manhattanvalue = 0
        for number in range(1,9):
            (verticalstatevalue, horizontalstatevalue) = (state.index(number) // 3, state.index(number) % 3)
            (verticalgoalvalue, horizontalgoalvalue) = (self.targetstate.index(number) // 3, self.targetstate.index(number) % 3)
            verticalvalue = abs(verticalstatevalue - verticalgoalvalue)
            horizontalvalue = abs(horizontalstatevalue - horizontalgoalvalue)
            manhattanvalue += verticalvalue + horizontalvalue
        return manhattanvalue
		
    def calculateHamming(self, state):
        """
        Tämä funktio laskee kuinka moni nykyisen tilan laatoista on väärällä paikalla ja palauttaa hamming-etäisyyden.
        """
        hammingvalue = 0
        #-------TÄHÄN SINUN KOODI--------
        for i in state:
            if(i != self.targetstate[state.index(i)] and i != 0):
                hammingvalue += 1             
        #--------------------------------
        return hammingvalue
		
    def evaluationFunction(self, state, evaluation):
        """
        Tämä funktio valitsee arviointifunktioksi joko hamming-etäisyyden 'hamming' tai city-block-etäisyyden. Valitaan city-block jos evaluation on jotain muuta kuin 'hamming'.
        """
        if evaluation == 'hamming':
            return self.calculateHamming(state)
        else:
            return self.calculateCityBlock(state)
            
		
    def getLowestF(self, openlist, fscore):
        """
        Tämä funktio tarkistaa avoimen listan pienimmän f-scoren arvon ja palauttaa kyseisen tilan.
        """
        value = 10000000000
        for state in openlist:
            if fscore[self.listToString(state)] < value:
                value = fscore[self.listToString(state)]
                current = state
        return current
		
    def listToString(self, numberlist):
        """
        Tämä funktio muuttaa listan stringiksi.
        """
        return ''.join(list(map(str, numberlist)))
	
    def reconstructPath(self, camefrom, current, expanded, print_path):
        """
        Tämä funktio rakentaa polun alkutilasta tavoitetilaan, kun tavoitetila on löydetty.
        """
        totalpath = []
        current = self.targetstate
        while self.listToString(current) in list(camefrom.keys()):
            current = camefrom[self.listToString(current)]
            totalpath.append(current)
        totalpath.reverse()
        totalpath.append(self.targetstate)
        if print_path:
            for i in totalpath:
                self.printState(i)
        print("Used {} moves to reach the goal state (expanded in total {} nodes)".format(len(totalpath)-1, expanded))
                    
    def aStar(self, state, evaluation='city-block', print_path=True):
        """
        Tämä funktio ratkaisee pelin käyttämällä hakualgoritmina A-tähti hakua.
        """
        closedlist = []
        openlist = []
        openlist.append(state)
        camefrom = {}
        gscore = {}
        gscore[self.listToString(state)] = 0
        fscore = {}
        fscore[self.listToString(state)] = gscore[self.listToString(state)] + self.evaluationFunction(state, evaluation)
        while len(openlist) != 0:
            current  = self.getLowestF(openlist, fscore)
            if current == self.targetstate:
                return self.reconstructPath(camefrom, current, len(closedlist)+1, print_path)
            openlist.remove(current)
            closedlist.append(current)
            nextstates = self.findNextStates(current)
            for nextstate in nextstates:
                tentativegscore = gscore[self.listToString(current)] + 1
                if nextstate in closedlist and tentativegscore >= gscore[self.listToString(nextstate)]:
                    continue
                if nextstate not in closedlist or tentativegscore < gscore[self.listToString(nextstate)]:
                    camefrom[self.listToString(nextstate)] = current
                    gscore[self.listToString(nextstate)] = tentativegscore
                    fscore[self.listToString(nextstate)] = gscore[self.listToString(nextstate)] + self.evaluationFunction(nextstate, evaluation)
                    if nextstate not in openlist:
                        openlist.append(nextstate)
        print("Couldn't find the solution")
        
		
def main():
    # Esimerkki ratkaisu a-tähti algoritmilla city-block-etäisyys arviointifunktiota käyttäen satunnaisesti sekoitetusta tilasta
    print('8-Puzzle solver for randomly shuffled start state')
    print(50 * '-')
    game = EightPuzzle()
    print('\nThe start state is:')
    initialstate = game.initialState(15)
    game.printState(initialstate)
    print('The goal state is:')
    game.printState(game.targetstate)
    print('The path of solution:')
    game.aStar(initialstate, evaluation='city-block', print_path=True)

    # Ratkaisu a-tähti algoritmilla city-block etäisyys ja hamming arviointifunktiota käyttäen alkutilalle start state 1
    print('\n8-Puzzle solver for three different start states with hamming-distance and city-block-distance')
    print(95 * '-')

    startstate1 = [1,5,2,4,8,3,0,7,6]
    print('\nThe start state 1 is:')
    game.printState(startstate1)
    print('The start state 1 with city-block-distance:')
    game.aStar(startstate1, evaluation='city-block', print_path=False)
    print('The start state 1 with hamming-distance:')
    game.aStar(startstate1, evaluation='hamming', print_path=False)	
    #-------TÄHÄN SINUN KOODI--------
    startstate2 = [8,1,0,5,3,2,4,7,6]
    print('\nThe start state 2 is:')
    game.printState(startstate2)
    print('The start state 2 with city-block-distance:')
    game.aStar(startstate2, evaluation='city-block', print_path=False)
    print('The start state 1 with hamming-distance:')
    game.aStar(startstate2, evaluation='hamming', print_path=False)	

    startstate3 = [2,8,0,5,7,3,4,1,6]
    print('\nThe start state 3 is:')
    game.printState(startstate3)
    print('The start state 3 with city-block-distance:')
    game.aStar(startstate3, evaluation='city-block', print_path=False)
    print('The start state 3 with hamming-distance:')
    game.aStar(startstate3, evaluation='hamming', print_path=False)
    
    #--------------------------------
	
 
    
	
if __name__ == '__main__':
    main()