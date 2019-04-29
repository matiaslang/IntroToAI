# -*- coding: utf-8 -*-
import sys
import time
import queue
import random
import copy
import hexapawnstates
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class HexaPawn:
    def __init__(self):
        """
        Alustetaan avautuvan pelilaudan klikkauksen koordinaatit
        """
        self.coordinates = []
        
    def printCanvas(self, state, playerscore, computerscore, perfectgame):
        """
        Tämä funktio tulostaa pelilaudan erilliseen ikkunaan graafista toteutusta varten
        """
        plt.ion()
        plt.figure(3)
        plt.clf()
        ax = plt.gca()
        backgroundimage = Image.open('pictures/background.jpg')
        clicked = Image.open('pictures/whiteclicked.jpg')
        player = Image.open('pictures/white.jpg')
        computer = Image.open('pictures/black.jpg')
        empty = Image.open('pictures/empty.jpg')
        current_img = backgroundimage
        for i in range(9):
            if state[i] == 0:
                img = empty
            elif state[i] == 1:
                img = player
            elif state[i] == 2:
                img = computer
            elif state[i] == 3:
                img = clicked
            x = i%3
            y = int(i/3)
            area = (int(x*124+(x*4)),int(y*124+(y*4)),int((x+1)*124+(x*4)),int((y+1)*124+(y*4)))
            current_img.paste(img, area)
        plt.title('THE SCORE IS ({}-{}) (PLAYER-COMPUTER)'.format(playerscore, computerscore))
        ax.set(xticks=[], yticks=[])
        plt.xlabel('(Perfect computer: {})'.format(str(perfectgame)))
        plt.imshow(current_img)
        plt.ioff()
        plt.pause(0.05)
        
    def onClick(self, event):
        """
        Tämä funktio palauttaa klikatun pisteen koordinaatit klikkauksen tapahtuessa
        """
        self.coordinates = []
        return self.coordinates.extend([event.xdata, event.ydata])
        
    def printPlayAgain(self, playerscore, computerscore, perfectgame):
        """
        Tämä funktio printtaa "pelataanko uudestaan" kutsun pelin päätyttyä
        """
        plt.ion()
        plt.figure(3)
        plt.clf()
        ax = plt.gca()
        backgroundimage = Image.open('pictures/playagain.jpg')
        current_img = backgroundimage
        plt.title('THE SCORE IS ({}-{}) (PLAYER-COMPUTER)'.format(playerscore, computerscore))
        ax.set(xticks=[], yticks=[])
        plt.xlabel('(Perfect computer: {})'.format(str(perfectgame)))
        plt.imshow(current_img)
        plt.ioff()
        plt.pause(0.05)     
        
    def printQ(self, Q, allstates):
        """
        Tämä funktio muuntaa erilliseen ikkunaan tila-liike taulukon saamat palautteet
        """
        plt.ion()
        plt.figure(1)
        plt.clf()
        dictlist = self.dictToList(Q, allstates)
        array = np.array(dictlist)
        plt.matshow(array, fignum=False, cmap=plt.cm.Greys)
        plt.colorbar()
        ax = plt.gca()
        scale_val = 1
        ax.format_coord = lambda x, y: 'r=%d,c=%d' % (scale_val * int(x + .5), scale_val * int(y + .5))
        plt.title('STATE-ACTION MAP')
        plt.ioff()
        plt.pause(0.05)
        
    def dictToList(self, Q, allstates):
        """
        Tämä funktio muuntaa tila-liike taulukon sanakirjasta matriisiksi
        """
        outputlist = []
        for i in range(len(allstates)):
            subarray = [0] * (len(allstates))
            outputlist.append(subarray)
        tempdict = {}
        for i in range(len(allstates)):    
            tempdict[''.join(str(e) for e in allstates[i])] = i
        keys = list(Q.keys())
        values = list(Q.values())
        for i in range(len(keys)):
            k1 = ''.join(str(e) for e in keys[i][0])
            k2 = ''.join(str(e) for e in keys[i][1])
            outputlist[tempdict[k1]][tempdict[k2]] = values[i]
        return outputlist
        
    def legalMoves(self, mark, board):
        """
        Tämä funktio selvittää tutkittavan tilan lailliset siirrot muodossa (8,5), missä 8 viittaa siirrettävän merkin alkukoordinaattiin ja 5 loppukoordinaattiin. Koordinaatit menevät oikeasta yläkulmasta vasempaan alakulmaan 0,1,2,3,4,5,6,7,8
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
        
    def generateNewBoard(self, board, move, mark):
        """
        Tämä funktio luo päivitetyn pelilaudan annetun alkukoordinaatin ja loppukoordinaatin perusteella
        """
        newboard = copy.deepcopy(board)
        newboard[move[1]] = newboard[move[0]]
        newboard[move[0]] = 0
        return newboard
        
    def terminalState(self, board, legalmoves):
        """
        Tämä funktio tarkastaa, onko kyseessä päätetila. Jos jompi kumpi pelaajista on päässyt kolmannelle riville tai jos laillisia siirtoja ei ole jäljellä, niin kyseessä on päätetila
        """
        if board[0] == 1 or board[1] == 1 or board[2] == 1:
            return True
        if board[6] == 2 or board[7] == 2 or board[8] == 2:
            return True 
        if legalmoves == []:
            return True
        
    def bestAction(self, Q, board, mark):
        """
        Tämä funktio valitsee tila-liike taulukosta parhaan mahdollisen siirron tutkittavalle tilalle. Mikäli tila-liike taulukon perusteella parhaita siirtoja on useita, arvotaan niistä yksi
        """
        Qvalues = []
        actions = []
        legalmoves = self.legalMoves(mark, board)
        for legalmove in legalmoves:
            action = self.generateNewBoard(board, legalmove, mark)
            actions.append(action)
            Qvalues.append(Q[(tuple(board),tuple(action))])
        maxQ = max(Qvalues)
        indexes = []
        for i in range(len(Qvalues)):
            if Qvalues[i] == maxQ:
                indexes.append(i)
        index = random.choice(indexes)
        return actions[index], Qvalues
                
    def initializeStateActionMap(self, allstates):
        """
        Tämä funktio alustaa tila-liike taulukon (sanakirjan) muuttujat arvolla 0
        """
        map = {}
        for state in allstates:
            for action in allstates:
                map[(tuple(state), tuple(action))] = 0
        return map
        
    def mirrorBoard(self, board):
        """
        Tämä funktio palauttaa pelilaudan peilikuvan
        """
        return [board[2], board[1], board[0], board[5], board[4], board[3], board[8], board[7], board[6]]        
        
    def perfectGameChecker(self, Q, importantstates):
        """
        Tämä funktio tarkastaa, pelaako tekoäly täydellisesti Q-arvojen perusteella
        """
        perfectstates = []
        Qlist = []
        for state in importantstates:
            board, Qvalues = self.bestAction(Q, state, 2)
            counter = 0         
            for i in range(len(Qvalues)):
                if Qvalues[i]==0:
                    counter += 1
            if state == self.mirrorBoard(state):
                counter = int(counter/2)
            value = True
            if max(Qvalues)==0 and counter>1:
                value = False       
            perfectstates.append(value)
            Qlist.append(Qvalues)
        return all(perfectstates), importantstates, perfectstates, Qlist
        
    def printImportantStates(self, importantstates, perfectstates, Qlist):
        """
        Tämä funktio tulostaa ikkunaan pelin ratkaisun kannalta kriittisille tiloille tiedon tekoälyn täydellisyydestä sekä seuraavien liikkeiden Q-arvot
        """
        plt.ion()
        plt.figure(2)
        plt.clf()
        backgroundimage = Image.open('pictures/background.jpg')
        clicked = Image.open('pictures/whiteclicked.jpg')
        player = Image.open('pictures/white.jpg')
        computer = Image.open('pictures/black.jpg')
        empty = Image.open('pictures/empty.jpg')
        for j in range(len(importantstates)):
            ax=plt.subplot(3, 3, j+1)
            state = importantstates[j]
            current_img = backgroundimage
            for i in range(9):
                if state[i] == 0:
                    img = empty
                elif state[i] == 1:
                    img = player
                elif state[i] == 2:
                    img = computer
                elif state[i] == 3:
                    img = clicked
                x = i%3
                y = int(i/3)
                area = (int(x*124+(x*4)),int(y*124+(y*4)),int((x+1)*124+(x*4)),int((y+1)*124+(y*4)))
                current_img.paste(img, area)
            ax.set(xticks=[], yticks=[])
            plt.title('Solves this state perfectly: {}'.format(str(perfectstates[j])), fontsize=7)
            plt.xlabel('Rewards of actions: {}'.format(', '.join(str(k) for k in Qlist[j])), fontsize=7)
            plt.imshow(current_img)
        plt.tight_layout()
        plt.ioff()
        plt.pause(0.05)
        
    def QLearning(self, Q, stateaction, alpha, reward):
        """
        Tämä funktio palauttaa Q-arvon tutkittavalle tila-liike parille
        """
        #-------TÄHÄN SINUN KOODI--------        
        return 0                          # Poista tämä rivi
        #--------------------------------
        
def main():
    allstates = hexapawnstates.main()
    print('\nHEXAPAWN WITH REINFORCEMENT LEARNING\n------------------------------------\nPLAYER 1 PLAYS WITH WHITE AND ALWAYS STARTS THE GAME. AI PLAYS WITH BLACK\n')
    game = HexaPawn()
    Q = game.initializeStateActionMap(allstates)
    playerscore, computerscore = 0, 0
    alpha = 1
    importantstates = [[2,2,2,1,0,0,0,1,1],[2,2,2,0,1,0,1,0,1],[2,0,2,2,1,0,0,0,1],[0,2,2,1,2,0,0,0,1],[0,2,2,0,1,0,0,0,1],[0,2,2,0,1,0,1,0,0],[0,2,0,2,1,1,0,0,0]]
    perfectgame, gameover = False, False
    game.printQ(Q, allstates)
    game.printImportantStates(importantstates, [False]*7, [[]]*7)
    while True:
        board = [2,2,2,0,0,0,1,1,1]
        gameisgoing = True
        mark = 1
        while gameisgoing:
            game.printCanvas(board, playerscore, computerscore, perfectgame)
            if mark == 2:
                time.sleep(2)
                state = board
                action, Qvalues = game.bestAction(Q, state, mark)
                board = action
                stateaction = ((tuple(state),tuple(action)))
                if game.terminalState(board, game.legalMoves(3-mark, board)):
                    print("COMPUTER WON THE GAME\n") 
                    computerscore += 1
                    reward = 5
                    game.printCanvas(board, playerscore, computerscore, perfectgame)
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
                fig = plt.figure(3)
                first_print = False
                clicked_first_time = clicked_second_time = True
                cid = fig.canvas.mpl_connect('button_press_event', game.onClick)
                while clicked_second_time:
                    while clicked_first_time:
                        clicked_first_time = plt.waitforbuttonpress()
                        [xcoord,ycoord] = game.coordinates
                        if xcoord == None or ycoord == None:
                            clicked_first_time = True
                            continue
                        clicked_position1 = int(xcoord/128) + 3*int(ycoord/128)
                        if clicked_position1 not in legalpositions:
                            clicked_first_time = True
                            continue
                        tempboard = copy.deepcopy(board)
                        tempboard[clicked_position1] = 3
                    if clicked_first_time == False and first_print == False:
                        game.printCanvas(tempboard, playerscore, computerscore, perfectgame)
                        first_print = True
                    clicked_second_time = plt.waitforbuttonpress()
                    [xcoord,ycoord] = game.coordinates
                    if xcoord == None or ycoord == None:
                        clicked_second_time = True
                        continue
                    clicked_position2 = int(xcoord/128) + 3*int(ycoord/128)
                    if [clicked_position1, clicked_position2] not in legalmoves:
                        if clicked_position1 == clicked_position2:
                            clicked_second_time = clicked_first_time = True
                            first_print = False
                            game.printCanvas(board, playerscore, computerscore, perfectgame)
                            continue
                        else:
                            clicked_second_time = True
                            first_print = False
                            continue
                nextmove = [clicked_position1,clicked_position2]
                board = game.generateNewBoard(board, nextmove, mark)
                if game.terminalState(board, game.legalMoves(3-mark, board)):
                    print("PLAYER WON THE GAME\n")
                    reward = -5
                    playerscore += 1
                    game.printCanvas(board, playerscore, computerscore, perfectgame)
                    Q[stateaction] = game.QLearning(Q, stateaction, alpha, reward)
                    if state != game.mirrorBoard(state) or action != game.mirrorBoard(action):
                        stateaction = ((tuple(game.mirrorBoard(state)),tuple(game.mirrorBoard(action))))
                        Q[stateaction] = game.QLearning(Q, stateaction, alpha, reward)
                    gameisgoing = False
                mark = 2
        time.sleep(1)
        game.printQ(Q, allstates)
        perfectlysolved, importantstates, perfectstates, Qlist = game.perfectGameChecker(Q, importantstates)
        game.printImportantStates(importantstates, perfectstates, Qlist)
        if perfectlysolved != perfectgame:
            plt.savefig("criticalstates.png")
            game.printQ(Q, allstates)
            plt.savefig("stateactionmap.png")
            game.printCanvas(board, playerscore, computerscore, perfectlysolved)
            plt.savefig("perfectgame.png")
            print("COMPUTER PLAYS NOW PERFECTLY")
        perfectgame = perfectlysolved
        print('THE SCORE IS ({}-{}) (PLAYER-COMPUTER)\n'.format(playerscore, computerscore))
        fig = plt.figure(3)
        game.printPlayAgain(playerscore, computerscore, perfectgame)
        clicked_first_time = True
        cid = fig.canvas.mpl_connect('button_press_event', game.onClick)
        while clicked_first_time:
            clicked_first_time = plt.waitforbuttonpress()
            [xcoord,ycoord] = game.coordinates
            if xcoord == None or ycoord == None:
                clicked_first_time = True
                continue
            clicked_position = int(xcoord/128) + 3*int(ycoord/128)
            if clicked_position == 6:
                break
            if clicked_position == 8:
                gameover = True
                break
            else:
                clicked_first_time = True
        if gameover:
            break   
    print("GAME OVER!")
    
if __name__ == '__main__':
    main()