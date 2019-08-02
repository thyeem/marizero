import curses 

class Stone:
    EMPTY = 0
    BLACK = 1
    WHITE = -1

class Player:
    HUMAN = 1
    SOFIAI = 2
    MARIAI = 3
    MARIZERO = 4

#-----------------------------------
BCF = True
API = True 
N = 19
WP = 10
GOAL = 5
## C = 8
## H = 64
## LEARNING_RATE = 1e-3
## GAMMA = 0.99

INF = 1.0e7
D_MINIMAX = 2
BRD_DATA = './__data__'
API_SOFIA = './bin/sofia'
API_MARIA = './bin/maria'
