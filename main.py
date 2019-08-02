import curses
import numpy as np
from const import Stone, Player

class Screen(object):

    def __init__(self):
        self.w = curses.initscr()
        self.w.keypad(True)
        self.w.refresh()
        curses.cbreak()
        curses.noecho()
        curses.start_color()
        curses.mousemask(1)
        curses.mouseinterval(0)
        curses.init_pair(1, curses.COLOR_WHITE,  curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_BLUE,   curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED,    curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_GREEN,  curses.COLOR_BLACK)



class Game(object):
    def __init__(self, scr):
        self.scr = scr

    def main_menu(self):
        self.scr.w.clear()
        self.scr.w.addstr(curses.LINES//2-8, curses.COLS//2-20, "SOFIA WELCOMES YOU.");
        self.scr.w.addstr(curses.LINES//2-6, curses.COLS//2-20, "[1] PLAYING WITH HUMAN");
        self.scr.w.addstr(curses.LINES//2-4, curses.COLS//2-20, "[2] PLAYING WITH SOFIAI");
        self.scr.w.addstr(curses.LINES//2-2, curses.COLS//2-20, "[3] PLAYING WITH MARIAI");
        self.scr.w.addstr(curses.LINES//2+0, curses.COLS//2-20, "[4] WATCHING A GAME BETWEEN AIs");
        self.scr.w.addstr(curses.LINES//2+2, curses.COLS//2-20, "[q] EXIT");
        self.scr.w.addstr(curses.LINES//2+4, curses.COLS//2-20, "PRESS THE KEY TO CONTINUE");
        return self.scr.w.getch()


    def select_menu(self, key):
        pass




try:
    board = np.zeros((19,19), dtype=int)
    scr = Screen()
    g = Game(scr)


    while True:
        c = chr(g.main_menu())
        if c == 'q': 
            break
        else:
            g.select_menu(c)


finally:
    curses.endwin()
            
