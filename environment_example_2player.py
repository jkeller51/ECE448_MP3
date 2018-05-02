# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 20:10:27 2018

@author: jkell
"""

""" example of how the environment is set up and accessed """

import graphics as gfx
import pong_model as png


def UpKey(Game):
    Game.move_up()
    
def DownKey(Game):
    Game.move_down()
    
def UpKey2(Game): # move player 2
    Game.move_up(2)
    
def DownKey2(Game):
    Game.move_down(2)


if __name__ == '__main__':
    
    window = gfx.GFX(wall_x=0, player_x=1, players=2)
    #window.fps = 5  # you can modify this for debugging purposes, default=30
    Game = png.PongModel(0.5, 0.5, 0.03, 0.01, 0.4, paddleX=1)   # initialize state
    Game.init2(0.4, 0)
    #################################################### Ignore this
    window.win.bind('<Up>',lambda eff: UpKey(Game))
    window.win.bind('<Down>',lambda eff: DownKey(Game))
    window.win.bind('w',lambda eff: UpKey2(Game))
    window.win.bind('s',lambda eff: DownKey2(Game))
    ######################################################## /ignore
        
    while 1:
        # main loop
        if (window._open == False):
            break
        
        Game.update(window)
        
        if (Game.lost == True):
            if (Game.won == 1):
                print("Player 1 wins!")
            else:
                print("Player 2 wins!")
            Game.reset()

    print("Game ended.")
    



    