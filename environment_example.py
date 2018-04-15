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


if __name__ == '__main__':
    
    window = gfx.GFX()
    #window.fps = 5  # you can modify this for debugging purposes, default=30
    Game = png.PongModel(0.5, 0.5, 0.03, 0.01, 0.4)   # initialize state
    #################################################### Ignore this
    window.win.bind('<Up>',lambda eff: UpKey(Game))
    window.win.bind('<Down>',lambda eff: DownKey(Game))
    ######################################################## /ignore
        
    Lflag = False
    while 1:
        # main loop
        if (window._open == False):
            break
        
        Game.update(window)
        
        if (Game.lost == True and Lflag == False):
            Game.reset()

    print("Game ended.")
    



    