# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:39:40 2018

@author: jkell
"""

import tkinter as tk
import time
import threading

class DrawThread(threading.Thread):
    
    def __init__(self, gfxobj):
        self.done = True
        self.gfx = gfxobj
        
    def run(self,fps):
        self.gfx._update()
        time.sleep(1/fps)
        self.done = True

class Ball:
    def __init__(self, canvas, color):
        self.canvas = canvas
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)
        self.canvas.move(self.id, 245, 100)
        self.color = color
        self.x = 0
        self.y = 0
        self.width=20
        self.height=20
    def draw(self):
        self.canvas.delete(self.id)
        self.id = self.canvas.create_oval(self.x-self.width/2, self.y-self.height/2, self.x+self.width/2, self.y+self.height/2,fill=self.color)


class Player:
    def __init__(self, canvas):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0, 0, 10, 50, fill="black")
        self.x = 0
        self.y = 0
        self.width=10
        self.height=80
        
    def draw(self):
        self.canvas.delete(self.id)
        self.id = self.canvas.create_rectangle(self.x-self.width/2, self.y-self.height/2, self.x+self.width/2, self.y+self.height/2, fill="black")


class Wall:
    def __init__(self, canvas, wall_x=0):
        self.canvas = canvas
        self.id = canvas.create_rectangle(wall_x, 0, wall_x+10, 400, fill="black")
        self.x = wall_x
        self.y = 0
        self.width=10
        self.height=400
    

class GFX:
    def __init__(self, wall_x=0, player_x=1):
        """ Initialize the game window and objects
        There is a ball and a player."""
        self.win = tk.Tk()
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.win.title = "Pong"
        self.win.resizable(0,0)
        self.win.wm_attributes("-topmost", 1)
        self.win.after(1, lambda: self.win.focus_force())
        
        self.width=400
        self.height=400
        
        self.canvas = tk.Canvas(self.win, width=self.width, height=self.height, bd=0, highlightthickness=0)
        self.canvas.pack()
        self._open=True     # variable to keep track of open window (DO NOT CHANGE)
    
        self.ball = Ball(self.canvas, "red")
        self.ball.width = 10
        self.ball.height= 10
        self.ball.x = 200
        self.ball.y = 200
        
        self.player = Player(self.canvas)
        self.player.x = player_x * 400
        self.player.y = 200
        
        self.wall = Wall(self.canvas, wall_x*self.width)
        self.wall.x = wall_x * 400
        
        self.thread = DrawThread(self)
        self.fps = 30

    
    def _update(self):
        # DON'T CALL THIS
        self.ball.draw()
        self.player.draw()
        
        self.win.update_idletasks()
        self.win.update()
        
    def update(self):
        # put this in the main loop
        if (self.thread.done == True):
            # run the thread
            
            self.thread.done = False
            self.thread.run(self.fps)
        
    def close(self):
        # close the window
        self.win.destroy()
        self._open = False
        