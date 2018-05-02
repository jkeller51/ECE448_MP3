#Internal representation of game state and actions.
#Different from actual game view
#This is the model component following an MVC design pattern

import random
import numpy as np

def intersect_line(x1,y1,x2,y2,linex):
    """
    Determine where the line drawn through (x1,y1) and (x2,y2)
    will intersect the line x=linex
    """    
    return ((y2-y1)/(x2-x1))*abs(linex-x1)+y1
    

class PongModel:
    def __init__(self, ballX, ballY, bvelocityX, bvelocityY, paddleY, paddleX=1.0):
        self.initial = (ballX, ballY, bvelocityX, bvelocityY, paddleY, paddleX)
        
        self.ball_x = ballX
        self.ball_y = ballY
        self.ball_velocity_x = bvelocityX
        self.ball_velocity_y = bvelocityY
        self.paddle_x = paddleX
        self.paddle_y = paddleY
        
        self.ball_lastx = ballX
        self.ball_lasty = ballY
        
        self.score = 0
        self.lost = False
        
        self.paddle2_x = -1
        self.score2 = 0
        
        self.won = 0
        
    def init2(self, paddle2Y, paddle2X=0.0):  # init for 2nd player
        
        self.initial2 = (paddle2Y, paddle2X)
        
        self.paddle2_x = paddle2X
        self.paddle2_y = paddle2Y
        self.score2 = 0
        
        
    def reset(self):
        """ Reset everything."""
        ballX, ballY, bvelocityX, bvelocityY, paddleY, paddleX = self.initial
        if (self.paddle2_x != -1):
            paddle2Y, paddle2X = self.initial2
            
        self.__init__(ballX, ballY, bvelocityX, bvelocityY, paddleY, paddleX)
        self.init2(paddle2Y, paddle2X)
        self.won=0

    def move(self, proposed_move = 0, player=1):
        """
           Move the paddle
           Args:
              proposed_move(float): the distance to move the paddle, in the y direction. 
        """
        if (player == 1):
            self.paddle_y += proposed_move
    
            if self.paddle_y+0.2 > 1:
                self.paddle_y = 0.8
            elif self.paddle_y < 0:
                self.paddle_y = 0
        else:
            self.paddle2_y += proposed_move
    
            if self.paddle2_y+0.2 > 1:
                self.paddle2_y = 0.8
            elif self.paddle2_y < 0:
                self.paddle2_y = 0

    def move_up(self, player=1):
        self.move(-0.04, player)
        
    def move_down(self, player=1):
        self.move(0.04, player)
        
    def get_state(self):
        return [self.ball_x, self.ball_y, self.ball_velocity_x, 
                self.ball_velocity_y, self.paddle_y]
        
    def get_state2(self):
        return [self.ball_x, self.ball_y, self.ball_velocity_x, 
                self.ball_velocity_y, self.paddle2_y]
        
    def bounce_wall(self, gfx):
        if (self.paddle2_x > -1):
            return False
        """ Check whether our ball hit the wall."""
        wall_x = gfx.wall.x / 400
        
        # wall is to the left of paddle
        if ((self.ball_x <= wall_x) and 
            (self.ball_velocity_x < 0) and (self.paddle_x > wall_x)):
            return True
        # wall is to the right of paddle
        elif ((self.ball_x >= wall_x) and 
              (self.ball_velocity_x > 0) and (self.paddle_x < wall_x)):
            return True
        else:
            return False
    
    def bounce_paddle(self, gfx, player=1):
        """ Check whether our ball hit the paddle."""
        px = self.paddle_x
        py = self.paddle_y
        if (player == 2):
            px = self.paddle2_x
            py = self.paddle2_y
        # paddle is to the right of wall
        if px > 0.5:
            intersect_y = intersect_line(self.ball_lastx, self.ball_lasty, 
                                         self.ball_x, self.ball_y, px)
            if (intersect_y >= py and 
                intersect_y <= py + 0.2 and 
                self.ball_lastx <= px):
                return True
            else:
                return False
        # paddle is to the left of wall
        elif px < 0.5:
            intersect_y = intersect_line(self.ball_lastx, self.ball_lasty, 
                                         self.ball_x, self.ball_y, px)
            if (intersect_y >= py and 
                intersect_y <= py + 0.2 and 
                self.ball_lastx >= px):
                return True
            else:
                return False
        
    def update(self, gfx):
        """
        Update the window based on the internal state
        """
        if gfx.thread.done == True:
            """
            we want to make sure we are only calculating the new position
            when the frame is refreshed
            """
            self.ball_lastx = self.ball_x
            self.ball_lasty = self.ball_y
            
            # update ball position
            self.ball_x += self.ball_velocity_x
            self.ball_y += self.ball_velocity_y
            
            # check bouncing
            if (self.ball_y >= 1 and self.ball_velocity_y > 0):
                self.ball_velocity_y *= -1
            elif (self.ball_y <= 0 and self.ball_velocity_y < 0):
                self.ball_velocity_y *= -1
            elif self.bounce_wall(gfx):
                self.ball_velocity_x *= -1
            elif ((self.ball_x >= self.paddle_x and self.ball_velocity_x > 0
                  and self.paddle_x > 0.5) or 
                 (self.ball_x <= self.paddle_x and self.ball_velocity_x < 0 
                  and self.paddle_x < 0.5)): # player 1 bounce
                if self.bounce_paddle(gfx):
                    # we hit the paddle!
                    self.score += 1
                    self.ball_x = 2*self.paddle_x - self.ball_x
                    
                    # randomize velocities
                    U = random.uniform(-0.015, 0.015)
                    V = random.uniform(-0.03,  0.03 )
                    temp_vx = -1*self.ball_velocity_x + U
                    temp_vy = -1*self.ball_velocity_y + V
                    if temp_vx >= 0 and temp_vx <= 0.03:
                        temp_vx = 0.03
                    elif temp_vx < 0 and temp_vx >= -0.03:
                        temp_vx = -0.03
                        
                    self.ball_velocity_x = temp_vx
                    self.ball_velocity_y = temp_vy
                    
                else:
                    # missed the paddle. lost!
                    self.lost = True
                    # player 2 wins
                    self.won = 2
                    
            elif (self.paddle2_x > -1): # multiplayer
                if (self.ball_x <= self.paddle2_x and self.ball_velocity_x < 0 ):
                    if self.bounce_paddle(gfx,2):
                        # we hit the paddle!
                        self.score2 += 1
                        self.ball_x = 2*self.paddle2_x - self.ball_x
                        
                        # randomize velocities
                        U = random.uniform(-0.015, 0.015)
                        V = random.uniform(-0.03,  0.03 )
                        temp_vx = -1*self.ball_velocity_x + U
                        temp_vy = -1*self.ball_velocity_y + V
                        if temp_vx >= 0 and temp_vx <= 0.03:
                            temp_vx = 0.03
                        elif temp_vx < 0 and temp_vx >= -0.03:
                            temp_vx = -0.03
                            
                        self.ball_velocity_x = temp_vx
                        self.ball_velocity_y = temp_vy
                    else: # player 1 wins!
                        self.lost = True
                        self.won = 1
           
            gfx.player.y = self.paddle_y * gfx.height + gfx.player.height/2  # to be consistent with spec, y represents top of player
            gfx.ball.x = self.ball_x * gfx.width
            gfx.ball.y = self.ball_y * gfx.height
            
            if (self.paddle2_x > -1):
                gfx.player2.y = self.paddle2_y * gfx.height + gfx.player.height/2
                    
            gfx.update() # this happens on a different thread


    
    
