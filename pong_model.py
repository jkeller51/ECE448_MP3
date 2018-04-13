#Internal representation of game state and actions.
#Different from actual game view
#This is the model component following an MVC design pattern



class PongModel:
    def __init__(self, ballX, ballY, bvelocityX, bvelocityY, paddleY):
        self.ball_x = ballX
        self.ball_y = ballY
        self.ball_velocity_x = bvelocityX
        self.ball_velocity_y = bvelocityY
        self.paddle_x = 1.0 #this is a constant
        self.paddle_y = paddleY
        self.paddle_height = 0.2
        self.rewards = 0
        


    def can_move(self, proposed_move):
        """
        Args:
           proposed_move(int): the distance the paddle wants to move, in the y direction
        Returns(bool) if proposed move is valid.
    
        """
        if proposed_move == 0 or proposed_move == 0.04 or proposed _move == -0.04:
            return True
        else:
            return False


    def move(self, proposed_move = 0):
        """
           Move the paddle
           Args:
              proposed_move(int): the distance to move the paddle, in the y direction. 
        """
        if self.can_move(proposed_move): #if the paddle can move in the specified direction
            self.paddle_y += proposed_move
        else:
            #we can't move that far because we'll go off the board.
            if proposed_move < 0: #trying to move up. so set to highest possible position instead
                self.paddle_y = 0
            else:
                #trying to move down.
                self.paddle_y = 1

    def game_terminated():
        """
           Returns(bool): True if ball's x coordinate is greater than x coordinate of  paddle (i.e. ball has passed paddle). False otherwise
        """
        if self.ball_x > self.paddle_x:
            return True
        else:
            return False

    def simulate_env():
        """Update environment at each timestep"""
        pass
        
    


    
    
