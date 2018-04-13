import graphics as gfx
import time


if __name__ == '__main__':
    
    window = gfx.GFX()
    start = time.time()
    
    while 1:
        # main loop
        window.update()
        if (time.time() > start+5 and time.time() < start+10):
            # example of moving the player up
            # window updates at 30 fps; keep this in mind
            window.player.y += 1
            window.ball.x += 1   # 30 pixels/second
        
        if (time.time() > start+10 and time.time() < start+15):
            # moving player down
            window.player.y -= 1
            window.ball.x -= 1
            
        
        



    
    
#tk.mainloop()