import numpy as np

class Camera:
    def __init__(self):
        pass
    
    def white_out_image(self, width, height):
        white_out_image = np.zeros((height,width,3), np.uint8)
        white_out_image[:,0:width] = (255, 255, 255)      # (B, G, R)
        return white_out_image