import gym
import numpy as np
import carla
SHOW_PREVIEW=True
IM_WIDTH = 224
IM_HEIGHT = 224
class CarlaEnv(gym.Env):
    def __init__(self):
        super(CarlaEnv,self).__init__()
        self.show_cam = SHOW_PREVIEW
        self.im_width = IM_WIDTH
        self.im_height = IM_HEIGHT
        
    def step(self):
        pass
    def reset(self):
        pass
    def process_image(self,image):
        i = np.array(image.raw_data)
        i2 = i.reshape()