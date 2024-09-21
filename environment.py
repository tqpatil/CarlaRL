import numpy as np
import carla
import random
import time
import cv2 
import math
SHOW_PREVIEW=False
IM_WIDTH = 224
IM_HEIGHT = 224
class CarlaEnv():
    def __init__(self):
        super(CarlaEnv,self).__init__()
        self.SP_EPISODE= 15
        self.show_cam = SHOW_PREVIEW
        self.im_width = IM_WIDTH
        self.im_height = IM_HEIGHT
        self.front_camera = None
        self.steer_amt = 1.0
        self.client = carla.Client("localhost",4000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.lanesensor = None
        self.sensor = None
        self.colsensor = None
        self.actor_list = []
    def start(self):
        if self.lanesensor is not None and self.lanesensor.is_listening:
            self.lanesensor.stop()
            self.colsensor.stop()
            self.sensor.stop()
        self.lanesensor = None
        self.sensor = None
        self.colsensor = None
        for actor in self.actor_list:
            actor.destroy()
        self.collision_hist = []
        self.laneIntr_hist = []
        self.actor_list = []
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)
        
        self.cam = self.blueprint_library.find("sensor.camera.rgb")
        self.cam.set_attribute("image_size_x", f"{self.im_width}")
        self.cam.set_attribute("image_size_y", f"{self.im_height}")
        self.cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z = 0.7))
        self.sensor = self.world.spawn_actor(self.cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_image(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        time.sleep(4)

        collision_sensor = self.blueprint_library.find("sensor.other.collision")
        lane_sensor = self.blueprint_library.find("sensor.other.lane_invasion")
        transform2 = carla.Transform(carla.Location(x=2.0, z = 0.8))
        self.colsensor = self.world.spawn_actor(collision_sensor, transform, attach_to= self.vehicle)
        self.lanesensor = self.world.spawn_actor(lane_sensor, transform2, attach_to = self.vehicle)
        self.actor_list.append(self.colsensor)
        self.actor_list.append(self.lanesensor)
        self.colsensor.listen(lambda x: self.collision_data(x))
        self.lanesensor.listen(lambda x: self.lane_data(x))

        while self.front_camera is None:
            time.sleep(0.01)
        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        return self.front_camera
    def reset(self):
        if self.lanesensor is not None and self.lanesensor.is_listening:
            self.lanesensor.stop()
            self.colsensor.stop()
            self.sensor.stop()
        self.lanesensor = None
        self.sensor = None
        self.colsensor = None
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []
        self.collision_hist = []
        self.laneIntr_hist = []
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle.set_location(self.transform.location)
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        while self.front_camera is None:
            time.sleep(0.01)
        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        return self.front_camera
        ## reset necessary vars and teleport vehicle back to a random spawn location
        
    def collision_data(self, event):
        self.collision_hist.append(event)
    def lane_data(self, event):
        self.laneIntr_hist.append(event)
    def step(self, action):
        if action < -1 or action > 1:
            if action < -1:
                action = -1
            else:
                action = 1
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= action * self.steer_amt))
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        reward = 0
        if len(self.laneIntr_hist) != 0:
            done = False
            reward -= 5 * len(self.laneIntr_hist)
            self.laneIntr_hist = []
        if len(self.collision_hist) != 0:
            done = True
            self.collision_hist = []
            reward -= 50
        elif kmh < 40:
            done = False
            reward -= 10
        else:
            done = False
            reward += 10
        if self.episode_start + self.SP_EPISODE < time.time():
            done = True
            self.collision_hist = []
            self.laneIntr_hist = []
        return self.front_camera, reward, done, None
    def process_image(self,image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.show_cam:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3