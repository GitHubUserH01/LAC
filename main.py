import lane
import car
import math
import visualize
import tensorflow as tf
import sys
import pandas as pd
import random
class World(object):
    def __init__(self, midlevel_exists = False):
        self.auto_cars = []
        self.human_cars = []
        self.lanes = []
        self.roads = []
        self.fences = []
        self.objects = []


def playground():
    world = World()
    clane = lane.StraightLane([0., -100.], [0., 100.], 0.17)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    #world.roads += [clane]
    #world.fences += [clane.shifted(2), clane.shifted(-2)]
    #world.cars.append(car.UserControlledCar(dyn, [0., 0., math.pi/2., 0.], color='orange'))
    world.auto_cars.append(car.Car([0, -0.5, -math.pi / 2., 0.], color='black', car_no=0))
    world.auto_cars.append(car.Car([0, -0.30, -math.pi/2., 0.], color='black', car_no=1))
    world.auto_cars.append(car.Car([0, -0.1, -math.pi/2., 0.], color='black', car_no=2))
    world.auto_cars.append(car.Car([0, 0.1, -math.pi/2., 0.], color='black', car_no=3))
    world.auto_cars.append(car.Car([0, 0.4, -math.pi / 2., 0.], color='black', car_no=4))

    world.human_cars.append(car.Car([-0.17, -0.30, -math.pi / 2. , 0.], color='red', car_no=5))



    for i in range(13):
        world.objects.append(car.Obj([0.35, 0.9-i*0.15]))
        world.objects.append(car.Obj([-0.35, 0.9 - i * 0.15]))
    return world



    #plt.plot(v2_df['Local_Y'], v2_df['Local_X'], 'b--', linewidth=.5)

if __name__ == '__main__':
    random.seed(310)
    tf.reset_default_graph()
    world = playground()
    #world.cars = world.cars[:0]
    vis = visualize.Visualizer(0.1, magnify=1.1)
    vis.use_world(world, )
    vis.run()
