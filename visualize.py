import pyglet
from car import Car2
import pyglet.gl as gl
import pyglet.graphics as graphics
import numpy as np
import time
import math
from pyglet.window import key
import rl_loop
import random
import tensorflow as tf
import control
import sys
PNG_PATH = './PNG/'

COLOR_RANDOM = ['blue', 'white',  'gray', 'black', 'red']


class Visualizer(object):
    def __init__(self, dt=0.5, fullscreen=False, name='unnamed', iters=1000, magnify=1.):
        pyglet.resource.path = [PNG_PATH]
        self.visible_cars = []
        self.magnify = magnify
        self.camera_center = None
        self.name = name
        self.objects = []
        self.event_loop = pyglet.app.EventLoop()
        self.window = pyglet.window.Window(1800, 1800, fullscreen=fullscreen, caption=name)
        self.grass = pyglet.resource.texture('grass.png')
        self.window.on_draw = self.on_draw
        self.lanes = []
        self.auto_cars = []
        self.human_cars = []
        self.dt = dt
        self.auto_anim_x = {}
        self.human_anim_x = {}
        self.mock_anim_x = []
        self.obj_anim_x = {}
        self.prev_x = {}
        self.vel_list = []
        self.main_car = None
        self.sess = None
        self.rounds = 0
        self.turn_flag = 1
        self.sec_turn_flag = 0
        self.arrive_flag = 0
        self.leader_list = []
        self.SEKIRO = []
        self.mock = []
        self.timerounds = 0
        self.finalcnnt = 0
        self.controller = control.controller()
        self.doublecheck = [0 for i in range(5)]
        self.historical_mock = []
        self.stop_flag = 0
        def centered_image(filename):
            img = pyglet.resource.image(filename)
            img.anchor_x = img.width/2.
            img.anchor_y = img.height/2.
            return img

        def car_sprite(color, scale=0.17/600.):
            sprite = pyglet.sprite.Sprite(centered_image( '{}.png'.format(color)), subpixel=True)

            sprite.scale = scale
            return sprite

        def object_sprite(name, scale=0.15/900.):
            sprite = pyglet.sprite.Sprite(centered_image( '{}.png'.format(name)), subpixel=True)
            sprite.scale = scale
            return sprite

        self.sprites = {c: car_sprite(c) if c!= 'black' else car_sprite(c, scale=0.2/600.) for c in ['red', 'white',  'gray', 'blue', 'black']}
        self.obj_sprites = {c: object_sprite(c) for c in ['tree', 'firetruck']}
        self.keys = key.KeyStateHandler()

        self.window.push_handlers(self.keys)
        self.window.on_key_press = self.on_key_press

    def on_key_press(self, symbol, modifiers):
        if symbol == key.UP:
            self.keys['up'] = True
        elif symbol == key.LEFT:
            self.keys['left'] = True
        elif symbol == key.RIGHT:
            self.keys['right'] = True

    def on_key_release(self, symbol, modifiers):
        if symbol == key.UP:
            self.keys['up'] = False
        elif symbol == key.LEFT:
            self.keys['left'] = False
        elif symbol == key.RIGHT:
            self.keys['right'] = False

    def use_world(self, world,):
        self.auto_cars = [c for c in world.auto_cars]
        self.human_cars = [c for c in world.human_cars]
        self.lanes = [c for c in world.lanes]
        self.objects = [c for c in world.objects]
        #self.vel_list = [0 for i in range(len(self.cars)-1)]

    def center(self):
        if self.main_car is None:
            return np.asarray([0., 0.])
        elif self.camera_center is not None:
            return np.asarray(self.camera_center[0:2])

    def camera(self):
        o = self.center()
        gl.glOrtho(o[0]-1./self.magnify, o[0]+1./self.magnify, o[1]-1./self.magnify, o[1]+1./self.magnify, -1., 1.)

    def draw_lane_surface(self, lane):
        gl.glColor3f(0.4, 0.4, 0.4)
        W = 1000
        graphics.draw(4, gl.GL_QUAD_STRIP, ('v2f',
            np.hstack([lane.p-lane.m*W-0.5*lane.w*lane.n, lane.p-lane.m*W+0.5*lane.w*lane.n,
                       lane.q+lane.m*W-0.5*lane.w*lane.n, lane.q+lane.m*W+0.5*lane.w*lane.n])
        ))

    def draw_lane_lines(self, lane):

        gl.glColor3f(1., 1., 1.)
        W = 1000
        graphics.draw(4, gl.GL_LINES, ('v2f',
            np.hstack([lane.p-lane.m*W-0.5*lane.w*lane.n, lane.p+lane.m*W-0.5*lane.w*lane.n,
                       lane.p-lane.m*W+0.5*lane.w*lane.n, lane.p+lane.m*W+0.5*lane.w*lane.n])
        ))

    def draw_car(self, x, color='yellow', opacity=255):
        sprite = self.sprites[color]
        sprite.x, sprite.y = x[0], x[1]
        sprite.rotation = x[2]*180./math.pi
        sprite.opacity = opacity
        sprite.draw()

    def draw_object(self, x):
        sprite = self.obj_sprites['tree']
        sprite.x, sprite.y = x[0], x[1]
        sprite.draw()

    def on_draw(self):
        self.window.clear()

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        self.camera()

        gl.glEnable(self.grass.target)
        gl.glEnable(gl.GL_BLEND)

        gl.glBindTexture(self.grass.target, self.grass.id)  # 纹理绑定
        W = 10000.

        graphics.draw(4, gl.GL_QUADS,
            ('v2f', (-W, -W, W, -W, W, W, -W, W)),
            ('t2f', (0., 0., W*5., 0., W*5., W*5., 0., W*5.))
        )

        gl.glDisable(self.grass.target)

        if self.turn_flag:
            self.leader_list = rl_loop.check_leader(self.auto_anim_x, self.human_anim_x)
        else:
            self.leader_list = [0 for i in range(5)]


        #print(leader_list)

        for lane in self.lanes:
            self.draw_lane_surface(lane)
        for lane in self.lanes:
            self.draw_lane_lines(lane)

        for obj in self.objects:
            self.draw_object(self.obj_anim_x[obj])

        for no, car in enumerate(self.human_cars):
            self.draw_car(self.human_anim_x[no], car.color)

        for no, car in enumerate(self.auto_cars):
            # if self.leader_list[no] == 1:
            #     self.draw_car(self.auto_anim_x[no], 'purple')
            # else:
            self.draw_car(self.auto_anim_x[no], car.color)

        for no, car in enumerate(self.mock):
            if self.mock_anim_x[no][1] >= -0.8:
                self.draw_car(self.mock_anim_x[no], car.color)


        # for car in self.SEKIRO:
        #     self.draw_car(car, 'yellow', opacity=50)


        # if self.heat is not None:
        #     self.draw_heatmap()
        gl.glPopMatrix()

    def reset(self):
        self.timerounds = 0
        self.finalcnnt += 1
        print(self.finalcnnt)

        if self.sec_turn_flag == 0:
            self.mock.append(Car2([0.17, -0.8, -math.pi / 2., 0.], velocity=0.005, aggr=1, car_no=0,color=COLOR_RANDOM[random.randint(0,len(COLOR_RANDOM)-1)]))
            self.mock_anim_x.append(self.mock[0].x0)

        for no, car in enumerate(self.auto_cars):
            #car.reset([0, -0.30 + 0.2*no, math.pi/2., 0.])
            self.auto_anim_x[no] = car.x0

        for no, car in enumerate(self.human_cars):
            self.human_anim_x[no] = car.x0

        for obj in self.objects:
            self.obj_anim_x[obj] = obj.x0

        for no, car in enumerate(self.mock):
            if car.velocity < 0.004:
                car.velocity = 0.004

    def right_turn(self):
        state = rl_loop.state_rec(self.mock_anim_x, self.mock, )

        temp_speed = self.controller.control_go(state, self.human_anim_x[0][1], self.sec_turn_flag, self.historical_mock)
        self.doublecheck.pop(0)
        self.doublecheck.append(temp_speed)

        if self.arrive_flag == 0:
            if temp_speed < 0.02 :
                temp_speed = 0.02
            elif temp_speed > 0.05:
                temp_speed = 0.05
        else:
            if temp_speed < -0.01:
                temp_speed = -0.01
            elif temp_speed > 0.05:
                temp_speed = 0.05

        average = sum(self.doublecheck)/len(self.doublecheck)
        check_flag = 0

        for i in self.doublecheck:
            if abs(average) > 0.1:
                check_flag = 1
                break
            if abs(i)> 0.1:
                check_flag = 1
                break
            if abs(i-average) > 0.05:
                check_flag = 1
                break

        if (self.human_anim_x[0][2] <= -math.pi / 2. + math.pi / 36.) \
                and (abs(self.human_anim_x[0][0]) < 0.01) and (self.sec_turn_flag == 0):
            self.human_cars[0].reset()
            self.arrive_flag = 1

        if (check_flag == 0) and (self.human_anim_x[0][2] <= -math.pi / 2. + math.pi / 36.) \
                and (abs(self.human_anim_x[0][0])<0.01) and (self.sec_turn_flag == 0):
            self.human_cars[0].reset()
            self.SEKIRO = []
            self.arrive_flag = 0
            self.sec_turn_flag = 1
            #

        if abs(self.human_anim_x[0][0] - 0.13 )<0.05:
            self.turn_flag = 0

        if self.human_anim_x[0][2] <= -math.pi / 2. and abs(self.human_anim_x[0][0] - 0.17 )<0.05 or self.human_anim_x[0][1]>0.7 \
                or self.human_anim_x[0][1]<-0.7:
            self.human_cars[0].reset()
            self.SEKIRO = []
            self.reset()
            self.turn_flag = 1
            self.arrive_flag = 0
            self.sec_turn_flag = 0


        #if self.human_anim_x[0][2] > -math.pi / 2. and temp_speed == 0: temp_speed = 0.04

        if abs(temp_speed / 5 - self.human_cars[0].velocity) < 0.0004:
            self.human_cars[0].velocity = temp_speed / 5
        else:
            if temp_speed / 5 - self.human_cars[0].velocity > 0:
                self.human_cars[0].velocity = 0.0004 + self.human_cars[0].velocity
            else:
                self.human_cars[0].velocity = -0.0004 + self.human_cars[0].velocity



        if self.arrive_flag == 0:
            if self.human_cars[0].velocity > 0.01: self.human_cars[0].velocity = 0.01
            #self.human_cars[0].velocity = 0.01
            rl_loop.qnn_training2(self.auto_anim_x, self.human_anim_x,
                                  self.auto_cars, self.human_cars, 1)
        else:
            if self.human_cars[0].velocity > 0.007: self.human_cars[0].velocity = 0.007
            temp_list = self.human_anim_x[0]
            self.human_anim_x[0] =  [temp_list[0], temp_list[1] + self.human_cars[0].velocity, temp_list[2], temp_list[3]]
        return self.human_cars[0].velocity

    def animation_loop(self, _):
        self.rounds += 1
        #print(self.timerounds)
        #print(self.arrive_flag)
        if self.arrive_flag == 1:
            self.timerounds += 1
        speed_TV = 0
        #print(self.rounds)
        if self.rounds == 1500000:
            self.rounds = 0


        if self.rounds > 10:
            self.stop_flag = 0
        alpha = 0

        #self.human_cars[0].velocity = 0
        temp_list = self.human_anim_x[0]
        # if self.keys[key.UP]:
        #     self.human_cars[0].velocity = 0.03
        #     self.human_cars[0].dyn.gamma = 0
        #     self.human_cars[0].reset()
        #     self.SEKIRO = []
        #     print(time.time() - fuck)
        # elif self.keys[key.DOWN]:
        #     self.human_cars[0].velocity = -0.01
        #     self.human_cars[0].reset()
        #     self.SEKIRO = []

        turn_v = 0
        if self.keys[key.LEFT]:
            self.human_cars[0].reset()
            self.SEKIRO = []

            self.reset()
            self.turn_flag = 1
            self.sec_turn_flag = 0
        elif self.keys[key.RIGHT]:
            self.turn_flag = 1

        if self.turn_flag == 1 or self.sec_turn_flag == 1:
            speed_TV = self.right_turn()

        # if self.keys[key.S]:
        #     for no, car in enumerate(self.auto_cars):
        #         car.save(no)

        for no, car in enumerate(self.human_cars):
            temp_list = self.human_anim_x[no]
            offset_list = car.dyn.update_dyn(car.velocity)
            self.human_anim_x[no] = [temp_list[0] + offset_list[0], temp_list[1] + offset_list[1],
                                    -math.pi / 2 + car.dyn.phi, temp_list[3]]
            if len(self.SEKIRO) < 100:
                self.SEKIRO.append(self.human_anim_x[no])

        if self.keys[key.X]:
            if self.turn_flag == 0:
                self.turn_flag = 1
                for i in range(3):
                    self.auto_cars[i].restore([1, 1])

        if self.keys[key.C]:
            self.rounds = 0
            self.stop_flag = 0


            # if self.turn_flag == 1:
            #     self.turn_flag = 0
            #     for i in range(3):
            #         self.auto_cars[i].restore([1, 1])
        if self.stop_flag == 0:
            for i in range(0,5):
                rl_loop.qnn_training(self.auto_anim_x, self.human_anim_x,
                                     self.auto_cars, self.human_cars, i, self.leader_list, speed_TV)

            for i in range(0, len(self.mock)):
                rl_loop.qnn_training3(self.mock_anim_x, self.mock, i, self.auto_cars, self.sec_turn_flag,
                                      self.human_anim_x, self.human_cars)

            for no, car in enumerate(self.auto_cars):
                temp_list = self.auto_anim_x[no]
                self.auto_anim_x[no] = [temp_list[0], temp_list[1] + car.velocity, temp_list[2], temp_list[3]]


            self.historical_mock.append([self.mock, self.mock_anim_x])
            if len(self.historical_mock) == 4:
                self.historical_mock.pop(0)

            for no, car in enumerate(self.mock):
                temp_list = self.mock_anim_x[no]
                self.mock_anim_x[no] = [temp_list[0], temp_list[1] + car.velocity, temp_list[2], temp_list[3]]

            temp_list = self.mock_anim_x[0]
            if (temp_list[1] > 1.2):
                self.mock_anim_x.pop(0)
                self.mock.pop(0)

            temp_list = self.mock_anim_x[-1]
            if temp_list[1] >= -0.8:
                self.mock.append(
                    Car2([0.17, temp_list[1] - self.mock[-1].carnext, -math.pi / 2., 0.], velocity=random.random() * 0.005 + 0.004, color=COLOR_RANDOM[random.randint(0, len(COLOR_RANDOM)-1)],aggr=1, car_no=self.rounds))
                self.mock_anim_x.append(self.mock[-1].x0)

            # rl_loop.state_rec(self.filepoint, self.mock_anim_x, self.mock,)
            # if time.time() - fuck > 0.02:
            #     rl_loop.state_rec(self.filepoint2, self.mock_anim_x, self.mock,)

            for obj in self.objects:
                temp_list = self.obj_anim_x[obj]
                self.obj_anim_x[obj] = [temp_list[0], temp_list[1] - 0.015]
                if self.obj_anim_x[obj][1]< -0.9:
                    self.obj_anim_x[obj][1] = 1.05
                self.draw_object(self.obj_anim_x[obj])

    def run(self, filename=None, pause_every=None):
        self.reset()
        pyglet.clock.schedule_interval(self.animation_loop, 0.02)
        self.event_loop.run()

