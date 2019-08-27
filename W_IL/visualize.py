import pyglet
import pyglet.gl as gl
import pyglet.graphics as graphics
import numpy as np
import time
import math
from pyglet.window import key
import rl_loop
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import sys
PNG_PATH = '../PNG/'


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
        self.obj_anim_x = {}
        self.prev_x = {}
        self.vel_list = []
        self.main_car = None
        self.sess = None
        self.rounds = 0
        self.irl_listx = []
        self.irl_listy = []
        self.irl_phi = []
        self.irl_steer = []
        self.irl_flag = 0
        self.filecount = 0
        self.recordflag = 0
        self.SEKIRO = []
        self.srounds = 0
        self.expertflag = 0
        self.totalrounds = 0
        self.recordsnum = 0
        self.stop_flag = bool(False)

        def centered_image(filename):
            img = pyglet.resource.image(filename)
            img.anchor_x = img.width/2.
            img.anchor_y = img.height/2.
            return img

        def car_sprite(color, scale=0.15/600.):
            sprite = pyglet.sprite.Sprite(centered_image( 'car-{}.png'.format(color)), subpixel=True)
            sprite.scale = scale
            return sprite

        def object_sprite(name, scale=0.15/900.):
            sprite = pyglet.sprite.Sprite(centered_image( '{}.png'.format(name)), subpixel=True)
            sprite.scale = scale
            return sprite

        self.sprites = {c: car_sprite(c) for c in ['red', 'yellow', 'purple', 'white', 'orange', 'gray', 'blue']}
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
        sprite.rotation = x[2]*180/math.pi
        sprite.opacity = x[3]
        sprite.draw()

    def draw_scar(self, x, color='yellow', opacity=255):
        sprite = self.sprites[color]
        sprite.x, sprite.y = x[0], x[1]
        sprite.rotation = x[2]*180/math.pi
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

        leader_list = rl_loop.check_leader(self.auto_anim_x, self.human_anim_x)
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
            if leader_list[no] == 1:
                self.draw_car(self.auto_anim_x[no], 'blue')
            else:
                self.draw_car(self.auto_anim_x[no], car.color)

        for car in self.SEKIRO:
            self.draw_scar(car, 'white', opacity=50)
        # if self.heat is not None:
        #     self.draw_heatmap()
        gl.glPopMatrix()

    def reset(self):
        for no,car in enumerate(self.auto_cars):
            car.reset()
            self.auto_anim_x[no] = car.x0

        for no,car in enumerate(self.human_cars):
            self.human_anim_x[no] = car.x0

        for obj in self.objects:
            self.obj_anim_x[obj] = obj.x0

        if self.auto_cars[2].end_index != -1:
            self.auto_cars[2].start_index = (self.auto_cars[2].end_index + 1) % 200

        self.auto_cars[2].end_index =  self.auto_cars[2].pointer

        self.irl_listx = []
        self.irl_listy = []
        self.irl_phi = []
        self.irl_steer = []
        self.recordflag = 0
        self.auto_cars[2].learn(refresh_flag=1)
        if self.stop_flag is False:
            self.SEKIRO = []
        self.rounds = 0

    def animation_loop(self, _):
        self.rounds += 1
        self.totalrounds += 1
        #print(self.rounds)
        if self.rounds == 150:
            #print(self.rounds)
            self.reset()
            if self.expertflag == 1:
                expert_df = pd.DataFrame()
                expert_df['Local_Y'] = pd.Series(self.irl_listy)
                expert_df['Local_X'] = pd.Series(self.irl_listx)
                expert_df['v_Vel'] = 0.01
                expert_df['phi'] = pd.Series(self.irl_phi)
                expert_df['steer'] = pd.Series(self.irl_steer)

                sys.exit(0)


            if self.irl_flag == 1:
                self.expertflag = 1
            self.rounds = 0

        if self.totalrounds % 100 == 0:
            self.totalrounds = 0
            self.recordsnum += 1
            wasser_est = self.auto_cars[2].learn(see_flag=1)
            print(self.auto_cars[2].w_irl)
            print('Iteration {}:'.format(self.recordsnum))
            print('Wass: '+ str(wasser_est[0]))
            # print('Parameters: ')
            # print((wasser_est[1]))
            error = rl_loop.analysis_func(auto_cars=self.auto_cars, car_no=2)
            if self.recordsnum > 100:
                sys.exit(0)

        alpha = 0
        leader_list = rl_loop.check_leader(self.auto_anim_x, self.human_anim_x)
        #
        # self.human_cars[0].velocity = 0
        # temp_list = self.human_anim_x[0]
        # if self.keys[key.UP]:
        #     offset_list = self.human_cars[0].dyn.update_dyn(0.01, 0)
        #     self.human_anim_x[0] = [temp_list[0] + offset_list[0], temp_list[1] + offset_list[1],
        #                             -math.pi/2 + self.human_cars[0].dyn.phi, temp_list[3]]
        #     print(self.human_cars[0].dyn.phi)
        #
        # elif self.keys[key.DOWN]:
        #     self.human_cars[0].velocity = -0.01
        #
        # turn_v = 0
        # if self.keys[key.RIGHT]:
        #     self.human_cars[0].velocity = 0.05
        #     offset_list = self.human_cars[0].dyn.update_dyn(0.01, math.pi/18)
        #     self.human_anim_x[0] = [temp_list[0] + offset_list[0], temp_list[1] + offset_list[1], -math.pi/2 + self.human_cars[0].dyn.phi, temp_list[3]]
        #
        # if self.keys[key.LEFT]:
        #     self.human_cars[0].velocity = 0.05
        #     offset_list = self.human_cars[0].dyn.update_dyn(0.01, -math.pi/18)
        #     self.human_anim_x[0] = [temp_list[0] + offset_list[0], temp_list[1] + offset_list[1], -math.pi/2 + self.human_cars[0].dyn.phi, temp_list[3]]

        if self.keys[key.S]:
            for no, car in enumerate(self.auto_cars):
                car.save(no)

        if self.keys[key.C]:
            self.stop_flag = False

        if self.keys[key.V]:
            self.stop_flag = True

        #
        #rl_loop.qnn_training(self.auto_anim_x, self.human_anim_x, self.auto_cars, self.human_cars, 0)
        #
       # rl_loop.qnn_training(self.auto_anim_x, self.human_anim_x, self.auto_cars, self.human_cars, 1)

        rl_loop.qnn_training2(self.auto_anim_x, self.human_anim_x, self.auto_cars, self.human_cars, 2,self.irl_flag)

        for no, car in enumerate(self.auto_cars):
            temp_list = self.auto_anim_x[no]
            offset_list = car.dyn.update_dyn(car.velocity)
            self.auto_anim_x[no] = [temp_list[0] + offset_list[0], temp_list[1] + offset_list[1],
                                    -math.pi / 2 + car.dyn.phi, temp_list[3]]
            if no == 2:
                if temp_list[0] <-0.2 or temp_list[0] > 0.2 or temp_list[1]< -0.4 or temp_list[1] > 0.4:
                    self.reset()
                self.irl_listx.append(temp_list[0] + offset_list[0])
                self.irl_listy.append(temp_list[1] + offset_list[1])
                self.irl_phi.append(car.dyn.phi)
                self.irl_steer.append(car.dyn.gamma)
                if self.stop_flag is False:
                    self.SEKIRO.append(self.auto_anim_x[no])

            if abs(temp_list[0] + offset_list[0]) < 0.05 and abs(car.dyn.phi)<math.pi / 36. and no==2 \
                    and self.recordflag == 0:
                self.recordflag = 1
                self.srounds += 1
                compare_list = []
                axis_list = []
                search_no = 0
                for i in range(len(car.expert)):
                    temp = car.expert.at[i, 'Local_Y']
                    while search_no < len(self.irl_listx) - 1:
                        if temp <= self.irl_listy[search_no+1]:
                            axis_list.append(temp)
                            compare_list.append((self.irl_listx[search_no] + self.irl_listx[search_no+1])/2)
                            break
                        search_no += 1

                sum_err = 0
                #print(len(axis_list))
                #pd.DataFrame([axis_list,compare_list]).to_csv('./Traj/traj{}'.format(self.filecount),index=None)
                self.filecount += 1
                for i in range(min(len(car.expert),len(compare_list))):
                    sum_err += ((car.expert.at[i,'Local_X'] - compare_list[i])*100) **2
                if len(compare_list) == 0:
                    sum_err = 999
                #print(len(compare_list))
                # print(sum_err)
                # if sum_err < 500:
                #     self.irl_flag = 1
                #     self.rounds = 149
                #     print('fuck')
                #self.reset()


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

