
import tensorflow as tf
import numpy as np
import math
import pandas as pd
from keras.constraints import maxnorm
from keras.backend import gradients

MEMORY_CAPACITY = 200
S_DIM = 3
A_DIM = 1
A_BOUND = 0.005
BATCH_SIZE = 32
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.95     # reward discount
TAU = 0.01      # soft replacement
TS = 1


class Dyn():
    def __init__(self):
        self.gamma = 0  # steering angle
        self.eta = -0.17  # distance between CoG and the center of the lane
        self.abscissa = -0.3
        self.beta = math.atan(0.5 * math.tan(self.gamma))
        self.phi = 0
        self.lr = 0.06
        self.lf = 0.06

    def update_dyn(self, velocity):
        self.beta = math.atan(0.5 * math.tan(self.gamma))
        self.phi += velocity * TS * math.sin(self.beta) / self.lr
        x_offset = velocity * TS * math.sin(self.phi + self.beta)
        y_offset = velocity * TS * math.cos(self.phi + self.beta)
        self.eta += x_offset
        self.abscissa += y_offset
        return [x_offset, y_offset]


class Car(object):
    def __init__(self, x0, color='yellow', velocity=0, trained=0, car_no=0):
        self.color = color
        self.velocity = 0
        self.dyn = Dyn()
        self.x0 = x0
        self.velocity = velocity
        self.graph = tf.Graph()
        self.graph2 = tf.Graph()
        self.random_rounds = 10
        self.state = None

        self.action = None
        self.distance = None
        self.w_irl = [0,0,0,0]
        self.parameter_w = [[0],[0],[0],[0]]
        self.start_index = 0
        self.end_index = -1
        if car_no == 2:
            self.expert = pd.read_csv('./toy/expert1.csv', index_col=None, sep=',')

            data_list = []
            for i in range(9,10):
                data_list.append(pd.read_csv('./toy/expert2.csv'.format(i), index_col=None, sep=','))
            self.expert2 = pd.concat(data_list,axis=0).reset_index(drop=True)
            self.expert2 = pd.read_csv('./toy/expert2.csv', index_col=None, sep=',')
            #self.expert2 = self.expert
        if trained:
            self.var = 0
        else:
            self.var = 0.1
            if car_no == 2:
                self.var = math.pi/3

        if car_no != 10:
            with self.graph.as_default():

                self.pointer = 0
                if car_no == 2:
                    self.a_dim, self.s_dim, self.a_bound = A_DIM , S_DIM , math.pi/4,
                else:
                    self.a_dim, self.s_dim, self.a_bound = A_DIM, S_DIM, A_BOUND,

                self.memory = np.zeros((MEMORY_CAPACITY, self.s_dim * 2 + self.a_dim + 1), dtype=np.float32)
                self.S = tf.placeholder(tf.float32, [None, self.s_dim], 's')
                self.S_ = tf.placeholder(tf.float32, [None, self.s_dim], 's_')
                self.R = tf.placeholder(tf.float32, [None, 1], 'r')

                def _build_a(s, reuse=None, custom_getter=None):
                    trainable = True if reuse is None else False
                    with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
                        net = tf.layers.dense(s, 32, activation=tf.nn.sigmoid, name='l1', trainable=trainable)
                        net = tf.layers.dense(net, 32, activation=tf.nn.sigmoid, name='l2', trainable=trainable)
                        a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
                        return tf.multiply(a, self.a_bound, name='scaled_a')

                def _build_c(s, a, reuse=None, custom_getter=None):
                    trainable = True if reuse is None else False
                    with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
                        n_l1 = 32
                        w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
                        w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
                        b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
                        net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
                        return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

                self.a = _build_a(self.S, )
                q = _build_c(self.S, self.a, )
                a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
                c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
                ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

                def ema_getter(getter, name, *args, **kwargs):
                    return ema.average(getter(name, *args, **kwargs))

                target_update = [ema.apply(a_params), ema.apply(c_params)]  # soft update operation
                a_ = _build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
                q_ = _build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

                a_loss = - tf.reduce_mean(q)  # maximize the q
                self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

                with tf.control_dependencies(target_update):  # soft replacement happened at here
                    q_target = self.R + GAMMA * q_
                    td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
                    self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

                self.session = tf.Session(graph=self.graph)
                self.saver = tf.train.Saver()
                if trained == 0:
                    init = tf.initialize_all_variables()
                    self.session.run(init)
                else:
                    self.restore(car_no)

            with self.graph2.as_default():
                self.S2 = tf.placeholder(tf.float32, [None, 4])
                self.para = tf.placeholder(tf.float32, shape=[4,1])
                self.w2_s = tf.get_variable('w2_s', shape=[4, 1], trainable=True,
                                            initializer=tf.constant_initializer([[-1.], [-0.03], [-0.02], [0]]))
                self.reward_gap = tf.matmul(self.S2, self.w2_s)
                self.rtrain = tf.train.RMSPropOptimizer(0.0001).minimize(self.reward_gap)
                self.assing_w = self.w2_s.assign(self.para)
                self.session2 = tf.Session(graph=self.graph2)
                init = tf.initialize_all_variables()
                self.session2.run(init)

    def choose_action(self, s):
        return self.session.run(self.a, {self.S: s[np.newaxis, :]})[0]


    def learn(self, refresh_flag=0, see_flag=0):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.session.run(self.atrain, {self.S: bs})
        self.session.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
#        self.parameter_w = self.session2.run(self.w2_s)

        if self.pointer >= self.end_index:
            bt = self.memory[self.start_index:self.end_index, :]
        else:
            bt = (self.memory[self.end_index:MEMORY_CAPACITY, :])
            bt2 = self.memory[0:self.start_index, :]
            bt = np.concatenate((bt, bt2), axis=0)
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        lenth_compare = min(len(bt), len(self.expert))
        self.parameter_w = self.session2.run(self.w2_s)
        if refresh_flag == 1:
            reward_ex = np.zeros([1, 4])

            for i in range(lenth_compare):
                reward_ex[0, 0] = (GAMMA ** i) * (bs[i,0]*2)**2
                reward_ex[0, 1] = (GAMMA ** i) *(bs[i,1]* 4 / math.pi)**2
                reward_ex[0, 2] = (GAMMA ** i) *(ba[i,0]/(2*math.pi))**2
                self.session2.run(self.rtrain, {self.S2: reward_ex})

            for i in range(lenth_compare):
                reward_ex[0, 0] = (GAMMA ** i) * (self.expert.at[i,'Local_X']*2)**2
                reward_ex[0, 1] = (GAMMA ** i) * (self.expert.at[i,'phi']* 4 / math.pi)**2
                reward_ex[0, 2] = (GAMMA ** i) * (self.expert.at[i,'steer']/(2*math.pi))**2
                self.session2.run(self.rtrain, {self.S2: -reward_ex})

            self.parameter_w = self.session2.run(self.w2_s)
            for i in range(4):
                if self.parameter_w[i,0] > -0.1:
                    self.parameter_w[i,0] = -0.1
                elif self.parameter_w[i,0] < -1:
                    self.parameter_w[i, 0] = -1
            self.session2.run(self.assing_w,{self.para: self.parameter_w})

        if see_flag == 1:
            reward_w = 0
            reward_ex = np.zeros([1, 4])
            for i in range(lenth_compare):
                reward_ex[0,0] -= (GAMMA ** i) * (bs[i, 0] * 2) ** 2
                reward_ex[0,1] -= (GAMMA ** i) * (bs[i, 1] * 4 / math.pi) ** 2
                reward_ex[0,2] -= (GAMMA ** i) * (ba[i, 0] / (2 * math.pi)) ** 2

            for i in range(lenth_compare):
                reward_ex[0, 0] += (GAMMA ** i) * (self.expert.at[i, 'Local_X'] * 2) ** 2
                reward_ex[0, 1] += (GAMMA ** i) * (self.expert.at[i, 'phi'] * 4 / math.pi) ** 2
                reward_ex[0, 2] += (GAMMA ** i) * (self.expert.at[i, 'steer'] / (2 * math.pi)) ** 2

            reward_w = reward_ex[0,0] * self.parameter_w[0] +  reward_ex[0,1] * self.parameter_w[1] + reward_ex[0,2] * self.parameter_w[2]

            return [reward_w[0], [0]]
        return (self.parameter_w)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        if index == 0:
            flag = 1
        else:
            flag = 0
        return flag

    def reset(self):
        self.velocity = 0
        self.action = np.array([0 for i in range(self.a_dim)])
        self.distance = None
        self.dyn.gamma = 0
        self.dyn.phi = 0
        self.dyn.beta = 0
        self.dyn.eta = -0.17
        self.dyn.abscissa = -0.3

    def save(self, car_no):
        self.saver.save(self.session, "./Model/save_net{}.ckpt".format(car_no))

    def restore(self,car_no):
        if car_no == 3:
            self.saver.restore(self.session, "./Model/save_2net2.ckpt")
        self.saver.restore(self.session, "./Model/save_net{}.ckpt".format(car_no))

class Obj:
    def __init__(self,x0):
        self.x0=x0
