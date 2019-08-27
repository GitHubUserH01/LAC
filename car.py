
import tensorflow as tf
import numpy as np
import math as math
import random
from autograd import jacobian

MEMORY_CAPACITY = 200
S_DIM = 2
A_DIM = 1
A_BOUND = 0.005
BATCH_SIZE = 64
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.95     # reward discount
TAU = 0.01      # soft replacement
TS = 1


class Dyn():
    def __init__(self):
        self.gamma = 0  # steering angle
        self.eta = -0.17  # distance between CoG and the center of the lane
        self.abscissa = 0
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
    def __init__(self, x0, color='yellow', velocity=0, car_no=0, aggr = 1):
        self.color = color
        self.aggressive = aggr
        self.carnext = random.random()*0.4 - 0.2 + 0.7 * 1/0.8
        self.velocity = 0

        self.x0 = x0
        self.velocity = velocity
        self.graph = tf.Graph()
        self.graph2 = tf.Graph()
        self.state = None
        self.state2 = None
        self.action = None
        self.distance = None
        self.dyn = Dyn()
        self.var = 0

        if car_no == 5:
            self.a_dim, self.s_dim, self.a_bound = A_DIM, S_DIM + 1, math.pi/4
            self.var = 0.01
        else:
            self.a_dim, self.s_dim, self.a_bound = A_DIM, S_DIM, A_BOUND

        self.memory = np.zeros((MEMORY_CAPACITY, self.s_dim * 2 + self.a_dim + 1), dtype=np.float32)
        self.pointer = 0

        with self.graph.as_default():
            self.S = tf.placeholder(tf.float32, [None, self.s_dim], 's')
            self.S_ = tf.placeholder(tf.float32, [None, self.s_dim], 's_')
            self.R = tf.placeholder(tf.float32, [None, 1], 'r')

            def _build_a(s, reuse=None, custom_getter=None):
                trainable = True if reuse is None else False
                with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
                    if car_no != 5:
                        net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
                    else:
                        net = tf.layers.dense(s, 32, activation=tf.nn.sigmoid, name='l1', trainable=trainable)
                        net = tf.layers.dense(net, 32, activation=tf.nn.sigmoid, name='l2', trainable=trainable)
                    a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
                    return tf.multiply(a, self.a_bound, name='scaled_a')

            def _build_c(s, a, reuse=None, custom_getter=None):
                trainable = True if reuse is None else False
                with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
                    if car_no !=5:
                        n_l1 = 30
                    else:
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

            a_loss = - tf.reduce_mean(q)  # maximize the

            self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

            with tf.control_dependencies(target_update):  # soft replacement happened at here
                q_target = self.R + GAMMA * q_
                td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
                self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

            self.session = tf.Session(graph=self.graph)
            self.saver = tf.train.Saver()

        with self.graph2.as_default():
            self.S2 = tf.placeholder(tf.float32, [None, self.s_dim], 's')
            self.S_2 = tf.placeholder(tf.float32, [None, self.s_dim], 's_')
            self.R2 = tf.placeholder(tf.float32, [None, 1], 'r')

            def _build_a(s, reuse=None, custom_getter=None):
                trainable = True if reuse is None else False
                with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
                    net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
                    a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
                    return tf.multiply(a, self.a_bound, name='scaled_a')

            self.a2 = _build_a(self.S2, )
            q = _build_c(self.S2, self.a2, )
            a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
            c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
            ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

            def ema_getter(getter, name, *args, **kwargs):
                return ema.average(getter(name, *args, **kwargs))

            target_update = [ema.apply(a_params), ema.apply(c_params)]  # soft update operation
            a_ = _build_a(self.S_2, reuse=True, custom_getter=ema_getter)  # replaced target parameters
            q_ = _build_c(self.S_2, a_, reuse=True, custom_getter=ema_getter)

            a_loss = - tf.reduce_mean(q)  # maximize the q
            self.atrain2 = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

            with tf.control_dependencies(target_update):  # soft replacement happened at here
                q_target = self.R2 + GAMMA * q_
                td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
                self.ctrain2 = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

            self.session2 = tf.Session(graph=self.graph2)
            self.saver2 = tf.train.Saver()

        if car_no == 5:
            self.restore(None)
        else:
            self.restore([1, 1])

        # fuck = self.session.run([tf.gradients(self.a, self.S)],feed_dict={self.S: np.array([0.4,0.01])[np.newaxis, :]})
        # print('fcuk')

    def choose_action(self, s,s2):
        temp_a = 0
        temp_b = 0
        if not s is None:
            temp_a = self.session.run(self.a, {self.S: np.array(s)[np.newaxis, :]})[0]
        if not s2 is None:
            temp_b = self.session2.run(self.a2, {self.S2: np.array(s2)[np.newaxis, :]})[0]
        return ( -temp_a + temp_b)

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        self.session.run(self.atrain, {self.S: bs})
        self.session.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def reset(self):
        #self.velocity = 0
        self.action = np.array([0 for i in range(self.a_dim)])
        self.distance = None
        self.dyn.gamma = 0
        self.dyn.phi = 0
        self.dyn.beta = 0
        self.dyn.eta = -0.17
        self.dyn.abscissa = 0

    def save(self, car_no):
        self.saver.save(self.session, "./Motion_prediction/save_2net{}.ckpt".format(car_no))

    def restore(self, network_list):
        """
        restore the network parameters
        :param network_list: [1, 2] The first no. denotes the network for the car ahead.
                                    1 denotes larger reward regards with distance!!!
        :return:
        """

        if network_list is None:
            self.saver.restore(self.session, "./Motion_prediction/Model2/save_net2.ckpt")
            #self.saver2.restore(self.session2, "./Motion_prediction/save_1net1.ckpt")
            return

        self.saver.restore(self.session, "./Motion_prediction/save_{}net1.ckpt".format(network_list[0]))
        self.saver2.restore(self.session2, "./Motion_prediction/save_{}net1.ckpt".format(network_list[1]))


class Car2(object):
    def __init__(self, x0, color='yellow', velocity=0, car_no=0, aggr = 1):
        self.car_no = car_no
        self.color = color
        self.aggressive = aggr
        self.carnext = random.random()*0.4 - 0.2 + 0.4 * 1/0.8
        self.velocity = 0
        self.setvelocity = velocity
        self.x0 = x0
        self.velocity = velocity
        self.graph = tf.Graph()
        self.graph2 = tf.Graph()
        self.state = None
        self.state2 = None
        self.action = None
        self.distance = None
        self.var = 0
        self.acceleration = 0
        self.v_length = 0.7
        self.v_width = 1
        self.v_class = 1


class Obj:
    def __init__(self, x0):
        self.x0 = x0
