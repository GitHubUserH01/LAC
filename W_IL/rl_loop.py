import gym
import numpy as np
from random import random
import tensorflow as tf
import matplotlib.pyplot as plt
import car
import math as math

def check_leader(auto_anim_x, human_anim_x):
    """
    decide the leadership among the agents
    :return: leader_list [0,...,1,...0] 0 denotes fools, 1 denotes leader
    """

    leader_list = [0 for i in range(len(auto_anim_x))]
    pos_human = human_anim_x[0][1]

    pos_list = []

    for pos in auto_anim_x:
        pos_list.append(auto_anim_x[pos][1])

    for no,i in enumerate(pos_list):
        if (i > pos_human) & (no == 0):
            break
        if (i < pos_human) & (no == len(pos_list)-1):
            break

        if i > pos_human:
            leader_list[no] = 1
            leader_list[no-1] = 1
            break

    return leader_list


# def qnn_setup
#
#
def qnn_training(auto_anim_x, human_anim_x, auto_cars, human_cars, car_no):
    """
     qnn training
     :return: None but directly set the velocity of auto_cars
     """

    # create lists to contain total rewards and steps per episode
    # 创建列表以包含每个episode对应的总回报与总步数。
    jList = []
    rList = []

    chosen_car = auto_cars[car_no]

    # Set learning parameters
    # 设置学习参数
    y = .9
    e = chosen_car.random_rounds / 10

    if car_no == 0:
        s_distance = human_anim_x[0][1] - auto_anim_x[car_no][1]
    else:
        s_distance = -human_anim_x[0][1] + auto_anim_x[car_no][1]

    s_v_human = human_cars[0].velocity
    s_v_auto = auto_cars[car_no].velocity

    # if s_distance>0:
    if chosen_car.distance is None:
        chosen_car.distance = s_distance

    reward = -1 * (0.3 - s_distance) ** 2 - 1 * chosen_car.velocity ** 2 - 1 * chosen_car.action[0] ** 2

    # print(reward)

    # chosen_car.distance = s_distance
    # else:
    #     reward = -1
    # The Q-Network
    # Q网络
    # Choose an action by greedily (with e chance of random action) from the Q-network
    # 基于Q网络的输出结果，贪婪地选择一个行动（有一定的概率选择随机行动）
    # action = [0] 代表0号行动， 0, 1, 2 代表减速，保持，加速
    def encoder(s_distance, s_v_human, s_v_auto):
        if car_no == 0:
            return [s_distance, s_v_human - s_v_auto]
        else:
            return [s_distance, -s_v_human + s_v_auto]

    if chosen_car.state is None:
        chosen_car.state = encoder(s_distance, s_v_human, s_v_auto)

    chosen_car.store_transition(chosen_car.state, chosen_car.action, [reward], encoder(s_distance, s_v_human, s_v_auto))

    if chosen_car.pointer > car.MEMORY_CAPACITY:
        chosen_car.var *= .9995  # decay the action randomness
        chosen_car.learn()

    chosen_car.state = encoder(s_distance, s_v_human, s_v_auto)
    # if np.random.rand(1) < e:
    #     temp = np.random.rand(1)
    #     if temp<0.33:
    #         chosen_car.action[0] = 0
    #     elif temp < 0.67:
    #         chosen_car.action[0] = 1
    #     else:
    #         chosen_car.action[0] = 2
    a = chosen_car.choose_action(np.array(chosen_car.state))
    chosen_car.action = np.clip(np.random.normal(a, chosen_car.var), -2,
                                2)  # add randomness to action selection for exploration

    chosen_car.velocity = chosen_car.velocity + chosen_car.action[0]
    if chosen_car.velocity < -0.01:
        chosen_car.velocity = -0.01

    elif chosen_car.velocity > 0.01:
        chosen_car.velocity = 0.01


def qnn_training2(auto_anim_x, human_anim_x, auto_cars, human_cars, car_no, flag):
    """
    qnn training
    :return: None but directly set the velocity of auto_cars
    """

    #create lists to contain total rewards and steps per episode
    # 创建列表以包含每个episode对应的总回报与总步数。
    jList = []
    rList = []

    chosen_car = auto_cars[car_no]
    # Set learning parameters
    # 设置学习参数
    y = .9
    e = chosen_car.random_rounds/10
    chosen_car.velocity = 0.01 * random()
    s_distance = chosen_car.dyn.eta
    #print(s_distance)
    s_steer_human = chosen_car.dyn.gamma
    s_phi_human = chosen_car.dyn.phi

    if chosen_car.distance is None:
        chosen_car.distance = s_distance

    # reward_ex = np.zeros([1, 4])
    # reward_ex[0, 0] += (s_distance * 2)**2
    # reward_ex[0, 1] += (s_steer_human * 4 / math.pi)** 1
    # reward_ex[0, 2] += (s_phi_human/(2*math.pi)) ** 1
    # reward = chosen_car.session2.run(chosen_car.w2_s, {chosen_car.S2: reward_ex})

    #print(reward)
    #
    reward = chosen_car.w_irl[0]*(s_distance * 2)**2\
              + chosen_car.w_irl[2] * (s_steer_human * 4 / math.pi)** 2 \
             + chosen_car.w_irl[1] * (s_phi_human/(2*math.pi)) ** 2

    if chosen_car.state is None:
        chosen_car.state = [s_distance, s_phi_human, chosen_car.velocity]

    refresh_flag = chosen_car.store_transition(chosen_car.state, chosen_car.action, [reward], [s_distance, s_phi_human, chosen_car.velocity])

    if chosen_car.pointer > car.MEMORY_CAPACITY:
        chosen_car.var *= .9995  # decay the action randomness
        if flag == 0:
            temp = chosen_car.learn(refresh_flag= 0)
            #print(temp)
            for i in range(3):
                chosen_car.w_irl[i] = temp[i, 0]

    chosen_car.state = [s_distance, s_phi_human, chosen_car.velocity]

    a = chosen_car.choose_action(np.array(chosen_car.state))
    chosen_car.action = np.random.normal(a, chosen_car.var)  # add randomness to action selection for exploration

    chosen_car.dyn.gamma = chosen_car.action[0]


def analysis_func(auto_cars, car_no):
    chosen_car = auto_cars[car_no]
    data_label = chosen_car.expert2
    sum = 0
    for i in range(len(data_label)):
        state = [data_label.at[i,'Local_X'], data_label.at[i,'phi'], data_label.at[i,'v_Vel']]
        a = chosen_car.choose_action(np.array(state))[0]
        sum += (a - data_label.at[i,'steer'])**2

    print(sum/len(data_label))
    return (sum/len(data_label))





