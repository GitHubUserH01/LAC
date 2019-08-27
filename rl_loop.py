import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import car
import pandas as pd
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
def qnn_training(auto_anim_x, human_anim_x, auto_cars, human_cars, car_no,leader_list,speed_TV):
    """
    qnn training
    :return: None but directly set the velocity of auto_cars
    """

    chosen_car = auto_cars[car_no]


    # s_distance is the distance from the car ahead
    loc_list = [auto_anim_x[i][1] for i in range(5)]
    vel_list = [auto_cars[i].velocity for i in range(5)]

    if 1 in leader_list:
        loc_list.append(human_anim_x[0][1])
        vel_list.append(human_cars[0].velocity)

    state_df = pd.DataFrame({'loc': loc_list, 'vel': vel_list})
    state_df.sort_values(by='loc', inplace=True)

    loc_list = list(state_df['loc'])
    vel_list = list(state_df['vel'])

    new_index = loc_list.index(auto_anim_x[car_no][1])

    if 1 in leader_list:
        TV_index = loc_list.index(human_anim_x[0][1])
    else:
        TV_index = -1

    speed_TV = 1
    if new_index == 0 or (TV_index == new_index - 1 and speed_TV < 0):
        s_distance = (loc_list[new_index+1] - loc_list[new_index])
        s_distance2 = None
    elif new_index == len(loc_list) - 1 :
        s_distance = None
        s_distance2 = (loc_list[new_index] - loc_list[new_index-1])
    else:
        s_distance = (loc_list[new_index+1] - loc_list[new_index])
        #s_distance = None
        s_distance2 = (loc_list[new_index] - loc_list[new_index-1])
        #s_distance2 = None

    if s_distance is not None and s_distance2 is not None:
        if s_distance < s_distance2 - 0.1:
            s_distance2 = None
    # if 1 in leader_list:
    #     human_index = loc_list.index(human_anim_x[0][1])
    #     if (new_index != human_index - 1) & (new_index != human_index + 1):
    #         if s_distance is not None:
    #             s_distance *= 2
    #         if s_distance2 is not None:
    #             s_distance2 *= 2

    if s_distance is not None:
        s_distance /= 1.2
    if s_distance2 is not None:
        s_distance2 /= 1.2

    if new_index == 0:
        s_vel = vel_list[new_index+1] - vel_list[new_index]
        s_vel2 = None
    elif new_index == len(vel_list) - 1:
        s_vel = None
        s_vel2 = vel_list[new_index] - vel_list[new_index-1]
    else:
        s_vel = vel_list[new_index+1] - vel_list[new_index]
        s_vel2 = vel_list[new_index] - vel_list[new_index-1]

    #s_vel = s_vel2 = 0

    # if s_distance>0:
    if s_distance is None:
        chosen_car.state = None
    else:
        chosen_car.state = [s_distance, s_vel]

    if s_distance2 is None:
        chosen_car.state2 = None
    else:
        chosen_car.state2 = [s_distance2, s_vel2]

    a = chosen_car.choose_action(chosen_car.state, chosen_car.state2)

    chosen_car.action = np.clip(np.random.normal(a, chosen_car.var), -2, 2)  # add randomness to action selection for exploration

    chosen_car.velocity = chosen_car.velocity + chosen_car.action[0]

    # if s_distance2 is not None:
    #     if (s_distance2 < 0.1):
    #         chosen_car.velocity =  0.005
    #
    # if s_distance is not None:
    #     if (s_distance < 0.1):
    #         chosen_car.velocity =  - 0.005




    if (1 not in leader_list) & (car_no == 4):
        chosen_car.velocity = 0

    if chosen_car.velocity < -0.01:
        chosen_car.velocity = -0.01

    elif chosen_car.velocity > 0.01:
        chosen_car.velocity = 0.01


def qnn_training2(auto_anim_x, human_anim_x, auto_cars, human_cars, car_no):
    """
    qnn training
    :return: None but directly set the velocity of auto_cars
    """

    #create lists to contain total rewards and steps per episode
    # 创建列表以包含每个episode对应的总回报与总步数。
    jList = []
    rList = []

    chosen_car = human_cars[0]
    # Set learning parameters
    # 设置学习参数
    y = .9
    s_distance = chosen_car.dyn.eta / 1.1

    s_steer_human = chosen_car.dyn.gamma
    s_phi_human = chosen_car.dyn.phi

    # if s_distance>0:
    if chosen_car.distance is None:
        chosen_car.distance = s_distance

    if chosen_car.state is None:
        chosen_car.state = [s_distance, s_phi_human, chosen_car.velocity]

    chosen_car.state = [s_distance, s_phi_human, chosen_car.velocity]

    # if np.random.rand(1) < e:
    #     temp = np.random.rand(1)
    #     if temp<0.33:
    #         chosen_car.action[0] = 0
    #     elif temp < 0.67:
    #         chosen_car.action[0] = 1
    #     else:
    #         chosen_car.action[0] = 2

    a = -chosen_car.choose_action(chosen_car.state, None)
    chosen_car.action = np.random.normal(a, chosen_car.var)  # add randomness to action selection for exploration

    if car_no == 0:
        chosen_car.dyn.gamma = -chosen_car.action[0]
    else:
        chosen_car.dyn.gamma = chosen_car.action[0]

    # if chosen_car.velocity < 0:
    #     chosen_car.velocity = 0
    #
    # elif chosen_car.velocity > 0.01:
    #     chosen_car.velocity = 0.01
    #


def qnn_training3(mock_anim_x, mock_cars, car_no, auto_cars, sec_turn_flag, human_anim_x,human_cars):
    """
    qnn training
    :return: None but directly set the velocity of auto_cars
    """

    chosen_car = mock_cars[car_no]
    control_car = auto_cars[0]
    last_velocity = chosen_car.velocity
    if chosen_car.aggressive == 1 and car_no != 0:
        if sec_turn_flag == 1 and mock_anim_x[car_no-1][1] > human_anim_x[0][1] > mock_anim_x[car_no][1]:
            s_distance = human_anim_x[0][1] - mock_anim_x[car_no][1]
            s_distance2 = None
            #print(s_distance)
            s_vel = 0
            s_vel2 = None
        else:
            s_distance = (mock_anim_x[car_no-1][1] - mock_anim_x[car_no][1])
            s_distance2 = None
            s_vel = mock_cars[car_no-1].velocity - mock_cars[car_no].velocity
            s_vel2 = None
        #s_vel = s_vel2 = 0
        if s_distance is not None:
            s_distance /= 1.2
        if s_distance2 is not None:
            s_distance2 /= 1.2
        # if s_distance>0:
        if s_distance is None:
            chosen_car.state = None
        else:
            chosen_car.state = [s_distance, s_vel]

        if s_distance2 is None:
            chosen_car.state2 = None
        else:
            chosen_car.state2 = [s_distance2, s_vel2]

        a = control_car.choose_action(chosen_car.state, chosen_car.state2)

        chosen_car.action = np.clip(np.random.normal(a, chosen_car.var), -2, 2)  # add randomness to action selection for exploration

        temp_velocity = chosen_car.velocity + chosen_car.action[0]

        chosen_car.velocity = temp_velocity

        if chosen_car.velocity < -0.001:
            chosen_car.velocity = -0.001

        elif temp_velocity > chosen_car.setvelocity:
            chosen_car.velocity = chosen_car.setvelocity

        chosen_car.acceleration = chosen_car.velocity - last_velocity
    else:
        if car_no == 0 and sec_turn_flag == 1:
            if human_anim_x[0][1] > mock_anim_x[car_no][1]:
                s_distance = human_anim_x[0][1] - mock_anim_x[car_no][1]
                s_distance2 = None
                #print(s_distance)
                s_vel = human_cars[0].velocity - chosen_car.velocity
                s_vel2 = None
            # if s_distance>0:
                if s_distance is None:
                    chosen_car.state = None
                else:
                    chosen_car.state = [s_distance, s_vel]

                if s_distance2 is None:
                    chosen_car.state2 = None
                else:
                    chosen_car.state2 = [s_distance2, s_vel2]

                a = control_car.choose_action(chosen_car.state, chosen_car.state2)

                chosen_car.action = np.clip(np.random.normal(a, chosen_car.var), -2,
                                            2)  # add randomness to action selection for exploration

                temp_velocity = chosen_car.velocity + chosen_car.action[0]


                chosen_car.velocity = temp_velocity

                if chosen_car.velocity < -0.001:
                    chosen_car.velocity = -0.001

                elif chosen_car.velocity > chosen_car.setvelocity:
                    chosen_car.velocity = chosen_car.setvelocity

            chosen_car.acceleration = chosen_car.velocity - last_velocity


def state_rec(mock_anim_x, mock_cars):
    temp_list = []

    for index, location in enumerate(mock_anim_x):
        temp_list += [location[1], mock_cars[index].velocity]

    return temp_list


def tar_rec(filepointer, mock_anim_x):
    target = 0.05
    temp_list = []

    if mock_anim_x[0][1] > 0.05:
        for index, location in enumerate(mock_anim_x):
            if location[1] < 0.05:
                break
        temp_list = [mock_anim_x[index - 1][1] - target, mock_anim_x[index][1] - target]
    else:
        temp_list = [1, mock_anim_x[0][1] - target]


    filepointer.write(str(temp_list).replace('[','').replace(']','') + '\n')