from keras.models import load_model
from keras.backend import gradients
import numpy as np
import tensorflow as tf

remain_list = [ 'Local_Y','Local_X', 'v_Vel', 'v_Acc', 'v_length', 'v_Width', 'v_Class', 'Space_Headway', 'Time_Headway']
max_length = 28
temporal_horizon = 3
MIXTURE = 10


class controller:
    def __init__(self,):
        max_length = 28
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph('./Car_following/all_model.ckpt.meta')
        self.graph = tf.get_default_graph()
        self.all_inputs = self.graph.get_tensor_by_name('all_inputs:0')
        self.decoder_inputs = self.graph.get_tensor_by_name('decoder_inputs:0')
        self.see_all = self.graph.get_tensor_by_name('see_all:0')
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./Car_following'))
        # a = self.sess.run(self.see_all, feed_dict={self.all_inputs: input_seq, self.decoder_inputs: target_seq})
        # decode_me(a, max_length)
        # print(a)

    def decode_me(self,prediction, max_length):
        output_list = []
        for i in range(max_length + 2):
            sum = 0
            for j in range(MIXTURE):
                sum += prediction[0][i, j] * prediction[1][i, j]
            output_list.append(sum)
        return output_list

    def car_following(self, historical_mock):
        encoder_input_data = np.zeros((1, temporal_horizon * max_length, len(remain_list)))
        car_no_min = historical_mock[2][0][0].car_no
        car_no_max = historical_mock[0][0][-1].car_no
        for i in range(temporal_horizon):
            count = 0
            for no, j in enumerate(historical_mock[i][0]):
                if j.car_no <= car_no_max and j.car_no >= car_no_min:
                    local_y = (historical_mock[i][1][no][1] +1) /2
                    local_x = historical_mock[i][1][no][0] * 0
                    v_vel = historical_mock[i][0][no].velocity * 100
                    v_acc = historical_mock[i][0][no].acceleration * 100
                    v_length = historical_mock[i][0][no].v_length
                    v_width = historical_mock[i][0][no].v_width
                    v_class = historical_mock[i][0][no].v_class
                    if no != 0:
                        v_space_headway = historical_mock[i][0][no-1].carnext
                    else:
                        v_space_headway = 0
                    v_time_headway = v_space_headway/v_vel
                    temp_list = [local_y, local_x, v_vel, v_acc, v_length, v_width, v_class, v_space_headway, v_time_headway]
                    for m in range(len(temp_list)):
                        encoder_input_data[0,i*max_length + count, m] = temp_list[m]
                    count += 1

        target_input_data = np.zeros((1, max_length + 1, 1))
        target_input_data[0,0,0] = 1
        for i in range(max_length):
            target_input_data[0,i+1,0] = encoder_input_data[0,2*max_length + i, 0]

        prediction = self.sess.run(self.see_all, feed_dict={self.all_inputs: encoder_input_data, self.decoder_inputs: target_input_data})

        return self.decode_me(prediction,max_length)




    def limited_go(self, state, location, flag):
        temp_list = [state[2*i] for i in range(int(len(state)/2))]
        temp_list += [-1 for i in range(10-len(temp_list))] + [location]
        # print(temp_list)
        # print(state)
        input_dim = 11
        safty_distance = 0.25
        for no, loc in enumerate(temp_list):
            goal = -1
            if loc + safty_distance <= temp_list[input_dim-1]:
                if no == 0:
                    goal = temp_list[input_dim - 1]
                    break
                if temp_list[no-1] - safty_distance > temp_list[input_dim - 1]:
                    goal = temp_list[input_dim - 1]
                    break
            if loc <= temp_list[input_dim - 1]:
                if no == 0:
                    goal = safty_distance + loc
                    if flag == 1:
                        goal = 0.8
                else:
                    if no > 1:
                        if temp_list[no] - temp_list[no + 1] > 2*safty_distance:
                            goal = temp_list[no] -safty_distance
                        else:
                            goal =temp_list[no + 1] - safty_distance
                break
        #return ((all_grad[-1][0][-1]))
        #print(-location + goal)
        return (goal - location)


    def control_go(self, state, location, flag, historical_mock):
        state_2 = None
        if len(historical_mock) == 3:
            state_2 = self.car_following(historical_mock)

        temp_list = [state[2*i] for i in range(int(len(state)/2))]
        temp_list += [-1 for i in range(10-len(temp_list))] + [location]
        # print(temp_list)
        # print(state)
        safty_distance = 0.25
        for no, loc in enumerate(temp_list):
            goal = -1
            if loc + safty_distance <= temp_list[-1]:
                if no == 0:
                    goal = temp_list[ - 1]
                    break
                if temp_list[no-1] - safty_distance > temp_list[- 1]:
                    goal = temp_list[- 1]
                    break
            if loc <= temp_list[- 1]:
                if no == 0:
                    goal = safty_distance + loc
                    if flag == 1:
                        goal = 0.8
                else:
                    if temp_list[no - 1] - loc > 2*safty_distance:
                        goal_temp = temp_list[no - 1] - safty_distance
                        goal_temp2 = loc + safty_distance

                        if flag == 1:
                            goal = goal_temp
                        else:
                            if goal_temp - temp_list[-1] > temp_list[-1] - goal_temp2:
                                goal = goal_temp2
                            else:
                                goal = goal_temp
                    elif no > 1:
                        if temp_list[no - 2] - temp_list[no - 1] > 2*safty_distance:
                            goal = (temp_list[no - 1] + safty_distance)
                        elif temp_list[no] - temp_list[no + 1] > 2*safty_distance:
                            goal = temp_list[no] -safty_distance
                        else:
                            goal =temp_list[no + 1] - safty_distance
                break

        result1 =  (goal - location)

        if state_2 is not None:
            temp_list = state_2
        else:
            return result1
        for i in range(len(temp_list)):
            temp_list[i] = temp_list[i]* 2 - 1
        temp_list += [-1 for i in range(10-len(temp_list))] + [location]
        # print(temp_list)
        # print(state)
        safty_distance = 0.25
        for no, loc in enumerate(temp_list):
            goal = -1
            if loc + safty_distance <= temp_list[-1]:
                if no == 0:
                    goal = temp_list[ - 1]
                    break
                if temp_list[no-1] - safty_distance > temp_list[- 1]:
                    goal = temp_list[- 1]
                    break
            if loc <= temp_list[- 1]:
                if no == 0:
                    goal = safty_distance + loc
                    if flag == 1:
                        goal = 0.8
                else:
                    if temp_list[no - 1] - loc > 2*safty_distance:
                        goal_temp = temp_list[no - 1] - safty_distance
                        goal_temp2 = loc + safty_distance

                        if flag == 1:
                            goal = goal_temp
                        else:
                            if goal_temp - temp_list[-1] > temp_list[-1] - goal_temp2:
                                goal = goal_temp2
                            else:
                                goal = goal_temp
                    elif no > 1:
                        if temp_list[no - 2] - temp_list[no - 1] > 2*safty_distance:
                            goal = (temp_list[no - 1] + safty_distance)
                        elif temp_list[no] - temp_list[no + 1] > 2*safty_distance:
                            goal = temp_list[no] -safty_distance
                        else:
                            goal =temp_list[no + 1] - safty_distance
                break

        result2 =  (goal - location)

        return result1 + result2*0.1