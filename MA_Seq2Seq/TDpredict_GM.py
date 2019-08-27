import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm
import os
from tqdm import tqdm
import TDmodel_GM
path = './toy'
remain_list = [ 'Local_Y','Local_X', 'v_Vel', 'v_Acc', 'v_length', 'v_Width', 'v_Class', 'Space_Headway', 'Time_Headway']

temporal_horizon = 3
far_togo = 1  # 预测多远
key_pred = 1  # 预测一个参数
stride = 5  # 步长


def data_prepare():
    all2D_df = []
    max_length = 0
    length_list = []
    for file in tqdm(os.listdir(path)):
        file_path = os.path.join(path, file)
        if file_path != path + '/.DS_Store':
            v_df = pd.read_csv(file_path, index_col=None, sep=',')
            local_length = 0
            for i,j in v_df.groupby('Global_Time'):
                local_length = len(j)
                if max_length<len(j):
                    max_length = len(j)
                break
            least_length = temporal_horizon * stride * local_length + (far_togo - 1) * stride * local_length + local_length
            if len(v_df) >= least_length:
                length_list.append([local_length,len(v_df)])
                all2D_df.append(v_df)
        else:
            continue
    all2D_df = pd.concat(all2D_df, axis=0, ignore_index=True).reset_index(drop=True)[remain_list]
    all2D_df = all2D_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return  length_list, max_length, all2D_df


def train_data(length_list, max_length, all2D_df, latent_dim):

    all_length = 0
    for i in length_list:
        all_length += int((i[1]-(stride * (temporal_horizon + far_togo - 1) * i[0] + i[0]))/i[0]) + 1

    encoder_input_data = np.zeros(
        (all_length, temporal_horizon * max_length, len(remain_list)),
        # 长度+1是为了固定预测时解码器的第一个输入
        dtype='float32')  # 数据量 编码时最长的一句话长度 编码时每个字母的长度
    decoder_input_data = np.zeros(
        (all_length, max_length + 1, key_pred),
        dtype='float32')

    decoder_target_data = []
    decoder_target_data1 = np.zeros(
        (all_length,  max_length + 1, key_pred),
        dtype='float32')


    cnt = 0
    cnt2 = 0
    for i in tqdm(range(len(length_list))):
        for no in range(int((length_list[i][1]-(stride * (temporal_horizon + far_togo - 1) * length_list[i][0]
                                                + length_list[i][0]))/length_list[i][0]) + 1):
            for no2 in range(temporal_horizon):
                for no3 in range(length_list[i][0]):
                    for no4 in range(len(remain_list)):
                        encoder_input_data[no + cnt2, no2 * max_length + no3, no4] \
                            = all2D_df.iat[no * length_list[i][0] + cnt + no2 * stride * length_list[i][0] + no3, no4]

            decoder_input_data[no + cnt2, 0, 0] = 1
            for no2 in range(length_list[i][0]):
                for no3 in range(key_pred):
                    decoder_input_data[no + cnt2, no2 + 1, no3] = \
                        all2D_df.iat[no * length_list[i][0] + cnt + no2 + (temporal_horizon)* stride * length_list[i][0], no3]
                decoder_target_data1[no + cnt2, no2, 0] = \
                    all2D_df.iat[no* length_list[i][0] + cnt + no2 + (temporal_horizon)* stride * length_list[i][0], 0]

        cnt += length_list[i][1]
        cnt2 += int((length_list[i][1]-(stride * (temporal_horizon + far_togo - 1) * length_list[i][0]
                                                + length_list[i][0]))/length_list[i][0]) + 1

    for i in tqdm(range(max_length + 1)):
        temp_data_target = np.zeros(
            (all_length, 1),
            dtype='float32')
        for j in (range(all_length)):
            temp_data_target[j, 0] = decoder_target_data1[j, i, 0]
        decoder_target_data.append(temp_data_target)
    decoder_target_data.append(decoder_target_data[0])
    return encoder_input_data, decoder_input_data, decoder_target_data


if __name__ == '__main__':
    latent_dim = 128
    length_list, max_length, all2D_df = data_prepare()
    max_length = 28
    encoder_input_data, decoder_input_data, decoder_target_data = train_data(length_list, max_length, all2D_df, latent_dim)

    TDmodel_GM.lstm_train(num_encoder_tokens=len(remain_list), latent_dim=latent_dim, batch_size=128, epochs=20, encoder_input_data=encoder_input_data,
                         decoder_input_data=decoder_input_data, decoder_target_data=decoder_target_data,
                       temporal_horizon=temporal_horizon, max_length=max_length)

    TDmodel_GM.lstm_predict(input_seq= encoder_input_data[16:17], target_seq = decoder_input_data[16:17])
    print('D')
    print(decoder_input_data[16:17])