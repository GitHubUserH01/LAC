from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Lambda, multiply, add, merge, concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np
import keras.backend as K
import math

MIXTURE = 10


def lstm_train(num_encoder_tokens, latent_dim, batch_size, epochs, encoder_input_data, decoder_input_data,
               decoder_target_data, temporal_horizon, max_length):
    # Define an input sequence and process it.
    #all_inputs = Input(shape=(None, num_encoder_tokens))
    all_inputs = tf.placeholder(tf.float32, (None,None, num_encoder_tokens), name='all_inputs')
    encoder_split = Lambda(lambda x: tf.split(x, num_or_size_splits= max_length * temporal_horizon, axis=-2))(all_inputs)

    state_list = []
    output_list = []

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.

    mid_layer1 = Dense(latent_dim, activation='relu')
    mid_layer2 = Dense(latent_dim, activation='relu')
    mid_layer3 = Dense(latent_dim, activation='relu')
    mid_layer4 = Dense(latent_dim, kernel_initializer='zeros',  trainable=False)
    encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)

    Zero_state = mid_layer4(encoder_split[0])
    _, Zero_state, Zero_output = encoder_lstm(Zero_state)

    state_list.append(Zero_state)

    for i in range(temporal_horizon):
        for j in range(max_length):
            if i == 0:
                last_time_state = Zero_state
                last_time_output = Zero_output
            else:
                last_time_state = state_list[(i - 1)*max_length + j + 1]
                last_time_output = output_list[(i - 1)*max_length + j]

            if j == 0:
                last_spat_state = Zero_state
                last_spat_output = Zero_output
            else:
                last_spat_state = state_list[(i)*max_length + j-1 + 1]
                last_spat_output = output_list[(i)*max_length + j-1]

            all_state = mid_layer2(concatenate([last_time_state, last_spat_state], axis=-1))
            all_output = mid_layer3(concatenate([last_time_output, last_spat_output], axis=-1))

            if i == 0 and j == 0:
                encoder_outputs, state_h, state_c = \
                    encoder_lstm(mid_layer1(encoder_split[i*max_length + j]), initial_state=[all_state,all_output])
            else:
                encoder_outputs, state_h, state_c = \
                    encoder_lstm(mid_layer1(encoder_split[i * max_length + j]), initial_state=[all_state, all_output])

            state_list.append(state_h)
            output_list.append(state_c)

    decoder_inputs = tf.placeholder(tf.float32, (None, None,1), name='decoder_inputs')
    decoder_split = Lambda(lambda x: tf.split(x, num_or_size_splits=max_length + 1, axis=-2))(decoder_inputs)

    attention_probs = Dense(temporal_horizon**2, activation='softmax', name='attention_vec')
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_lstm_sp = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_dense = Dense(1, activation='sigmoid')
    mdn_decoder_dense = Dense(MIXTURE*3, activation='sigmoid')
    decoder_output_list = []
    mkn_output_list = []
    de_state_h = None
    de_state_c = None

    for i in range(max_length + 1):
        temp_list = []
        add_list = []
        if i == 0:
            decoder_outputs, de_state_h, de_state_c = decoder_lstm(decoder_split[i],
                                                 initial_state=[state_list[-1], output_list[-1]])
        else:
            decoder_outputs, de_state_h, de_state_c = decoder_lstm(decoder_split[i],
                                                                   initial_state=[de_state_h,
                                                                                  de_state_c])

        for m in range(temporal_horizon):
            if i == 0:
                temp_list.append(state_list[m*max_length + i + 1])
                temp_list.append(state_list[m*max_length + i + 1])
                temp_list.append(state_list[m*max_length + i + 1])
            if i == 1:
                temp_list.append(state_list[m * max_length + i])
                temp_list.append(state_list[m * max_length + i])
                temp_list.append(state_list[m * max_length + i + 1])
            if i!=0 and i!=1 and i!=max_length:
                temp_list.append(state_list[m*max_length + i-1])
                temp_list.append(state_list[m*max_length + i])
                temp_list.append(state_list[m*max_length + i + 1])
            if i == max_length:
                temp_list.append(state_list[m*max_length + i-1])
                temp_list.append(state_list[m*max_length + i])
                temp_list.append(state_list[m*max_length + i])

        attention_input = concatenate(temp_list, axis=-1)

        if i == 4:
            look_attetion = attention_probs(attention_input)

        attention_split = Lambda(lambda x: tf.split(x, num_or_size_splits=temporal_horizon**2, axis=-1))(attention_probs(attention_input))


        # for m in range(temporal_horizon**2):
        #     add_list.append(multiply([attention_split[m], temp_list[m]]))

        for m in range(temporal_horizon**2):
            add_list.append(multiply([attention_split[m], temp_list[m]]))

        add_layer = concatenate([add(add_list), de_state_h], axis=-1)

        decoder_finaloutput = decoder_dense(add_layer)
        decoder_output_list.append(decoder_finaloutput)


        mkn_output_list.append(mdn_decoder_dense(add_layer))


    encoder_lstm_sp = LSTM(latent_dim, return_sequences=True, return_state=True)
    mid_layer_sp = Dense(latent_dim, activation='relu')
    decoder_dense_sp = Dense(1, activation='sigmoid')
    mkn_dense_sp = Dense(MIXTURE*3, activation='sigmoid')

    for i in range(temporal_horizon):
        if i == 0:
            _, state_h, state_c = encoder_lstm_sp(mid_layer_sp(encoder_split[i*max_length]))
        else:
            _, state_h, state_c = encoder_lstm_sp(mid_layer_sp(encoder_split[i * max_length]), initial_state=[state_h,state_c])

    decoder_outputs, de_state_h, de_state_c = decoder_lstm_sp(decoder_split[0],
                                                           initial_state=[state_h, state_c])

    decoder_output_list.append(decoder_dense_sp(de_state_h))
    mkn_output_list.append(mkn_dense_sp(de_state_h))

    weight_layer = Dense(MIXTURE, activation='softmax', name='attention')
    all_parameters_list = []

    for i in range(len(mkn_output_list)):
        all_parameters = mkn_output_list[i]
        weight, mu_out, sigma = tf.split(all_parameters, 3, -1)
        weight_out = weight_layer(weight)
        sigma_out = tf.exp(sigma, name='sigma')
        all_parameters_list.append([weight_out, mu_out, sigma_out])

    see_all = tf.concat(all_parameters_list, axis=-2, name='see_all')
    all_outputs = tf.placeholder(tf.float32, (None, None, 1), name='all_outputs')

    def mkn_loss(all_parameters_list, all_outputs):
        loss_final = 0
        for i in range(len(all_parameters_list)):
            all_parameters = all_parameters_list[i]
            weight_out = all_parameters[0]
            mu_out = all_parameters[1]
            sigma_out = all_parameters[2]
            factor = 1 / math.sqrt(2 * math.pi)
            epsilon = 1e-5
            tmp = - tf.square((all_outputs[i] - mu_out)) / (2 * tf.square(tf.maximum(sigma_out, epsilon)))
            y_normal = factor * tf.exp(tmp) / tf.maximum(sigma_out, epsilon)
            loss = tf.reduce_sum(tf.multiply(y_normal, weight_out), keepdims=True)
            loss = -tf.log(tf.maximum(loss, epsilon))
            loss_final += tf.reduce_mean(loss)
        return loss_final

    loss = mkn_loss(all_parameters_list, all_outputs)

    train_step = tf.train.AdamOptimizer(learning_rate=0.02).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(30):
            _, lossval = sess.run([train_step, loss],
                     feed_dict={all_inputs: encoder_input_data, decoder_inputs: decoder_input_data,
                                all_outputs: decoder_target_data})
            print(lossval)

        saver = tf.train.Saver()
        saver.save(sess, './all_model.ckpt')

    # all_model = Motion_prediction([all_inputs, decoder_inputs], [look_attetion] + decoder_output_list)
    # all_model.save('All_s2s.h5')


def decode_me(all_output, max_length):
    output_list = []
    for i in range(max_length+2):
        sum = 0
        for j in range(MIXTURE):
            sum += all_output[0][i,j] * all_output[1][i,j]
        output_list.append(sum)

    print(output_list)


def lstm_predict(input_seq, target_seq):
    max_length = 28
    temporal_horizon = 3
    # all_model  = load_model('All_s2s.h5', custom_objects={'tf': tf, 'max_length': max_length, 'temporal_horizon': temporal_horizon})
    # # Encode the input as state vectors.
    # print(all_model.predict([input_seq,target_seq]))
    #
    # target_input = np.ones(
    #     (1, max_length + 1, 1),
    #     dtype='float32')
    # for i in range(max_length):
    #     allout = all_model.predict([input_seq, target_input])
    #     target_input[0,i+1,0] = allout[i+1][0,0]
    #
    # target_input[0,1,0] = allout[max_length+2][0,0]
    # attention = allout[0]
    #
    # print(target_input)
    # print(attention)

    sess = tf.Session()
    saver = tf.train.import_meta_graph('./all_model.ckpt.meta')
    graph = tf.get_default_graph()
    all_inputs = graph.get_tensor_by_name('all_inputs:0')
    decoder_inputs = graph.get_tensor_by_name('decoder_inputs:0')
    see_all = graph.get_tensor_by_name('see_all:0')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    a = sess.run(see_all, feed_dict={all_inputs: input_seq, decoder_inputs:target_seq})
    decode_me(a,max_length)
    print(a)

