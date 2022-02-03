# Set directory of main map (without final /)
directory = 'C:/user/neural network project'

# 1. Imports
for h in range(1):
    import numpy as np
    import pandas as pd
    import matplotlib
    # matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    import os
    import random
    import tensorflow as tf
    from keras.layers import LSTM
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.models import Sequential
    import keras
    from keras import regularizers
    from keras.models import save_model
    from keras.models import load_model
    from keras import backend as K
    # session_conf = tf.config(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    # K.set_session(sess)
    from matplotlib.pyplot import figure
    from sklearn.metrics import mean_squared_error
    import pylab as P
    from scipy.signal import savgol_filter
    def lighten_color(color, amount=0.5):
        """
        Lightens the given color by multiplying (1-luminosity) by the given amount.
        Input can be matplotlib color string, hex string, or RGB tuple.

        Examples:
        >> lighten_color('g', 0.3)
        >> lighten_color('#F034A3', 0.6)
        >> lighten_color((.3,.55,.1), 0.5)
        """
        import matplotlib.colors as mc
        import colorsys
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    from scipy.stats import ttest_ind as tti
    from scipy.stats import ttest_rel as ttr
    # import loaddata

# 2. Setting seeds
for h in range(1):
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
    tf.compat.v2.random.set_seed(1)
    random.seed(1)
    os.environ['PYTHONHASHSEED']=str(1)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    np.random.seed(1)
        

# 3. Project configuration options.
# Relevant for final results:
    maxTraining_time=32
    minTraining_time=0
    logscaled_training = True
    resmoothing_enabled = True
# Irrelevant for final results:
    Srxn1_as_input = False
    derivative_training = False
    # Dissected training
    multiplication_factor = 100
    section_length = 100
    # ODE training set
    training_set = 3

# 4. time and dose vector generation
for h in range(1):
    # time vector scaler
    scalarmaxtime = 96
    scalarmintime = 0
    scalerlist = []
    for i in range(2000):
        timepoint = ((scalarmaxtime-scalarmintime)/2000)*i + scalarmintime
        scalerlist.append(timepoint)
    scalerlist = np.expand_dims(scalerlist, axis=1)
    # Set scaler to normalize to 0-1
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(scalerlist)
    # scalerlist = scaler.transform(scalerlist)
    # scalerlist

    # Prepare time vector
    max_time = maxTraining_time
    min_time = minTraining_time
    list = []
    # generate time vector in which the value goes from 1 to 32 over 2000 data points
    for i in range(2000):
        timepoint = ((max_time-min_time)/2000)*i + min_time
        list.append(timepoint)
    list = np.expand_dims(list, axis = 1)
    listt = scaler.transform(list)
    listt = np.array(listt)

    # dose vector scaler
    scaling_conclist = [0.1,40]
    for i in range(2):
        scaling_conclist[i] = np.log10(scaling_conclist[i])
    scaling_conclist = np.expand_dims(scaling_conclist, axis=1)
    scaler.fit(scaling_conclist)
    scaling_conclist

# 5. Loading in data
for h in range(1):
    # datascalerlistNrf2 = np.array([np.log10(0.00005), np.log10(6)])
    # datascalerlistNrf2 = np.array([0, 6])
    # datascalerlistNrf2 = np.log10(datascalerlistNrf2)
    # datascalerlistSrxn1 = np.array([np.log10(0.001), np.log10(50)])
    if logscaled_training == False:
        datascalerlistSrxn1 = np.array([0, 100])
        # datascalerlistSrxn1 = np.array([0, 100])
        datascalerlistNrf2 = np.array([0,6])

    if logscaled_training == True:
        datascalerlistSrxn1 = np.array([0, 2])
        datascalerlistNrf2 = np.array([0, 1])

    # datascalerlistSrxn1 = np.log10(datascalerlistSrxn1)
    # datascalerlistSrxn1 = datascalerlistSrxn1.reshape(2, 1)
    # datascalerlistNrf2 = datascalerlistNrf2.reshape(2, 1)


    def dataprocessing(dataset):
        mean = []
        for i in range(len(dataset)):
            m = np.sum(dataset[i, :]) / np.shape(dataset[1])
            mean.append(m)
        mean = np.array(mean)
        mean = mean.reshape(len(mean), 1)
        mean = scaler.transform(mean)

        Urange = []
        Lrange = []
        for i in range(len(dataset)):
            upper = np.amax(dataset[i, :])
            Urange.append(upper)
            lower = np.amin(dataset[i, :])
            Lrange.append(lower)
        # Urange = np.expand_dims(Urange, axis=1)
        # Urange = scaler.transform(Urange)
        # Lrange = np.expand_dims(Lrange, axis=1)
        # Lrange = scaler.transform(Lrange)

        Rmean = []
        for i in range(len(Urange)):
            m = (Urange[i] + Lrange[i]) / 2
            Rmean.append(m)
        Rmean = np.array(Rmean)
        return mean, Urange, Lrange, Rmean

    def dataprocessingtrans(dataset, dtraining = False):
        mean = []
        for i in range(len(dataset)):
            m = np.sum(dataset[i, :]) / np.shape(dataset[1])
            mean.append(m)
        mean = np.array(mean)
        # for i in range(len(mean)):
        #     if mean[i] > 0:
        #         mean[i] = np.log10(mean[i])
        # mean[2:2000] = np.log10(mean[2:2000])
        mean = mean.reshape(len(mean), 1)
        mean = scaler.transform(mean)

        Urange = []
        Lrange = []
        for i in range(len(dataset)):
            upper = np.amax(dataset[i, :])
            Urange.append(upper)
            lower = np.amin(dataset[i, :])
            Lrange.append(lower)
        # Urange[2:2000] = np.log10(Urange[2:2000])
        # Lrange[2:2000] = np.log10(Lrange[2:2000])
        # for i in range(len(Urange)):
        #     if Urange[i] > 0:
        #         Urange[i] = np.log10(Urange[i])
        # for i in range(len(Lrange)):
        #     if Lrange[i] > 0:
        #         Lrange[i] = np.log10(Lrange[i])
        #
        Rmean = []
        for i in range(len(Urange)):
            rm = (Urange[i] + Lrange[i]) / 2
            Rmean.append(rm)
        if logscaled_training == True:
            for i in range(len(Rmean)):
                # Rmean[i] = np.around(Rmean[i],2)
                Rmean[i] = Rmean[i] + 1
                Rmean[i] = np.log10(Rmean[i])
                # if Rmean[i] <= 0.001:
                #     Rmean[i] = -3
                # if Rmean[i] > 0.001:
                #     Rmean[i] = np.log10(Rmean[i])

        Rmean = np.expand_dims(Rmean, axis=1)
        # Rmean = scaler.transform(Rmean)
        Urange = np.expand_dims(Urange, axis=1)
        # Urange = scaler.transform(Urange)
        Lrange = np.expand_dims(Lrange, axis=1)
        # Lrange = scaler.transform(Lrange)


        return mean, Urange, Lrange, Rmean

    #ODE Data
    for h in range(1):
        # Interpolation
        from scipy import interpolate
        interpolation_length = 2000
        hours = 49
        interpolation_length = np.linspace(1, hours, interpolation_length)
        time = []
        for j in range(hours):
            j = j + 1
            time.append(j)

        # Combined training
        for h in range(1):
            Nrf2_training_ES1_1 = np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 1/Nrf2_training1.txt', sep=" "))[:,0]
            Nrf2_training_ES1_1 = interpolate.interp1d(time, Nrf2_training_ES1_1, kind='cubic')(interpolation_length)
            Nrf2_training_ES1_2 = np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 1/Nrf2_training2.txt', sep=" "))[:,0]
            Nrf2_training_ES1_2 = interpolate.interp1d(time, Nrf2_training_ES1_2, kind='cubic')(interpolation_length)
            Nrf2_training_ES1_3 = np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 1/Nrf2_training3.txt', sep=" "))[:,0]
            Nrf2_training_ES1_3 = interpolate.interp1d(time, Nrf2_training_ES1_3, kind='cubic')(interpolation_length)
            Nrf2_training_ES1_4 = np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 1/Nrf2_training4.txt', sep=" "))[:,0]
            Nrf2_training_ES1_4 = interpolate.interp1d(time, Nrf2_training_ES1_4, kind='cubic')(interpolation_length)
            Nrf2_test_ES1_1 = np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 1/Nrf2_test1.txt', sep=" "))[:,0]
            Nrf2_test_ES1_1 = interpolate.interp1d(time, Nrf2_test_ES1_1, kind='cubic')(interpolation_length)
            Nrf2_test_ES1_2 = np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 1/Nrf2_test2.txt', sep=" "))[:,0]
            Nrf2_test_ES1_2 = interpolate.interp1d(time, Nrf2_test_ES1_2, kind='cubic')(interpolation_length)

            Srxn1_training_ES1_1 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 1/Srxn1_training1.txt', sep=" "))[:,0])
            Srxn1_training_ES1_1 = interpolate.interp1d(time, Srxn1_training_ES1_1, kind='cubic')(interpolation_length)
            Srxn1_training_ES1_2 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 1/Srxn1_training2.txt', sep=" "))[:,0])
            Srxn1_training_ES1_2 = interpolate.interp1d(time, Srxn1_training_ES1_2, kind='cubic')(interpolation_length)
            Srxn1_training_ES1_3 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 1/Srxn1_training3.txt', sep=" "))[:,0])
            Srxn1_training_ES1_3 = interpolate.interp1d(time, Srxn1_training_ES1_3, kind='cubic')(interpolation_length)
            Srxn1_training_ES1_4 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 1/Srxn1_training4.txt', sep=" "))[:,0])
            Srxn1_training_ES1_4 = interpolate.interp1d(time, Srxn1_training_ES1_4, kind='cubic')(interpolation_length)
            Srxn1_test_ES1_1 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 1/Srxn1_test1.txt', sep=" "))[:,0])
            Srxn1_test_ES1_1 = interpolate.interp1d(time, Srxn1_test_ES1_1, kind='cubic')(interpolation_length)
            Srxn1_test_ES1_2 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 1/Srxn1_test2.txt', sep=" "))[:,0])
            Srxn1_test_ES1_2 = interpolate.interp1d(time, Srxn1_test_ES1_2, kind='cubic')(interpolation_length)

            Nrf2_training_ES2_1 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 2/Nrf2_train1.txt', sep=" "))[:,0])
            Nrf2_training_ES2_1 = interpolate.interp1d(time, Nrf2_training_ES2_1, kind='cubic')(interpolation_length)
            Nrf2_training_ES2_2 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 2/Nrf2_train2.txt', sep=" "))[:,0])
            Nrf2_training_ES2_2 = interpolate.interp1d(time, Nrf2_training_ES2_2, kind='cubic')(interpolation_length)
            Nrf2_training_ES2_3 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 2/Nrf2_train3.txt', sep=" "))[:,0])
            Nrf2_training_ES2_3 = interpolate.interp1d(time, Nrf2_training_ES2_3, kind='cubic')(interpolation_length)
            Nrf2_training_ES2_4 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 2/Nrf2_train4.txt', sep=" "))[:,0])
            Nrf2_training_ES2_4 = interpolate.interp1d(time, Nrf2_training_ES2_4, kind='cubic')(interpolation_length)
            Nrf2_test_ES2_1 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 2/Nrf2_test1.txt', sep=" "))[:,0])
            Nrf2_test_ES2_1 = interpolate.interp1d(time, Nrf2_test_ES2_1, kind='cubic')(interpolation_length)
            Nrf2_test_ES2_2 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 2/Nrf2_test2.txt', sep=" "))[:,0])
            Nrf2_test_ES2_2 = interpolate.interp1d(time, Nrf2_test_ES2_2, kind='cubic')(interpolation_length)

            Srxn1_training_ES2_1 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 2/Srxn1_train1.txt', sep=" "))[:,0])
            Srxn1_training_ES2_1 = interpolate.interp1d(time, Srxn1_training_ES2_1, kind='cubic')(interpolation_length)
            Srxn1_training_ES2_2 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 2/Srxn1_train2.txt', sep=" "))[:,0])
            Srxn1_training_ES2_2 = interpolate.interp1d(time, Srxn1_training_ES2_2, kind='cubic')(interpolation_length)
            Srxn1_training_ES2_3 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 2/Srxn1_train3.txt', sep=" "))[:,0])
            Srxn1_training_ES2_3 = interpolate.interp1d(time, Srxn1_training_ES2_3, kind='cubic')(interpolation_length)
            Srxn1_training_ES2_4 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 2/Srxn1_train4.txt', sep=" "))[:,0])
            Srxn1_training_ES2_4 = interpolate.interp1d(time, Srxn1_training_ES2_4, kind='cubic')(interpolation_length)
            Srxn1_test_ES2_1 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 2/Srxn1_test1.txt', sep=" "))[:,0])
            Srxn1_test_ES2_1 = interpolate.interp1d(time, Srxn1_test_ES2_1, kind='cubic')(interpolation_length)
            Srxn1_test_ES2_2 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 2/Srxn1_test2.txt', sep=" "))[:,0])
            Srxn1_test_ES2_2 = interpolate.interp1d(time, Srxn1_test_ES2_2, kind='cubic')(interpolation_length)

            Nrf2_training_ES3_1 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 3/Nrf2_training1.txt', sep=" "))[:,0])
            Nrf2_training_ES3_1 = interpolate.interp1d(time, Nrf2_training_ES3_1, kind='cubic')(interpolation_length)
            Nrf2_training_ES3_2 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 3/Nrf2_training2.txt', sep=" "))[:,0])
            Nrf2_training_ES3_2 = interpolate.interp1d(time, Nrf2_training_ES3_2, kind='cubic')(interpolation_length)
            Nrf2_training_ES3_3 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 3/Nrf2_training3.txt', sep=" "))[:,0])
            Nrf2_training_ES3_3 = interpolate.interp1d(time, Nrf2_training_ES3_3, kind='cubic')(interpolation_length)
            Nrf2_training_ES3_4 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 3/Nrf2_training4.txt', sep=" "))[:,0])
            Nrf2_training_ES3_4 = interpolate.interp1d(time, Nrf2_training_ES3_4, kind='cubic')(interpolation_length)
            Nrf2_test_ES3_1 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 3/Nrf2_test1.txt', sep=" "))[:,0])
            Nrf2_test_ES3_1 = interpolate.interp1d(time, Nrf2_test_ES3_1, kind='cubic')(interpolation_length)
            Nrf2_test_ES3_2 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 3/Nrf2_test2.txt', sep=" "))[:,0])
            Nrf2_test_ES3_2 = interpolate.interp1d(time, Nrf2_test_ES3_2, kind='cubic')(interpolation_length)
            Nrf2_test_ES3_3 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 3/Nrf2_test3.txt', sep=" "))[:,0])
            Nrf2_test_ES3_3 = interpolate.interp1d(time, Nrf2_test_ES3_3, kind='cubic')(interpolation_length)
            Nrf2_test_ES3_4 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 3/Nrf2_test4.txt', sep=" "))[:,0])
            Nrf2_test_ES3_4 = interpolate.interp1d(time, Nrf2_test_ES3_4, kind='cubic')(interpolation_length)

            Srxn1_training_ES3_1 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 3/Srxn1_training1.txt', sep=" "))[:,0])
            Srxn1_training_ES3_1 = interpolate.interp1d(time, Srxn1_training_ES3_1, kind='cubic')(interpolation_length)
            Srxn1_training_ES3_2 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 3/Srxn1_training2.txt', sep=" "))[:,0])
            Srxn1_training_ES3_2 = interpolate.interp1d(time, Srxn1_training_ES3_2, kind='cubic')(interpolation_length)
            Srxn1_training_ES3_3 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 3/Srxn1_training3.txt', sep=" "))[:,0])
            Srxn1_training_ES3_3 = interpolate.interp1d(time, Srxn1_training_ES3_3, kind='cubic')(interpolation_length)
            Srxn1_training_ES3_4 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 3/Srxn1_training4.txt', sep=" "))[:,0])
            Srxn1_training_ES3_4 = interpolate.interp1d(time, Srxn1_training_ES3_4, kind='cubic')(interpolation_length)
            Srxn1_test_ES3_1 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 3/Srxn1_test1.txt', sep=" "))[:,0])
            Srxn1_test_ES3_1 = interpolate.interp1d(time, Srxn1_test_ES3_1, kind='cubic')(interpolation_length)
            Srxn1_test_ES3_2 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 3/Srxn1_test2.txt', sep=" "))[:,0])
            Srxn1_test_ES3_2 = interpolate.interp1d(time, Srxn1_test_ES3_2, kind='cubic')(interpolation_length)
            Srxn1_test_ES3_3 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 3/Srxn1_test3.txt', sep=" "))[:,0])
            Srxn1_test_ES3_3 = interpolate.interp1d(time, Srxn1_test_ES3_3, kind='cubic')(interpolation_length)
            Srxn1_test_ES3_4 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 3/Srxn1_test4.txt', sep=" "))[:,0])
            Srxn1_test_ES3_4 = interpolate.interp1d(time, Srxn1_test_ES3_4, kind='cubic')(interpolation_length)


            Nrf2_training_ES5_1 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 5/Nrf2_training1.txt', sep=" "))[:,0])
            Nrf2_training_ES5_1 = interpolate.interp1d(time, Nrf2_training_ES5_1, kind='cubic')(interpolation_length)
            Nrf2_training_ES5_2 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 5/Nrf2_training2.txt', sep=" "))[:,0])
            Nrf2_training_ES5_2 = interpolate.interp1d(time, Nrf2_training_ES5_2, kind='cubic')(interpolation_length)
            Nrf2_training_ES5_3 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 5/Nrf2_training3.txt', sep=" "))[:,0])
            Nrf2_training_ES5_3 = interpolate.interp1d(time, Nrf2_training_ES5_3, kind='cubic')(interpolation_length)
            Nrf2_training_ES5_4 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 5/Nrf2_training4.txt', sep=" "))[:,0])
            Nrf2_training_ES5_4 = interpolate.interp1d(time, Nrf2_training_ES5_4, kind='cubic')(interpolation_length)
            Nrf2_test_ES5_1 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 5/Nrf2_test1.txt', sep=" "))[:,0])
            Nrf2_test_ES5_1 = interpolate.interp1d(time, Nrf2_test_ES5_1, kind='cubic')(interpolation_length)
            Nrf2_test_ES5_2 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 5/Nrf2_test2.txt', sep=" "))[:,0])
            Nrf2_test_ES5_2 = interpolate.interp1d(time, Nrf2_test_ES5_2, kind='cubic')(interpolation_length)
            Nrf2_test_ES5_3 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 5/Nrf2_test3.txt', sep=" "))[:,0])
            Nrf2_test_ES5_3 = interpolate.interp1d(time, Nrf2_test_ES5_3, kind='cubic')(interpolation_length)
            Nrf2_test_ES5_4 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 5/Nrf2_test4.txt', sep=" "))[:,0])
            Nrf2_test_ES5_4 = interpolate.interp1d(time, Nrf2_test_ES5_4, kind='cubic')(interpolation_length)

            Srxn1_training_ES5_1 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 5/Srxn1_training1.txt', sep=" "))[:,0])
            Srxn1_training_ES5_1 = interpolate.interp1d(time, Srxn1_training_ES5_1, kind='cubic')(interpolation_length)
            Srxn1_training_ES5_2 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 5/Srxn1_training2.txt', sep=" "))[:,0])
            Srxn1_training_ES5_2 = interpolate.interp1d(time, Srxn1_training_ES5_2, kind='cubic')(interpolation_length)
            Srxn1_training_ES5_3 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 5/Srxn1_training3.txt', sep=" "))[:,0])
            Srxn1_training_ES5_3 = interpolate.interp1d(time, Srxn1_training_ES5_3, kind='cubic')(interpolation_length)
            Srxn1_training_ES5_4 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 5/Srxn1_training4.txt', sep=" "))[:,0])
            Srxn1_training_ES5_4 = interpolate.interp1d(time, Srxn1_training_ES5_4, kind='cubic')(interpolation_length)
            Srxn1_test_ES5_1 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 5/Srxn1_test1.txt', sep=" "))[:,0])
            Srxn1_test_ES5_1 = interpolate.interp1d(time, Srxn1_test_ES5_1, kind='cubic')(interpolation_length)
            Srxn1_test_ES5_2 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 5/Srxn1_test2.txt', sep=" "))[:,0])
            Srxn1_test_ES5_2 = interpolate.interp1d(time, Srxn1_test_ES5_2, kind='cubic')(interpolation_length)
            Srxn1_test_ES5_3 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 5/Srxn1_test3.txt', sep=" "))[:,0])
            Srxn1_test_ES5_3 = interpolate.interp1d(time, Srxn1_test_ES5_3, kind='cubic')(interpolation_length)
            Srxn1_test_ES5_4 = np.log10(np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/R - OSR model/Training set 5/Srxn1_test4.txt', sep=" "))[:,0])
            Srxn1_test_ES5_4 = interpolate.interp1d(time, Srxn1_test_ES5_4, kind='cubic')(interpolation_length)

    # SUL DMSO normalized data 36 hours
    for h in range(1):
        Nrf2_035DN = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n1/0.35_tr1.txt'))[
                     :, 1]
        Nrf2_035DN = pd.DataFrame(Nrf2_035DN)
        Nrf2_035DN[1] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n1/0.35_tr2.txt'))[
                        :, 1]
        Nrf2_035DN[2] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n2/0.35_tr1.txt'))[
                        :, 1]
        Nrf2_035DN[3] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n2/0.35_tr2.txt'))[
                        :, 1]
        Nrf2_035DN[4] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n3/0.35_tr1.txt'))[
                        :, 1]
        Nrf2_035DN[5] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n3/0.35_tr2.txt'))[
                        :, 1]
        Nrf2_035DN = np.array(Nrf2_035DN)

        Nrf2_075DN = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n1/0.75_tr1.txt'))[
                     :, 1]
        Nrf2_075DN = pd.DataFrame(Nrf2_075DN)
        Nrf2_075DN[1] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n1/0.75_tr2.txt'))[
                        :, 1]
        Nrf2_075DN[2] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n2/0.75_tr1.txt'))[
                        :, 1]
        Nrf2_075DN[3] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n2/0.75_tr2.txt'))[
                        :, 1]
        Nrf2_075DN[4] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n3/0.75_tr1.txt'))[
                        :, 1]
        Nrf2_075DN[5] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n3/0.75_tr2.txt'))[
                        :, 1]
        Nrf2_075DN = np.array(Nrf2_075DN)

        Nrf2_162DN = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n1/1.62_tr1.txt'))[
                     :, 1]
        Nrf2_162DN = pd.DataFrame(Nrf2_162DN)
        Nrf2_162DN[1] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n1/1.62_tr2.txt'))[
                        :, 1]
        Nrf2_162DN[2] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n2/1.62_tr1.txt'))[
                        :, 1]
        Nrf2_162DN[3] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n2/1.62_tr2.txt'))[
                        :, 1]
        Nrf2_162DN[4] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n3/1.62_tr1.txt'))[
                        :, 1]
        Nrf2_162DN[5] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n3/1.62_tr2.txt'))[
                        :, 1]
        Nrf2_162DN = np.array(Nrf2_162DN)

        Nrf2_350DN = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n1/3.5_tr1.txt'))[
                     :, 1]
        Nrf2_350DN = pd.DataFrame(Nrf2_350DN)
        Nrf2_350DN[1] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n1/3.5_tr2.txt'))[
                        :, 1]
        Nrf2_350DN[2] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n2/3.5_tr1.txt'))[
                        :, 1]
        Nrf2_350DN[3] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n2/3.5_tr2.txt'))[
                        :, 1]
        Nrf2_350DN[4] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n3/3.5_tr1.txt'))[
                        :, 1]
        Nrf2_350DN[5] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n3/3.5_tr2.txt'))[
                        :, 1]
        Nrf2_350DN = np.array(Nrf2_350DN)

        Nrf2_754DN = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n1/7.54_tr1.txt'))[
                     :, 1]
        Nrf2_754DN = pd.DataFrame(Nrf2_754DN)
        Nrf2_754DN[1] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n1/7.54_tr2.txt'))[
                        :, 1]
        Nrf2_754DN[2] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n2/7.54_tr1.txt'))[
                        :, 1]
        Nrf2_754DN[3] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n2/7.54_tr2.txt'))[
                        :, 1]
        Nrf2_754DN[4] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n3/7.54_tr1.txt'))[
                        :, 1]
        Nrf2_754DN[5] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n3/7.54_tr2.txt'))[
                        :, 1]
        Nrf2_754DN = np.array(Nrf2_754DN)

        Nrf2_1625DN = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n1/16.25_tr1.txt'))[
                      :, 1]
        Nrf2_1625DN = pd.DataFrame(Nrf2_1625DN)
        Nrf2_1625DN[1] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n1/16.25_tr2.txt'))[
                         :, 1]
        Nrf2_1625DN[2] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n2/16.25_tr1.txt'))[
                         :, 1]
        Nrf2_1625DN[3] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n2/16.25_tr2.txt'))[
                         :, 1]
        Nrf2_1625DN[4] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n3/16.25_tr1.txt'))[
                         :, 1]
        Nrf2_1625DN[5] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n3/16.25_tr2.txt'))[
                         :, 1]
        Nrf2_1625DN = np.array(Nrf2_1625DN)

        Nrf2_3500DN = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n1/35.0_tr1.txt'))[
                      :, 1]
        Nrf2_3500DN = pd.DataFrame(Nrf2_3500DN)
        Nrf2_3500DN[1] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n1/35.0_tr2.txt'))[
                         :, 1]
        Nrf2_3500DN[2] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n2/35.0_tr1.txt'))[
                         :, 1]
        Nrf2_3500DN[3] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n2/35.0_tr2.txt'))[
                         :, 1]
        Nrf2_3500DN[4] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n3/35.0_tr1.txt'))[
                         :, 1]
        Nrf2_3500DN[5] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Nrf2 Sulf DMSO normalized 2000 n3/35.0_tr2.txt'))[
                         :, 1]
        Nrf2_3500DN = np.array(Nrf2_3500DN)

        datascalerlistNrf2 = np.array([0, 10])
        datascalerlistNrf2 = datascalerlistNrf2.reshape(2, 1)
        scaler.fit(datascalerlistNrf2)

        mean_Nrf2_36_035, Urange_Nrf2_36_035, Lrange_Nrf2_36_035, Rmean_Nrf2_36_035 = dataprocessingtrans(
            Nrf2_035DN)
        mean_Nrf2_36_075, Urange_Nrf2_36_075, Lrange_Nrf2_36_075, Rmean_Nrf2_36_075 = dataprocessingtrans(
            Nrf2_075DN)
        mean_Nrf2_36_162, Urange_Nrf2_36_162, Lrange_Nrf2_36_162, Rmean_Nrf2_36_162 = dataprocessingtrans(
            Nrf2_162DN)
        mean_Nrf2_36_350, Urange_Nrf2_36_350, Lrange_Nrf2_36_350, Rmean_Nrf2_36_350 = dataprocessingtrans(
            Nrf2_350DN)
        mean_Nrf2_36_754, Urange_Nrf2_36_754, Lrange_Nrf2_36_754, Rmean_Nrf2_36_754 = dataprocessingtrans(
            Nrf2_754DN)
        mean_Nrf2_36_1625, Urange_Nrf2_36_1625, Lrange_Nrf2_36_1625, Rmean_Nrf2_36_1625 = dataprocessingtrans(
            Nrf2_1625DN)
        mean_Nrf2_36_3500, Urange_Nrf2_36_3500, Lrange_Nrf2_36_3500, Rmean_Nrf2_36_3500 = dataprocessingtrans(
            Nrf2_3500DN)

        # Srxn1 36 hours
        Srxn1_035DN = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n1/0.35_tr1.txt'))[
                      :, 1]
        Srxn1_035DN = pd.DataFrame(Srxn1_035DN)
        Srxn1_035DN[1] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n1/0.35_tr2.txt'))[
                         :, 1]
        Srxn1_035DN[2] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n2/0.35_tr1.txt'))[
                         :, 1]
        Srxn1_035DN[3] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n2/0.35_tr2.txt'))[
                         :, 1]
        # Srxn1_035DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n3/0.35_tr1.txt'))[:, 1]
        # Srxn1_035DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n3/0.35_tr2.txt'))[:, 1]
        Srxn1_035DN = np.array(Srxn1_035DN)

        Srxn1_075DN = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n1/0.75_tr1.txt'))[
                      :, 1]
        Srxn1_075DN = pd.DataFrame(Srxn1_075DN)
        Srxn1_075DN[1] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n1/0.75_tr2.txt'))[
                         :, 1]
        Srxn1_075DN[2] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n2/0.75_tr1.txt'))[
                         :, 1]
        Srxn1_075DN[3] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n2/0.75_tr2.txt'))[
                         :, 1]
        # Srxn1_075DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n3/0.75_tr1.txt'))[:,1]
        # Srxn1_075DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n3/0.75_tr2.txt'))[:,1]
        Srxn1_075DN = np.array(Srxn1_075DN)

        Srxn1_162DN = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n1/1.62_tr1.txt'))[
                      :, 1]
        Srxn1_162DN = pd.DataFrame(Srxn1_162DN)
        Srxn1_162DN[1] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n1/1.62_tr2.txt'))[
                         :, 1]
        Srxn1_162DN[2] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n2/1.62_tr1.txt'))[
                         :, 1]
        Srxn1_162DN[3] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n2/1.62_tr2.txt'))[
                         :, 1]
        # Srxn1_162DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n3/1.62_tr1.txt'))[:,1]
        # Srxn1_162DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n3/1.62_tr2.txt'))[:,1]
        Srxn1_162DN = np.array(Srxn1_162DN)

        Srxn1_350DN = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n1/3.5_tr1.txt'))[
                      :, 1]
        Srxn1_350DN = pd.DataFrame(Srxn1_350DN)
        Srxn1_350DN[1] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n1/3.5_tr2.txt'))[
                         :, 1]
        Srxn1_350DN[2] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n2/3.5_tr1.txt'))[
                         :, 1]
        Srxn1_350DN[3] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n2/3.5_tr2.txt'))[
                         :, 1]
        # Srxn1_350DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n3/3.5_tr1.txt'))[:,1]
        # Srxn1_350DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n3/3.5_tr2.txt'))[:,1]
        Srxn1_350DN = np.array(Srxn1_350DN)

        Srxn1_754DN = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n1/7.54_tr1.txt'))[
                      :, 1]
        Srxn1_754DN = pd.DataFrame(Srxn1_754DN)
        Srxn1_754DN[1] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n1/7.54_tr2.txt'))[
                         :, 1]
        Srxn1_754DN[2] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n2/7.54_tr1.txt'))[
                         :, 1]
        Srxn1_754DN[3] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n2/7.54_tr2.txt'))[
                         :, 1]
        # Srxn1_754DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n3/7.54_tr1.txt'))[:,1]
        # Srxn1_754DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n3/7.54_tr2.txt'))[:,1]
        Srxn1_754DN = np.array(Srxn1_754DN)

        Srxn1_1625DN = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n1/16.25_tr1.txt'))[
                       :, 1]
        Srxn1_1625DN = pd.DataFrame(Srxn1_1625DN)
        Srxn1_1625DN[1] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n1/16.25_tr2.txt'))[
                          :, 1]
        Srxn1_1625DN[2] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n2/16.25_tr1.txt'))[
                          :, 1]
        Srxn1_1625DN[3] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n2/16.25_tr2.txt'))[
                          :, 1]
        # Srxn1_1625DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n3/16.25_tr1.txt'))[:,1]
        # Srxn1_1625DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n3/16.25_tr2.txt'))[:,1]
        Srxn1_1625DN = np.array(Srxn1_1625DN)

        Srxn1_3500DN = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n1/35.0_tr1.txt'))[
                       :, 1]
        Srxn1_3500DN = pd.DataFrame(Srxn1_3500DN)
        Srxn1_3500DN[1] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n1/35.0_tr2.txt'))[
                          :, 1]
        Srxn1_3500DN[2] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n2/35.0_tr1.txt'))[
                          :, 1]
        Srxn1_3500DN[3] = np.array(pd.read_csv(
            str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n2/35.0_tr2.txt'))[
                          :, 1]
        # Srxn1_3500DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n3/35.0_tr1.txt'))[:,1]
        # Srxn1_3500DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Sulf DMSO normalized 2000 n3/35.0_tr2.txt'))[:,1]
        Srxn1_3500DN = np.array(Srxn1_3500DN)

        datascalerlistSrxn1 = np.array([0, 100])
        datascalerlistSrxn1 = datascalerlistSrxn1.reshape(2, 1)
        scaler.fit(datascalerlistSrxn1)

        mean_Srxn1_36_035, Urange_Srxn1_36_035, Lrange_Srxn1_36_035, Rmean_Srxn1_36_035 = dataprocessingtrans(
            Srxn1_035DN)
        mean_Srxn1_36_075, Urange_Srxn1_36_075, Lrange_Srxn1_36_075, Rmean_Srxn1_36_075 = dataprocessingtrans(
            Srxn1_075DN)
        mean_Srxn1_36_162, Urange_Srxn1_36_162, Lrange_Srxn1_36_162, Rmean_Srxn1_36_162 = dataprocessingtrans(
            Srxn1_162DN)
        mean_Srxn1_36_350, Urange_Srxn1_36_350, Lrange_Srxn1_36_350, Rmean_Srxn1_36_350 = dataprocessingtrans(
            Srxn1_350DN)
        mean_Srxn1_36_754, Urange_Srxn1_36_754, Lrange_Srxn1_36_754, Rmean_Srxn1_36_754 = dataprocessingtrans(
            Srxn1_754DN)
        mean_Srxn1_36_1625, Urange_Srxn1_36_1625, Lrange_Srxn1_36_1625, Rmean_Srxn1_36_1625 = dataprocessingtrans(
            Srxn1_1625DN)
        mean_Srxn1_36_3500, Urange_Srxn1_36_3500, Lrange_Srxn1_36_3500, Rmean_Srxn1_36_3500 = dataprocessingtrans(
            Srxn1_3500DN)

    # CDDO DMSO normalized data 36 hours
    for h in range(1):
        Nrf2_001DN = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n1/0.01_tr1.txt'))[:, 1]
        Nrf2_001DN = pd.DataFrame(Nrf2_001DN)
        Nrf2_001DN[1] = np.array(pd.read_csv( str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n1/0.01_tr2.txt'))[:, 1]
        Nrf2_001DN[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n2/0.01_tr1.txt'))[:, 1]
        Nrf2_001DN[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n2/0.01_tr2.txt'))[:, 1]
        Nrf2_001DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n3/0.01_tr1.txt'))[:, 1]
        Nrf2_001DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n3/0.01_tr2.txt'))[:, 1]
        Nrf2_001DN = np.array(Nrf2_001DN)

        Nrf2_002DN = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n1/0.02_tr1.txt'))[:,1]
        Nrf2_002DN = pd.DataFrame(Nrf2_002DN)
        Nrf2_002DN[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n1/0.02_tr2.txt'))[:,1]
        Nrf2_002DN[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n2/0.02_tr1.txt'))[:,1]
        Nrf2_002DN[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n2/0.02_tr2.txt'))[:,1]
        Nrf2_002DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n3/0.02_tr1.txt'))[:,1]
        Nrf2_002DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n3/0.02_tr2.txt'))[:,1]
        Nrf2_002DN = np.array(Nrf2_002DN)

        Nrf2_005DN = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n1/0.05_tr1.txt'))[:,1]
        Nrf2_005DN = pd.DataFrame(Nrf2_005DN)
        Nrf2_005DN[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n1/0.05_tr2.txt'))[:,1]
        Nrf2_005DN[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n2/0.05_tr1.txt'))[:,1]
        Nrf2_005DN[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n2/0.05_tr2.txt'))[:,1]
        Nrf2_005DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n3/0.05_tr1.txt'))[:,1]
        Nrf2_005DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n3/0.05_tr2.txt'))[:,1]
        Nrf2_005DN = np.array(Nrf2_005DN)

        Nrf2_010DN = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n1/0.1_tr1.txt'))[:,1]
        Nrf2_010DN = pd.DataFrame(Nrf2_010DN)
        Nrf2_010DN[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n1/0.1_tr2.txt'))[:,1]
        Nrf2_010DN[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n2/0.1_tr1.txt'))[:,1]
        Nrf2_010DN[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n2/0.1_tr2.txt'))[:,1]
        Nrf2_010DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n3/0.1_tr1.txt'))[:,1]
        Nrf2_010DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n3/0.1_tr2.txt'))[:,1]
        Nrf2_010DN = np.array(Nrf2_010DN)

        Nrf2_022DN = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n1/0.22_tr1.txt'))[:,1]
        Nrf2_022DN = pd.DataFrame(Nrf2_022DN)
        Nrf2_022DN[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n1/0.22_tr2.txt'))[:,1]
        Nrf2_022DN[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n2/0.22_tr1.txt'))[:,1]
        Nrf2_022DN[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n2/0.22_tr2.txt'))[:,1]
        Nrf2_022DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n3/0.22_tr1.txt'))[:,1]
        Nrf2_022DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n3/0.22_tr2.txt'))[:,1]
        Nrf2_022DN = np.array(Nrf2_022DN)

        Nrf2_046DN = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n1/0.46_tr1.txt'))[:,1]
        Nrf2_046DN = pd.DataFrame(Nrf2_046DN)
        Nrf2_046DN[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n1/0.46_tr2.txt'))[:,1]
        Nrf2_046DN[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n2/0.46_tr1.txt'))[:,1]
        Nrf2_046DN[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n2/0.46_tr2.txt'))[:,1]
        Nrf2_046DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n3/0.46_tr1.txt'))[:,1]
        Nrf2_046DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n3/0.46_tr2.txt'))[:,1]
        Nrf2_046DN = np.array(Nrf2_046DN)

        Nrf2_100DN = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n1/1.0_tr1.txt'))[:,1]
        Nrf2_100DN = pd.DataFrame(Nrf2_100DN)
        Nrf2_100DN[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n1/1.0_tr2.txt'))[:,1]
        Nrf2_100DN[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n2/1.0_tr1.txt'))[:,1]
        Nrf2_100DN[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n2/1.0_tr2.txt'))[:,1]
        Nrf2_100DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n3/1.0_tr1.txt'))[:,1]
        Nrf2_100DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 CDDO DMSO normalized 2000 n3/1.0_tr2.txt'))[:,1]
        Nrf2_100DN = np.array(Nrf2_100DN)


        scaler.fit(datascalerlistNrf2)

        mean_Nrf2_36_001, Urange_Nrf2_36_001, Lrange_Nrf2_36_001, Rmean_Nrf2_36_001 = dataprocessingtrans(Nrf2_001DN)
        mean_Nrf2_36_002, Urange_Nrf2_36_002, Lrange_Nrf2_36_002, Rmean_Nrf2_36_002 = dataprocessingtrans(Nrf2_002DN)
        mean_Nrf2_36_005, Urange_Nrf2_36_005, Lrange_Nrf2_36_005, Rmean_Nrf2_36_005 = dataprocessingtrans(Nrf2_005DN)
        mean_Nrf2_36_010, Urange_Nrf2_36_010, Lrange_Nrf2_36_010, Rmean_Nrf2_36_010 = dataprocessingtrans(Nrf2_010DN)
        mean_Nrf2_36_022, Urange_Nrf2_36_022, Lrange_Nrf2_36_022, Rmean_Nrf2_36_022 = dataprocessingtrans(Nrf2_022DN)
        mean_Nrf2_36_046, Urange_Nrf2_36_046, Lrange_Nrf2_36_046, Rmean_Nrf2_36_046 = dataprocessingtrans(Nrf2_046DN)
        mean_Nrf2_36_100, Urange_Nrf2_36_100, Lrange_Nrf2_36_100, Rmean_Nrf2_36_100 = dataprocessingtrans(Nrf2_100DN)



        # Srxn1 36 hours
        Srxn1_001DN = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n1/0.01_tr1.txt'))[:, 1]
        Srxn1_001DN = pd.DataFrame(Srxn1_001DN)
        Srxn1_001DN[1] = np.array(pd.read_csv( str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n1/0.01_tr2.txt'))[:, 1]
        Srxn1_001DN[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n2/0.01_tr1.txt'))[:, 1]
        Srxn1_001DN[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n2/0.01_tr2.txt'))[:, 1]
        # Srxn1_001DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n3/0.01_tr1.txt'))[:, 1]
        # Srxn1_001DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n3/0.01_tr2.txt'))[:, 1]
        Srxn1_001DN = np.array(Srxn1_001DN)

        Srxn1_002DN = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n1/0.02_tr1.txt'))[:,1]
        Srxn1_002DN = pd.DataFrame(Srxn1_002DN)
        Srxn1_002DN[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n1/0.02_tr2.txt'))[:,1]
        Srxn1_002DN[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n2/0.02_tr1.txt'))[:,1]
        Srxn1_002DN[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n2/0.02_tr2.txt'))[:,1]
        # Srxn1_002DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n3/0.02_tr1.txt'))[:,1]
        # Srxn1_002DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n3/0.02_tr2.txt'))[:,1]
        Srxn1_002DN = np.array(Srxn1_002DN)

        Srxn1_005DN = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n1/0.05_tr1.txt'))[:,1]
        Srxn1_005DN = pd.DataFrame(Srxn1_005DN)
        Srxn1_005DN[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n1/0.05_tr2.txt'))[:,1]
        Srxn1_005DN[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n2/0.05_tr1.txt'))[:,1]
        Srxn1_005DN[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n2/0.05_tr2.txt'))[:,1]
        # Srxn1_005DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n3/0.05_tr1.txt'))[:,1]
        # Srxn1_005DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n3/0.05_tr2.txt'))[:,1]
        Srxn1_005DN = np.array(Srxn1_005DN)

        Srxn1_010DN = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n1/0.1_tr1.txt'))[:,1]
        Srxn1_010DN = pd.DataFrame(Srxn1_010DN)
        Srxn1_010DN[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n1/0.1_tr2.txt'))[:,1]
        Srxn1_010DN[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n2/0.1_tr1.txt'))[:,1]
        Srxn1_010DN[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n2/0.1_tr2.txt'))[:,1]
        # Srxn1_010DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n3/0.1_tr1.txt'))[:,1]
        # Srxn1_010DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n3/0.1_tr2.txt'))[:,1]
        Srxn1_010DN = np.array(Srxn1_010DN)

        Srxn1_022DN = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n1/0.22_tr1.txt'))[:,1]
        Srxn1_022DN = pd.DataFrame(Srxn1_022DN)
        Srxn1_022DN[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n1/0.22_tr2.txt'))[:,1]
        Srxn1_022DN[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n2/0.22_tr1.txt'))[:,1]
        Srxn1_022DN[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n2/0.22_tr2.txt'))[:,1]
        # Srxn1_022DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n3/0.22_tr1.txt'))[:,1]
        # Srxn1_022DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n3/0.22_tr2.txt'))[:,1]
        Srxn1_022DN = np.array(Srxn1_022DN)

        Srxn1_046DN = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n1/0.46_tr1.txt'))[:,1]
        Srxn1_046DN = pd.DataFrame(Srxn1_046DN)
        Srxn1_046DN[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n1/0.46_tr2.txt'))[:,1]
        Srxn1_046DN[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n2/0.46_tr1.txt'))[:,1]
        Srxn1_046DN[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n2/0.46_tr2.txt'))[:,1]
        # Srxn1_046DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n3/0.46_tr1.txt'))[:,1]
        # Srxn1_046DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n3/0.46_tr2.txt'))[:,1]
        Srxn1_046DN = np.array(Srxn1_046DN)

        Srxn1_100DN = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n1/1.0_tr1.txt'))[:,1]
        Srxn1_100DN = pd.DataFrame(Srxn1_100DN)
        Srxn1_100DN[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n1/1.0_tr2.txt'))[:,1]
        Srxn1_100DN[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n2/1.0_tr1.txt'))[:,1]
        Srxn1_100DN[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n2/1.0_tr2.txt'))[:,1]
        # Srxn1_100DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n3/1.0_tr1.txt'))[:,1]
        # Srxn1_100DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 CDDO DMSO normalized 2000 n3/1.0_tr2.txt'))[:,1]
        Srxn1_100DN = np.array(Srxn1_100DN)


        scaler.fit(datascalerlistSrxn1)

        mean_Srxn1_36_001, Urange_Srxn1_36_001, Lrange_Srxn1_36_001, Rmean_Srxn1_36_001 = dataprocessingtrans(Srxn1_001DN)
        mean_Srxn1_36_002, Urange_Srxn1_36_002, Lrange_Srxn1_36_002, Rmean_Srxn1_36_002 = dataprocessingtrans(Srxn1_002DN)
        mean_Srxn1_36_005, Urange_Srxn1_36_005, Lrange_Srxn1_36_005, Rmean_Srxn1_36_005 = dataprocessingtrans(Srxn1_005DN)
        mean_Srxn1_36_010, Urange_Srxn1_36_010, Lrange_Srxn1_36_010, Rmean_Srxn1_36_010 = dataprocessingtrans(Srxn1_010DN)
        mean_Srxn1_36_022, Urange_Srxn1_36_022, Lrange_Srxn1_36_022, Rmean_Srxn1_36_022 = dataprocessingtrans(Srxn1_022DN)
        mean_Srxn1_36_046, Urange_Srxn1_36_046, Lrange_Srxn1_36_046, Rmean_Srxn1_36_046 = dataprocessingtrans(Srxn1_046DN)
        mean_Srxn1_36_100, Urange_Srxn1_36_100, Lrange_Srxn1_36_100, Rmean_Srxn1_36_100 = dataprocessingtrans(Srxn1_100DN)

    # Andr DMSO normalized data 36 hours
    for h in range(1):
        Nrf2_010DNandr = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n1/0.1_tr1.txt'))[:, 1]
        Nrf2_010DNandr = pd.DataFrame(Nrf2_010DNandr)
        Nrf2_010DNandr[1] = np.array(pd.read_csv( str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n1/0.1_tr2.txt'))[:, 1]
        Nrf2_010DNandr[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n2/0.1_tr1.txt'))[:, 1]
        Nrf2_010DNandr[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n2/0.1_tr2.txt'))[:, 1]
        Nrf2_010DNandr[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n3/0.1_tr1.txt'))[:, 1]
        Nrf2_010DNandr[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n3/0.1_tr2.txt'))[:, 1]
        Nrf2_010DNandr = np.array(Nrf2_010DNandr)

        Nrf2_032DNandr = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n1/0.32_tr1.txt'))[:,1]
        Nrf2_032DNandr = pd.DataFrame(Nrf2_032DNandr)
        Nrf2_032DNandr[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n1/0.32_tr2.txt'))[:,1]
        Nrf2_032DNandr[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n2/0.32_tr1.txt'))[:,1]
        Nrf2_032DNandr[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n2/0.32_tr2.txt'))[:,1]
        Nrf2_032DNandr[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n3/0.32_tr1.txt'))[:,1]
        Nrf2_032DNandr[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n3/0.32_tr2.txt'))[:,1]
        Nrf2_032DNandr = np.array(Nrf2_032DNandr)

        Nrf2_100DNandr = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n1/1.0_tr1.txt'))[:,1]
        Nrf2_100DNandr = pd.DataFrame(Nrf2_100DNandr)
        Nrf2_100DNandr[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n1/1.0_tr2.txt'))[:,1]
        Nrf2_100DNandr[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n2/1.0_tr1.txt'))[:,1]
        Nrf2_100DNandr[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n2/1.0_tr2.txt'))[:,1]
        Nrf2_100DNandr[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n3/1.0_tr1.txt'))[:,1]
        Nrf2_100DNandr[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n3/1.0_tr2.txt'))[:,1]
        Nrf2_100DNandr = np.array(Nrf2_100DNandr)

        Nrf2_316DNandr = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n1/3.16_tr1.txt'))[:,1]
        Nrf2_316DNandr = pd.DataFrame(Nrf2_316DNandr)
        Nrf2_316DNandr[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n1/3.16_tr2.txt'))[:,1]
        Nrf2_316DNandr[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n2/3.16_tr1.txt'))[:,1]
        Nrf2_316DNandr[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n2/3.16_tr2.txt'))[:,1]
        Nrf2_316DNandr[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n3/3.16_tr1.txt'))[:,1]
        Nrf2_316DNandr[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n3/3.16_tr2.txt'))[:,1]
        Nrf2_316DNandr = np.array(Nrf2_316DNandr)

        Nrf2_1000DNandr = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n1/10.0_tr1.txt'))[:,1]
        Nrf2_1000DNandr = pd.DataFrame(Nrf2_1000DNandr)
        Nrf2_1000DNandr[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n1/10.0_tr2.txt'))[:,1]
        Nrf2_1000DNandr[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n2/10.0_tr1.txt'))[:,1]
        Nrf2_1000DNandr[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n2/10.0_tr2.txt'))[:,1]
        Nrf2_1000DNandr[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n3/10.0_tr1.txt'))[:,1]
        Nrf2_1000DNandr[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n3/10.0_tr2.txt'))[:,1]
        Nrf2_1000DNandr = np.array(Nrf2_1000DNandr)

        Nrf2_3162DNandr = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n1/31.62_tr1.txt'))[:,1]
        Nrf2_3162DNandr = pd.DataFrame(Nrf2_3162DNandr)
        Nrf2_3162DNandr[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n1/31.62_tr2.txt'))[:,1]
        Nrf2_3162DNandr[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n2/31.62_tr1.txt'))[:,1]
        Nrf2_3162DNandr[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n2/31.62_tr2.txt'))[:,1]
        Nrf2_3162DNandr[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n3/31.62_tr1.txt'))[:,1]
        Nrf2_3162DNandr[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n3/31.62_tr2.txt'))[:,1]
        Nrf2_3162DNandr = np.array(Nrf2_3162DNandr)

        Nrf2_10000DNandr = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n1/100.0_tr1.txt'))[:,1]
        Nrf2_10000DNandr = pd.DataFrame(Nrf2_10000DNandr)
        Nrf2_10000DNandr[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n1/100.0_tr2.txt'))[:,1]
        Nrf2_10000DNandr[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n2/100.0_tr1.txt'))[:,1]
        Nrf2_10000DNandr[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n2/100.0_tr2.txt'))[:,1]
        Nrf2_10000DNandr[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n3/100.0_tr1.txt'))[:,1]
        Nrf2_10000DNandr[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Nrf2 Andr DMSO normalized 2000 n3/100.0_tr2.txt'))[:,1]
        Nrf2_10000DNandr = np.array(Nrf2_10000DNandr)


        scaler.fit(datascalerlistNrf2)

        mean_Nrf2_36_010, Urange_Nrf2_36_010, Lrange_Nrf2_36_010, Rmean_Andr_Nrf2_36_010 = dataprocessingtrans(Nrf2_010DNandr)
        mean_Nrf2_36_032, Urange_Nrf2_36_032, Lrange_Nrf2_36_032, Rmean_Andr_Nrf2_36_032 = dataprocessingtrans(Nrf2_032DNandr)
        mean_Nrf2_36_100, Urange_Nrf2_36_100, Lrange_Nrf2_36_100, Rmean_Andr_Nrf2_36_100 = dataprocessingtrans(Nrf2_100DNandr)
        mean_Nrf2_36_316, Urange_Nrf2_36_316, Lrange_Nrf2_36_316, Rmean_Andr_Nrf2_36_316 = dataprocessingtrans(Nrf2_316DNandr)
        mean_Nrf2_36_1000, Urange_Nrf2_36_1000, Lrange_Nrf2_36_1000, Rmean_Andr_Nrf2_36_1000 = dataprocessingtrans(Nrf2_1000DNandr)
        mean_Nrf2_36_3162, Urange_Nrf2_36_3162, Lrange_Nrf2_36_3162, Rmean_Andr_Nrf2_36_3162 = dataprocessingtrans(Nrf2_3162DNandr)
        mean_Nrf2_36_10000, Urange_Nrf2_36_10000, Lrange_Nrf2_36_10000, Rmean_Andr_Nrf2_36_10000 = dataprocessingtrans(Nrf2_10000DNandr)



        # Srxn1 36 hours
        Srxn1_010DN = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n1/0.1_tr1.txt'))[:, 1]
        Srxn1_010DN = pd.DataFrame(Srxn1_010DN)
        Srxn1_010DN[1] = np.array(pd.read_csv( str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n1/0.1_tr2.txt'))[:, 1]
        Srxn1_010DN[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n2/0.1_tr1.txt'))[:, 1]
        Srxn1_010DN[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n2/0.1_tr2.txt'))[:, 1]
        # Srxn1_010DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n3/0.1_tr1.txt'))[:, 1]
        # Srxn1_010DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n3/0.1_tr2.txt'))[:, 1]
        Srxn1_010DN = np.array(Srxn1_010DN)

        Srxn1_032DN = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n1/0.32_tr1.txt'))[:,1]
        Srxn1_032DN = pd.DataFrame(Srxn1_032DN)
        Srxn1_032DN[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n1/0.32_tr2.txt'))[:,1]
        Srxn1_032DN[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n2/0.32_tr1.txt'))[:,1]
        Srxn1_032DN[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n2/0.32_tr2.txt'))[:,1]
        # Srxn1_032DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n3/0.32_tr1.txt'))[:,1]
        # Srxn1_032DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n3/0.32_tr2.txt'))[:,1]
        Srxn1_032DN = np.array(Srxn1_032DN)

        Srxn1_100DN = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n1/1.0_tr1.txt'))[:,1]
        Srxn1_100DN = pd.DataFrame(Srxn1_100DN)
        Srxn1_100DN[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n1/1.0_tr2.txt'))[:,1]
        Srxn1_100DN[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n2/1.0_tr1.txt'))[:,1]
        Srxn1_100DN[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n2/1.0_tr2.txt'))[:,1]
        # Srxn1_100DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n3/1.0_tr1.txt'))[:,1]
        # Srxn1_100DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n3/1.0_tr2.txt'))[:,1]
        Srxn1_100DN = np.array(Srxn1_100DN)

        Srxn1_316DN = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n1/3.16_tr1.txt'))[:,1]
        Srxn1_316DN = pd.DataFrame(Srxn1_316DN)
        Srxn1_316DN[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n1/3.16_tr2.txt'))[:,1]
        Srxn1_316DN[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n2/3.16_tr1.txt'))[:,1]
        Srxn1_316DN[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n2/3.16_tr2.txt'))[:,1]
        # Srxn1_316DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n3/3.16_tr1.txt'))[:,1]
        # Srxn1_316DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n3/3.16_tr2.txt'))[:,1]
        Srxn1_316DN = np.array(Srxn1_316DN)

        Srxn1_1000DN = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n1/10.0_tr1.txt'))[:,1]
        Srxn1_1000DN = pd.DataFrame(Srxn1_1000DN)
        Srxn1_1000DN[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n1/10.0_tr2.txt'))[:,1]
        Srxn1_1000DN[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n2/10.0_tr1.txt'))[:,1]
        Srxn1_1000DN[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n2/10.0_tr2.txt'))[:,1]
        # Srxn1_1000DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n3/10.0_tr1.txt'))[:,1]
        # Srxn1_1000DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n3/10.0_tr2.txt'))[:,1]
        Srxn1_1000DN = np.array(Srxn1_1000DN)

        Srxn1_3162DN = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n1/31.62_tr1.txt'))[:,1]
        Srxn1_3162DN = pd.DataFrame(Srxn1_3162DN)
        Srxn1_3162DN[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n1/31.62_tr2.txt'))[:,1]
        Srxn1_3162DN[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n2/31.62_tr1.txt'))[:,1]
        Srxn1_3162DN[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n2/31.62_tr2.txt'))[:,1]
        # Srxn1_3162DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n3/31.62_tr1.txt'))[:,1]
        # Srxn1_3162DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n3/31.62_tr2.txt'))[:,1]
        Srxn1_3162DN = np.array(Srxn1_3162DN)

        Srxn1_10000DN = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n1/100.0_tr1.txt'))[:,1]
        Srxn1_10000DN = pd.DataFrame(Srxn1_10000DN)
        Srxn1_10000DN[1] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n1/100.0_tr2.txt'))[:,1]
        Srxn1_10000DN[2] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n2/100.0_tr1.txt'))[:,1]
        Srxn1_10000DN[3] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n2/100.0_tr2.txt'))[:,1]
        # Srxn1_10000DN[4] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n3/100.0_tr1.txt'))[:,1]
        # Srxn1_10000DN[5] = np.array(pd.read_csv(str(directory) + 'Data/Training/Srxn1 Andr DMSO normalized 2000 n3/100.0_tr2.txt'))[:,1]
        Srxn1_10000DN = np.array(Srxn1_10000DN)


        scaler.fit(datascalerlistSrxn1)

        mean_Srxn1_36_010, Urange_Srxn1_36_010, Lrange_Srxn1_36_010, Rmean_Andr_Srxn1_36_010 = dataprocessingtrans(Srxn1_010DN)
        mean_Srxn1_36_032, Urange_Srxn1_36_032, Lrange_Srxn1_36_032, Rmean_Andr_Srxn1_36_032 = dataprocessingtrans(Srxn1_032DN)
        mean_Srxn1_36_100, Urange_Srxn1_36_100, Lrange_Srxn1_36_100, Rmean_Andr_Srxn1_36_100 = dataprocessingtrans(Srxn1_100DN)
        mean_Srxn1_36_316, Urange_Srxn1_36_316, Lrange_Srxn1_36_316, Rmean_Andr_Srxn1_36_316 = dataprocessingtrans(Srxn1_316DN)
        mean_Srxn1_36_1000, Urange_Srxn1_36_1000, Lrange_Srxn1_36_1000, Rmean_Andr_Srxn1_36_1000 = dataprocessingtrans(Srxn1_1000DN)
        mean_Srxn1_36_3162, Urange_Srxn1_36_3162, Lrange_Srxn1_36_3162, Rmean_Andr_Srxn1_36_3162 = dataprocessingtrans(Srxn1_3162DN)
        mean_Srxn1_36_10000, Urange_Srxn1_36_10000, Lrange_Srxn1_36_10000, Rmean_Andr_Srxn1_36_10000 = dataprocessingtrans(Srxn1_10000DN)

# 6. Scaler configurations
def scaler_config(config):
    # SUL config
    if config == 0.1:
        # Setup 1
        datascalerlistNrf2 = np.array([0, 3])
        datascalerlistSrxn1 = np.array([0, 3])
        # used in double scaling
        # diffscalerlistSrxn1 = np.array([-0.001, 0.005])
        diffscalerlistSrxn1 = np.array([-0.001, 0.005])
        diffscalerlistNrf2 = np.array([-10,15])
    # SUL config 2
    if config == 0.12:
        # Setup 1
        datascalerlistNrf2 = np.array([0, 3])
        datascalerlistSrxn1 = np.array([0, 3])
        # used in double scaling
        # diffscalerlistSrxn1 = np.array([-0.001, 0.005])
        diffscalerlistSrxn1 = np.array([-0.002, 0.005])
        diffscalerlistNrf2 = np.array([-10,15])
    # SUL config 3
    if config == 0.13:
        # Setup 1
        datascalerlistNrf2 = np.array([-3, 3])
        datascalerlistSrxn1 = np.array([-3, 3])
        # used in double scaling
        # diffscalerlistSrxn1 = np.array([-0.001, 0.005])
        diffscalerlistSrxn1 = np.array([-0.005, 0.005])
        diffscalerlistNrf2 = np.array([-10,15])
    # SUL config 4 (conv LSTM)
    if config == 0.14:
        # Setup 1
        datascalerlistNrf2 = np.array([0, 2])
        datascalerlistSrxn1 = np.array([0, 3])
        # used in double scaling
        # diffscalerlistSrxn1 = np.array([-0.001, 0.005])
        diffscalerlistSrxn1 = np.array([-0.005, 0.005])
        diffscalerlistNrf2 = np.array([-10,15])
    # SUL config 5 (conv LSTM)
    if config == 0.15:
        # Setup 1
        datascalerlistNrf2 = np.array([0, 1])
        datascalerlistSrxn1 = np.array([0, 2])
        # used in double scaling
        # diffscalerlistSrxn1 = np.array([-0.001, 0.005])
        diffscalerlistSrxn1 = np.array([-0.005, 0.005])
        diffscalerlistNrf2 = np.array([-10,15])
    # SUL config 6 (NN1 conv LSTM)
    if config == 0.16:
        datascalerlistNrf2 = np.array([0, 1.5])
        datascalerlistSrxn1 = np.array([-3, 3])
        # used in double scaling
        # diffscalerlistSrxn1 = np.array([-0.001, 0.005])
        diffscalerlistSrxn1 = np.array([-0.005, 0.005])
        diffscalerlistNrf2 = np.array([-10,15])
    if config == 0.17:
        datascalerlistNrf2 = np.array([0, 1])
        datascalerlistSrxn1 = np.array([-3, 3])
        # used in double scaling
        # diffscalerlistSrxn1 = np.array([-0.001, 0.005])
        diffscalerlistSrxn1 = np.array([-0.005, 0.005])
        diffscalerlistNrf2 = np.array([-10,15])

    # ES1 config
    if config == 1:
        # Setup 1
        datascalerlistNrf2 = np.array([0, 1200])
        datascalerlistSrxn1 = np.array([-1, 4])
        # used in double scaling
        # diffscalerlistSrxn1 = np.array([-0.001, 0.005])
        diffscalerlistSrxn1 = np.array([-0.001, 0.005])
        diffscalerlistNrf2 = np.array([-10,15])
    if config == 2:
        # Setup 2
        datascalerlistNrf2 = np.array([0, 2400])
        datascalerlistSrxn1 = np.array([-2, 6])
        diffscalerlist = np.array([-0.005, 0.005])
    # ES3 config
    if config == 3:
        datascalerlistNrf2 = np.array([-1, 5])
        datascalerlistSrxn1 = np.array([-2, 9])
        diffscalerlistSrxn1 = np.array([-0.005, 0.005])
        diffscalerlistNrf2 = np.array([-10, 15])
    # ES5 config
    if config == 5.1:
        datascalerlistNrf2 = np.array([0, 5])
        datascalerlistSrxn1 = np.array([-7, 8])
        diffscalerlistSrxn1 = np.array([-0.005, 0.01])
        diffscalerlistNrf2 = np.array([-10, 15])
    if config == 5.2:
        datascalerlistNrf2 = np.array([0, 5])
        datascalerlistSrxn1 = np.array([-7, 8])
        diffscalerlistSrxn1 = np.array([-0.005, 0.1])
        diffscalerlistNrf2 = np.array([-10, 15])

    if config == 5.3:
        datascalerlistNrf2 = np.array([0, 5])
        datascalerlistSrxn1 = np.array([-7, 8])
        diffscalerlistSrxn1 = np.array([-0.005, 0.1])
        diffscalerlistNrf2 = np.array([-10, 15])
    if config == 5.4:
        datascalerlistNrf2 = np.array([0, 5])
        datascalerlistSrxn1 = np.array([-8, 8])
        diffscalerlistSrxn1 = np.array([-0.02, 0.1])
        diffscalerlistNrf2 = np.array([-10, 15])

    datascalerlistSrxn1 = datascalerlistSrxn1.reshape(2, 1)
    datascalerlistNrf2 = datascalerlistNrf2.reshape(2, 1)
    diffscalerlistSrxn1 = diffscalerlistSrxn1.reshape(2, 1)
    diffscalerlistNrf2 = diffscalerlistNrf2.reshape(2, 1)
    return datascalerlistNrf2, datascalerlistSrxn1, diffscalerlistSrxn1, diffscalerlistNrf2

# 7. Forming processed dataframes
for h in range(1):
    # Index generators for training and validation respectively (identical in function however)
    def trainindex(traindoselist, datatype):
        traindoseindexlist = []
        if datatype == 'ES1' or datatype == 'ES5':
            for i in range(len(traindoselist)):
                if traindoselist[i] == 1:
                    index = 0
                    traindoseindexlist.append(index)
                if traindoselist[i] == 2:
                    index = 1
                    traindoseindexlist.append(index)
                if traindoselist[i] == 3:
                    index = 2
                    traindoseindexlist.append(index)
                if traindoselist[i] == 4:
                    index = 3
                    traindoseindexlist.append(index)
                if traindoselist[i] == 5:
                    index = 4
                    traindoseindexlist.append(index)
                if traindoselist[i] == 6:
                    index = 5
                    traindoseindexlist.append(index)
                if traindoselist[i] == 7:
                    index = 6
                    traindoseindexlist.append(index)
                if traindoselist[i] == 8:
                    index = 7
                    traindoseindexlist.append(index)
        if datatype == 'SUL':
            for i in range(len(traindoselist)):
                if traindoselist[i] == 0.35:
                    index = 0
                    traindoseindexlist.append(index)
                if traindoselist[i] == 0.75:
                    index = 1
                    traindoseindexlist.append(index)
                if traindoselist[i] == 1.62:
                    index = 2
                    traindoseindexlist.append(index)
                if traindoselist[i] == 3.5:
                    index = 3
                    traindoseindexlist.append(index)
                if traindoselist[i] == 7.54:
                    index = 4
                    traindoseindexlist.append(index)
                if traindoselist[i] == 16.25:
                    index = 5
                    traindoseindexlist.append(index)
                if traindoselist[i] == 35.00:
                    index = 6
                    traindoseindexlist.append(index)
        return traindoseindexlist
    def valindex(valdoselist, datatype):
        valdoseindexlist = []
        if datatype == 'ES1' or datatype == 'ES5':
            for i in range(len(valdoselist)):
                if valdoselist[i] == 1:
                    index = 0
                    valdoseindexlist.append(index)
                if valdoselist[i] == 2:
                    index = 1
                    valdoseindexlist.append(index)
                if valdoselist[i] == 3:
                    index = 2
                    valdoseindexlist.append(index)
                if valdoselist[i] == 4:
                    index = 3
                    valdoseindexlist.append(index)
                if valdoselist[i] == 5:
                    index = 4
                    valdoseindexlist.append(index)
                if valdoselist[i] == 6:
                    index = 5
                    valdoseindexlist.append(index)
                if valdoselist[i] == 7:
                    index = 6
                    valdoseindexlist.append(index)
                if valdoselist[i] == 8:
                    index = 7
                    valdoseindexlist.append(index)
        if datatype == 'SUL':
            for i in range(len(valdoselist)):
                if valdoselist[i] == 0.35:
                    index = 0
                    valdoseindexlist.append(index)
                if valdoselist[i] == 0.75:
                    index = 1
                    valdoseindexlist.append(index)
                if valdoselist[i] == 1.62:
                    index = 2
                    valdoseindexlist.append(index)
                if valdoselist[i] == 3.5:
                    index = 3
                    valdoseindexlist.append(index)
                if valdoselist[i] == 7.54:
                    index = 4
                    valdoseindexlist.append(index)
                if valdoselist[i] == 16.25:
                    index = 5
                    valdoseindexlist.append(index)
                if valdoselist[i] == 35.00:
                    index = 6
                    valdoseindexlist.append(index)
        # if datatype == 'CDDO':
        return valdoseindexlist

    # Multiple exposure index generator (for multiple exposure dataset)
    def multiexpindex(exp1, exp2, compound):
        traindoseindexlist = []
        if compound == 'SUL':
            doselist = [0, 0.35, 0.75, 1.62, 3.5, 7.54, 16.25, 35.00]
            doselistshort = [0.35, 0.75, 1.62, 3.5, 7.54, 16.25, 35.00]
            if exp1 == 0:
                for i in range(len(doselist)):
                    if exp1 == doselist[i]:
                        index1 = i
                for i in range(len(doselistshort)):
                    if exp2 == doselistshort[i]:
                        index2 = i
            else:
                for i in range(len(doselist)):
                    if exp1 == doselist[i]:
                        index1 = i
                    if exp2 == doselist[i]:
                        index2 = i
            print('index1 = ' + str(index1))
            print('index2 = ' + str(index2))
            index = (index1*8 + index2)

        return index

    # collecting dataframes
    for h in range(1):
        pDF_Nrf2_Rmean_ES1 = pd.DataFrame(Nrf2_training_ES1_1)
        pDF_Nrf2_Rmean_ES1[1] = Nrf2_training_ES1_2
        pDF_Nrf2_Rmean_ES1[2] = Nrf2_training_ES1_3
        pDF_Nrf2_Rmean_ES1[3] = Nrf2_training_ES1_4
        pDF_Nrf2_Rmean_ES1[4] = Nrf2_test_ES1_1
        pDF_Nrf2_Rmean_ES1[5] = Nrf2_test_ES1_2
        pDF_Nrf2_Rmean_ES1 = np.array(pDF_Nrf2_Rmean_ES1)

        pDF_Srxn1_Rmean_ES1 = pd.DataFrame(Srxn1_training_ES1_1)
        pDF_Srxn1_Rmean_ES1[1] = Srxn1_training_ES1_2
        pDF_Srxn1_Rmean_ES1[2] = Srxn1_training_ES1_3
        pDF_Srxn1_Rmean_ES1[3] = Srxn1_training_ES1_4
        pDF_Srxn1_Rmean_ES1[4] = Srxn1_test_ES1_1
        pDF_Srxn1_Rmean_ES1[5] = Srxn1_test_ES1_2
        pDF_Srxn1_Rmean_ES1 = np.array(pDF_Srxn1_Rmean_ES1)

        pDF_Nrf2_Rmean_ES2 = pd.DataFrame(Nrf2_training_ES2_1)
        pDF_Nrf2_Rmean_ES2[1] = Nrf2_training_ES2_2
        pDF_Nrf2_Rmean_ES2[2] = Nrf2_training_ES2_3
        pDF_Nrf2_Rmean_ES2[3] = Nrf2_training_ES2_4
        pDF_Nrf2_Rmean_ES2[4] = Nrf2_test_ES2_1
        pDF_Nrf2_Rmean_ES2[5] = Nrf2_test_ES2_2
        pDF_Nrf2_Rmean_ES2 = np.array(pDF_Nrf2_Rmean_ES2)

        pDF_Srxn1_Rmean_ES2 = pd.DataFrame(Srxn1_training_ES2_1)
        pDF_Srxn1_Rmean_ES2[1] = Srxn1_training_ES2_2
        pDF_Srxn1_Rmean_ES2[2] = Srxn1_training_ES2_3
        pDF_Srxn1_Rmean_ES2[3] = Srxn1_training_ES2_4
        pDF_Srxn1_Rmean_ES2[4] = Srxn1_test_ES2_1
        pDF_Srxn1_Rmean_ES2[5] = Srxn1_test_ES2_2
        pDF_Srxn1_Rmean_ES2 = np.array(pDF_Srxn1_Rmean_ES2)

        pDF_Nrf2_Rmean_ES3 = pd.DataFrame(Nrf2_training_ES3_1)
        pDF_Nrf2_Rmean_ES3[1] = Nrf2_training_ES3_2
        pDF_Nrf2_Rmean_ES3[2] = Nrf2_training_ES3_3
        pDF_Nrf2_Rmean_ES3[3] = Nrf2_training_ES3_4
        pDF_Nrf2_Rmean_ES3[4] = Nrf2_test_ES3_1
        pDF_Nrf2_Rmean_ES3[5] = Nrf2_test_ES3_2
        pDF_Nrf2_Rmean_ES3[6] = Nrf2_test_ES3_3
        pDF_Nrf2_Rmean_ES3[7] = Nrf2_test_ES3_4
        pDF_Nrf2_Rmean_ES3 = np.array(pDF_Nrf2_Rmean_ES3)

        pDF_Srxn1_Rmean_ES3 = pd.DataFrame(Srxn1_training_ES3_1)
        pDF_Srxn1_Rmean_ES3[1] = Srxn1_training_ES3_2
        pDF_Srxn1_Rmean_ES3[2] = Srxn1_training_ES3_3
        pDF_Srxn1_Rmean_ES3[3] = Srxn1_training_ES3_4
        pDF_Srxn1_Rmean_ES3[4] = Srxn1_test_ES3_1
        pDF_Srxn1_Rmean_ES3[5] = Srxn1_test_ES3_2
        pDF_Srxn1_Rmean_ES3[6] = Srxn1_test_ES3_3
        pDF_Srxn1_Rmean_ES3[7] = Srxn1_test_ES3_4
        pDF_Srxn1_Rmean_ES3 = np.array(pDF_Srxn1_Rmean_ES3)

        pDF_Nrf2_Rmean_ES5 = pd.DataFrame(Nrf2_training_ES5_1)
        pDF_Nrf2_Rmean_ES5[1] = Nrf2_training_ES5_2
        pDF_Nrf2_Rmean_ES5[2] = Nrf2_training_ES5_3
        pDF_Nrf2_Rmean_ES5[3] = Nrf2_training_ES5_4
        pDF_Nrf2_Rmean_ES5[4] = Nrf2_test_ES5_1
        pDF_Nrf2_Rmean_ES5[5] = Nrf2_test_ES5_2
        pDF_Nrf2_Rmean_ES5[6] = Nrf2_test_ES5_3
        pDF_Nrf2_Rmean_ES5[7] = Nrf2_test_ES5_4
        pDF_Nrf2_Rmean_ES5 = np.array(pDF_Nrf2_Rmean_ES5)

        pDF_Srxn1_Rmean_ES5 = pd.DataFrame(Srxn1_training_ES5_1)
        pDF_Srxn1_Rmean_ES5[1] = Srxn1_training_ES5_2
        pDF_Srxn1_Rmean_ES5[2] = Srxn1_training_ES5_3
        pDF_Srxn1_Rmean_ES5[3] = Srxn1_training_ES5_4
        pDF_Srxn1_Rmean_ES5[4] = Srxn1_test_ES5_1
        pDF_Srxn1_Rmean_ES5[5] = Srxn1_test_ES5_2
        pDF_Srxn1_Rmean_ES5[6] = Srxn1_test_ES5_3
        pDF_Srxn1_Rmean_ES5[7] = Srxn1_test_ES5_4
        pDF_Srxn1_Rmean_ES5 = np.array(pDF_Srxn1_Rmean_ES5)
    
        pDF_Nrf2_Rmean_SUL32 = pd.DataFrame(Rmean_Nrf2_36_035)
        pDF_Nrf2_Rmean_SUL32[1] = Rmean_Nrf2_36_075
        pDF_Nrf2_Rmean_SUL32[2] = Rmean_Nrf2_36_162
        pDF_Nrf2_Rmean_SUL32[3] = Rmean_Nrf2_36_350
        pDF_Nrf2_Rmean_SUL32[4] = Rmean_Nrf2_36_754
        pDF_Nrf2_Rmean_SUL32[5] = Rmean_Nrf2_36_1625
        pDF_Nrf2_Rmean_SUL32[6] = Rmean_Nrf2_36_3500
        pDF_Nrf2_Rmean_SUL32 = np.array(pDF_Nrf2_Rmean_SUL32)

        pDF_Srxn1_Rmean_SUL32 = pd.DataFrame(Rmean_Srxn1_36_035)
        pDF_Srxn1_Rmean_SUL32[1] = Rmean_Srxn1_36_075
        pDF_Srxn1_Rmean_SUL32[2] = Rmean_Srxn1_36_162
        pDF_Srxn1_Rmean_SUL32[3] = Rmean_Srxn1_36_350
        pDF_Srxn1_Rmean_SUL32[4] = Rmean_Srxn1_36_754
        pDF_Srxn1_Rmean_SUL32[5] = Rmean_Srxn1_36_1625
        pDF_Srxn1_Rmean_SUL32[6] = Rmean_Srxn1_36_3500
        pDF_Srxn1_Rmean_SUL32 = np.array(pDF_Srxn1_Rmean_SUL32)
    
        pDF_Nrf2_Rmean_CDDO32 = pd.DataFrame(Rmean_Nrf2_36_001)
        pDF_Nrf2_Rmean_CDDO32[1] = Rmean_Nrf2_36_002
        pDF_Nrf2_Rmean_CDDO32[2] = Rmean_Nrf2_36_005
        pDF_Nrf2_Rmean_CDDO32[3] = Rmean_Nrf2_36_010
        pDF_Nrf2_Rmean_CDDO32[4] = Rmean_Nrf2_36_022
        pDF_Nrf2_Rmean_CDDO32[5] = Rmean_Nrf2_36_046
        pDF_Nrf2_Rmean_CDDO32[6] = Rmean_Nrf2_36_100
        pDF_Nrf2_Rmean_CDDO32 = np.array(pDF_Nrf2_Rmean_CDDO32)

        pDF_Srxn1_Rmean_CDDO32 = pd.DataFrame(Rmean_Srxn1_36_001)
        pDF_Srxn1_Rmean_CDDO32[1] = Rmean_Srxn1_36_002
        pDF_Srxn1_Rmean_CDDO32[2] = Rmean_Srxn1_36_005
        pDF_Srxn1_Rmean_CDDO32[3] = Rmean_Srxn1_36_010
        pDF_Srxn1_Rmean_CDDO32[4] = Rmean_Srxn1_36_022
        pDF_Srxn1_Rmean_CDDO32[5] = Rmean_Srxn1_36_046
        pDF_Srxn1_Rmean_CDDO32[6] = Rmean_Srxn1_36_100
        pDF_Srxn1_Rmean_CDDO32 = np.array(pDF_Srxn1_Rmean_CDDO32)


        pDF_Nrf2_Rmean_Andr32 = pd.DataFrame(Rmean_Andr_Nrf2_36_010)
        pDF_Nrf2_Rmean_Andr32[1] = Rmean_Andr_Nrf2_36_032
        pDF_Nrf2_Rmean_Andr32[2] = Rmean_Andr_Nrf2_36_100
        pDF_Nrf2_Rmean_Andr32[3] = Rmean_Andr_Nrf2_36_316
        pDF_Nrf2_Rmean_Andr32[4] = Rmean_Andr_Nrf2_36_1000
        pDF_Nrf2_Rmean_Andr32[5] = Rmean_Andr_Nrf2_36_3162
        pDF_Nrf2_Rmean_Andr32[6] = Rmean_Andr_Nrf2_36_10000
        pDF_Nrf2_Rmean_Andr32 = np.array(pDF_Nrf2_Rmean_Andr32)

        pDF_Srxn1_Rmean_Andr32 = pd.DataFrame(Rmean_Andr_Srxn1_36_010)
        pDF_Srxn1_Rmean_Andr32[1] = Rmean_Andr_Srxn1_36_032
        pDF_Srxn1_Rmean_Andr32[2] = Rmean_Andr_Srxn1_36_100
        pDF_Srxn1_Rmean_Andr32[3] = Rmean_Andr_Srxn1_36_316
        pDF_Srxn1_Rmean_Andr32[4] = Rmean_Andr_Srxn1_36_1000
        pDF_Srxn1_Rmean_Andr32[5] = Rmean_Andr_Srxn1_36_3162
        pDF_Srxn1_Rmean_Andr32[6] = Rmean_Andr_Srxn1_36_10000
        pDF_Srxn1_Rmean_Andr32 = np.array(pDF_Srxn1_Rmean_Andr32)

# 8. resmoothing
for h in range(1):
    def resmoothing(df):
            for i in range(np.shape(df)[1]):
                # df[:,i] = savgol_filter(df[:,i],1501,1)
                df[:,i] = savgol_filter(df[:,i],1501,5)
            return df
    if resmoothing_enabled == True:
            # pDF_Nrf2_Rmean_ES1 = resmoothing(pDF_Nrf2_Rmean_ES1)
            # pDF_Srxn1_Rmean_ES1 = resmoothing(pDF_Srxn1_Rmean_ES1)
            pDF_Nrf2_Rmean_SUL32 = resmoothing(pDF_Nrf2_Rmean_SUL32)
            pDF_Srxn1_Rmean_SUL32 = resmoothing(pDF_Srxn1_Rmean_SUL32)
            pDF_Nrf2_Rmean_CDDO32 = resmoothing(pDF_Nrf2_Rmean_CDDO32)
            pDF_Srxn1_Rmean_CDDO32 = resmoothing(pDF_Srxn1_Rmean_CDDO32)
            # pDF_Srxn1_Rmean_SUL_8_24_ME = resmoothing(pDF_Srxn1_Rmean_SUL_8_24_ME)
            # pDF_Nrf2_Rmean_SUL_8_24_ME = resmoothing(pDF_Nrf2_Rmean_SUL_8_24_ME)

# 9. Main training functions used
for h in range(1):
    # LSTMcv dose-Nrf2
    def trainXfunc11_NN1_convLSTM(traindoselist, valdoselist):
        # Prepare concentration vector scaler
        scaling_conclist = [0.1, 20]
        for i in range(2):
            scaling_conclist[i] = np.log10(scaling_conclist[i])
        scaling_conclist = np.expand_dims(scaling_conclist, axis=1)
        scaler.fit(scaling_conclist)

        # doselist: first group of doses is for training, second group of doses is for predictions. Groups = doses in ascending order.
        # concentrations are log scaled and an entire column of listt is filled with this value
        train = np.zeros((len(traindoselist), 2000, 2))
        for i in range(len(traindoselist)):
            conc = np.log10(traindoselist[i])
            conclist2 = []
            for j in range(2000):
                conclist2.append(conc)
            conclist2 = np.array(conclist2)
            conclist2 = np.expand_dims(conclist2, axis=1)
            conclist2 = scaler.transform(conclist2)
            train[i,:,0] = listt[:, 0]
            train[i,:,1] = conclist2[:, 0]
        test = np.zeros((len(valdoselist), 2000, 2))
        for i in range(len(valdoselist)):
            conc = np.log10(valdoselist[i])
            conclist2 = []
            for j in range(2000):
                conclist2.append(conc)
            conclist2 = np.array(conclist2)
            conclist2 = np.expand_dims(conclist2, axis=1)
            conclist2 = scaler.transform(conclist2)
            test[i,:, 0] = listt[:, 0]
            test[i,:, 1] = conclist2[:, 0]
        return train, test
    def trainYfunc11_NN1_convLSTM(df_Nrf2, traindoseindexlist, valdoseindexlist):
        traindata = {}
        for i in range(len(traindoseindexlist)):
            traindata['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, traindoseindexlist[i]]
        traindata = pd.DataFrame.from_dict(traindata)
        traindata = np.array(traindata)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(traindata)[1]):
            vector = traindata[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            print(np.shape(vector))
            traindata[:,i] = vector[:,0]
        valdata = {}
        for i in range(len(valdoseindexlist)):
            valdata['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, valdoseindexlist[i]]
        valdata = pd.DataFrame.from_dict(valdata)
        valdata = np.array(valdata)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(valdata)[1]):
            vector = valdata[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            print(np.shape(vector))
            valdata[:,i] = vector[:,0]

        trainy = np.zeros((len(traindoseindexlist), len(df_Nrf2),1))
        valy = np.zeros((len(valdoseindexlist), len(df_Nrf2),1))
        for i in range(len(traindoseindexlist)):
            trainy[i, :, 0] = traindata[:,i]
        for i in range(len(valdoseindexlist)):
            valy[i, :, 0] = valdata[:,i]
        return trainy, valy

    # LSTMtp dose-Nrf2
    def trainXfunc10_NN1_tp(traindoselist, valdoselist):
        # Prepare concentration vector scaler
        scaling_conclist = [0.1, 20]
        for i in range(2):
            scaling_conclist[i] = np.log10(scaling_conclist[i])
        scaling_conclist = np.expand_dims(scaling_conclist, axis=1)
        scaler.fit(scaling_conclist)

        # doselist: first group of doses is for training, second group of doses is for predictions. Groups = doses in ascending order.
        # concentrations are log scaled and an entire column of listt is filled with this value
        train = np.zeros((2000*len(traindoselist),1,2))
        for i in range(len(traindoselist)):
            conc = np.log10(traindoselist[i])
            conclist2 = []
            for j in range(2000):
                conclist2.append(conc)
            conclist2 = np.array(conclist2)
            conclist2 = np.expand_dims(conclist2, axis=1)
            conclist2 = scaler.transform(conclist2)
            train[(0+2000*i):(2000+2000*i),0,0] = listt[:,0]
            train[(0+2000*i):(2000+2000*i),0,1] = conclist2[:,0]
        test = np.zeros((2000*len(valdoselist),1,2))
        for i in range(len(valdoselist)):
            conc = np.log10(valdoselist[i])
            conclist2 = []
            for j in range(2000):
                conclist2.append(conc)
            conclist2 = np.array(conclist2)
            conclist2 = np.expand_dims(conclist2, axis=1)
            conclist2 = scaler.transform(conclist2)
            test[(0+2000*i):(2000+2000*i),0,0] = listt[:,0]
            test[(0+2000*i):(2000+2000*i),0,1] = conclist2[:,0]

        return train, test
    def trainYfunc10_NN1_tp(df_Nrf2, traindoseindexlist, valdoseindexlist):
        traindataNrf2 = {}
        for i in range(len(traindoseindexlist)):
            traindataNrf2['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, traindoseindexlist[i]]
        traindataNrf2 = pd.DataFrame.from_dict(traindataNrf2)
        traindataNrf2 = np.array(traindataNrf2)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(traindataNrf2)[1]):
            vector = traindataNrf2[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindataNrf2[:,i] = vector[:,0]
        traindataSrxn1 = {}
        for i in range(len(traindoseindexlist)):
            traindataSrxn1['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, traindoseindexlist[i]]
        traindataSrxn1 = pd.DataFrame.from_dict(traindataSrxn1)
        traindataSrxn1 = np.array(traindataSrxn1)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(traindataSrxn1)[1]):
            vector = traindataSrxn1[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindataSrxn1[:,i] = vector[:,0]
        # validation data
        valdataNrf2 = {}
        for i in range(len(valdoseindexlist)):
            valdataNrf2['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, valdoseindexlist[i]]
        valdataNrf2 = pd.DataFrame.from_dict(valdataNrf2)
        valdataNrf2 = np.array(valdataNrf2)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(valdataNrf2)[1]):
            vector = valdataNrf2[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            valdataNrf2[:,i] = vector[:,0]
        valdataSrxn1 = {}
        for i in range(len(valdoseindexlist)):
            valdataSrxn1['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, valdoseindexlist[i]]
        valdataSrxn1 = pd.DataFrame.from_dict(valdataSrxn1)
        valdataSrxn1 = np.array(valdataSrxn1)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(valdataSrxn1)[1]):
            vector = valdataSrxn1[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            valdataSrxn1[:,i] = vector[:,0]
        # Appending to training and validation data
        trainx = np.zeros((2000*np.shape(traindataNrf2)[1],1,1))
        for i in range(np.shape(traindataNrf2)[1]):
            trainx[int(2000*i):int(2000*(i+1)), 0, 0] = traindataNrf2[0:2000, i]
            # trainx[int(2000*i):int(2000*(i+1)), 0, 2] = traindataSrxn1[0:2000, i]
        valx = np.zeros((2000*np.shape(valdataNrf2)[1],1,1))
        for i in range(np.shape(valdataNrf2)[1]):
            print('check' + str(i+1) + '.3')
            valx[int(2000*i):int(2000*(i+1)), 0, 0] = valdataNrf2[0:2000, i]
            # valx[int(2000*i):int(2000*(i+1)), 0, 2] = valdataSrxn1[0:2000, i]
            print('check' + str(i+1) + '.4')
        return trainx, valx

    # LSTMcv Nrf2-Srxn1
    def trainXfunc2(df_Nrf2, df_Srxn1):
        traindata = {}
        for i in range(len(traindoseindexlist)):
            traindata['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, traindoseindexlist[i]]
        traindata = pd.DataFrame.from_dict(traindata)
        traindata = np.array(traindata)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(traindata)[1]):
            vector = traindata[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            print(np.shape(vector))
            traindata[:,i] = vector[:,0]
        print('check1')
        valdata = {}
        for i in range(len(valdoseindexlist)):
            valdata['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, valdoseindexlist[i]]
        valdata = pd.DataFrame.from_dict(valdata)
        valdata = np.array(valdata)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(valdata)[1]):
            vector = valdata[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            print(np.shape(vector))
            valdata[:,i] = vector[:,0]
        if Srxn1_as_input == False:
            print('check3')
            trainx = np.zeros((len(traindoseindexlist), len(df_Nrf2), 2))
            print('check5')
            valx = np.zeros((int(len(valdoseindexlist)), len(df_Nrf2), 2))
            print('check4')
        if Srxn1_as_input == True:
            trainx = np.zeros((int(len(traindoseindexlist)), int((len(df_Nrf2)), 3)))
            valx = np.zeros((int(len(valdoseindexlist)), int((len(df_Nrf2)), 3)))
        time = listt[:,0]
        print('check2')
        for i in range(len(traindoseindexlist)):
            trainx[i, :, 0] = time
            trainx[i, :, 1] = traindata[:,i]
        for i in range(len(valdoseindexlist)):
            valx[i, :, 0] = time
            valx[i, :, 1] = valdata[:,i]
        return trainx, valx
    def trainYfunc2(df_Srxn1):
        traindata = {}
        for i in range(len(traindoseindexlist)):
            traindata['trainy_Nrf2_{0}'.format(i)] = df_Srxn1[:, traindoseindexlist[i]]
        traindata = pd.DataFrame.from_dict(traindata)
        traindata = np.array(traindata)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(traindata)[1]):
            vector = traindata[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            print(np.shape(vector))
            traindata[:,i] = vector[:,0]
        valdata = {}
        for i in range(len(valdoseindexlist)):
            valdata['trainy_Nrf2_{0}'.format(i)] = df_Srxn1[:, valdoseindexlist[i]]
        valdata = pd.DataFrame.from_dict(valdata)
        valdata = np.array(valdata)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(valdata)[1]):
            vector = valdata[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            print(np.shape(vector))
            valdata[:,i] = vector[:,0]
        print('check1')
        print('check6')

        if Srxn1_as_input == False:
            print('check3')
            trainy = np.zeros((len(traindoseindexlist), len(df_Srxn1), 1))
            print('check5')
            valy = np.zeros((int(len(valdoseindexlist)), len(df_Srxn1), 1))
            print('check4')
        time = listt[:,0]
        print('check2')
        for i in range(len(traindoseindexlist)):
            trainy[i, :, 0] = traindata[:,i]
        for i in range(len(valdoseindexlist)):
            valy[i, :, 0] = valdata[:,i]
        return trainy, valy

    # LSTMtp Nrf2-Srxn1
    def trainXfunc9_Nrf2_to_Srxn1(df_Nrf2, df_Srxn1, traindoseindexlist, valdoseindexlist):
        traindataNrf2 = {}
        for i in range(len(traindoseindexlist)):
            traindataNrf2['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, traindoseindexlist[i]]
        traindataNrf2 = pd.DataFrame.from_dict(traindataNrf2)
        traindataNrf2 = np.array(traindataNrf2)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(traindataNrf2)[1]):
            vector = traindataNrf2[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindataNrf2[:,i] = vector[:,0]
        traindataSrxn1 = {}
        for i in range(len(traindoseindexlist)):
            traindataSrxn1['trainy_Nrf2_{0}'.format(i)] = df_Srxn1[:, traindoseindexlist[i]]
        traindataSrxn1 = pd.DataFrame.from_dict(traindataSrxn1)
        traindataSrxn1 = np.array(traindataSrxn1)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(traindataSrxn1)[1]):
            vector = traindataSrxn1[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindataSrxn1[:,i] = vector[:,0]
        # validation data
        valdataNrf2 = {}
        for i in range(len(valdoseindexlist)):
            valdataNrf2['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, valdoseindexlist[i]]
        valdataNrf2 = pd.DataFrame.from_dict(valdataNrf2)
        valdataNrf2 = np.array(valdataNrf2)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(valdataNrf2)[1]):
            vector = valdataNrf2[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            valdataNrf2[:,i] = vector[:,0]
        valdataSrxn1 = {}
        for i in range(len(valdoseindexlist)):
            valdataSrxn1['trainy_Nrf2_{0}'.format(i)] = df_Srxn1[:, valdoseindexlist[i]]
        valdataSrxn1 = pd.DataFrame.from_dict(valdataSrxn1)
        valdataSrxn1 = np.array(valdataSrxn1)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(valdataSrxn1)[1]):
            vector = valdataSrxn1[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            valdataSrxn1[:,i] = vector[:,0]
        # Appending to training and validation data
        trainx = np.zeros((2000*np.shape(traindataNrf2)[1],1,2))
        for i in range(np.shape(traindataNrf2)[1]):
            trainx[int(2000*i):int(2000*(i+1)), 0, 0] = listt[:, 0]
            trainx[int(2000*i):int(2000*(i+1)), 0, 1] = traindataNrf2[0:2000, i]
            # trainx[int(2000*i):int(2000*(i+1)), 0, 2] = traindataSrxn1[0:2000, i]
        valx = np.zeros((2000*np.shape(valdataNrf2)[1],1,2))
        for i in range(np.shape(valdataNrf2)[1]):
            print('check' + str(i+1) + '.3')
            valx[int(2000*i):int(2000*(i+1)), 0, 0] = listt[:, 0]
            valx[int(2000*i):int(2000*(i+1)), 0, 1] = valdataNrf2[0:2000, i]
            # valx[int(2000*i):int(2000*(i+1)), 0, 2] = valdataSrxn1[0:2000, i]
            print('check' + str(i+1) + '.4')
        return trainx, valx
    def trainYfunc9_Nrf2_to_Srxn1(df_Srxn1, traindoseindexlist, valdoseindexlist):
        traindata = {}
        for i in range(len(traindoseindexlist)):
            traindata['trainy_Srxn1_{0}'.format(i)] = df_Srxn1[:, traindoseindexlist[i]]
        traindata = pd.DataFrame.from_dict(traindata)
        traindata = np.array(traindata)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(traindata)[1]):
            vector = traindata[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindata[:,i] = vector[:,0]
        valdata = {}
        for i in range(len(valdoseindexlist)):
            valdata['trainy_Srxn1_{0}'.format(i)] = df_Srxn1[:, valdoseindexlist[i]]
        valdata = pd.DataFrame.from_dict(valdata)
        valdata = np.array(valdata)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(valdata)[1]):
            vector = valdata[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            valdata[:,i] = vector[:,0]
        trainy = np.zeros((2000*np.shape(traindata)[1],1,1))
        # if training_set == 1:
        #     diffscalerlist = np.array([-0.001,0.005])
        # if training_set == 2:
        #     diffscalerlist = np.array([-0.001,0.020])
        # diffscalerlist = diffscalerlist.reshape(2,1)
        # diffscaler = scaler.fit(diffscalerlist)
        for i in range(np.shape(traindata)[1]):
            # trainy[int(2000*i):int(2000*(i+1)), 0, 0] = listt[:-1, 0]
            # trainy[int(2000*i):int(2000*(i+1)), 0, 0] = traindata[0:2000, i]
            print('check' + str(i+1) + '.1')
            diff = traindata[: , i]
            # diff = np.diff(diff)
            # diff = diff.reshape(len(diff),1)
            # diff = diffscaler.transform(diff)
            trainy[int(2000*i):int(2000*(i+1)), 0, 0] = diff[:]
            print('check' + str(i+1) + '.2')
        valy = np.zeros((2000*np.shape(valdata)[1],1,1))
        for i in range(np.shape(valdata)[1]):
            # valy[int(2000*i):int(2000*(i+1)), 0, 0] = listt[:-1, 0]
            # valy[int(2000*i):int(2000*(i+1)), 0, 0] = valdata[0:2000, i]
            diff = valdata[: , i]
            # diff = np.diff(diff)
            # diff = diff.reshape(len(diff), 1)
            # diff = diffscaler.transform(diff)
            valy[int(2000*i):int(2000*(i+1)), 0, 0] = diff[:]
            print('check' + str(i+1) + '.4')
        return trainy, valy

    # LSTMdtp Nrf2-Srxn1
    # Default version:
    def trainXfunc6_Nrf2_Srxn1_to_dSrxn1(df_Nrf2, df_Srxn1, trainindex, valindex):
        traindataNrf2 = {}
        for i in range(len(traindoseindexlist)):
            traindataNrf2['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, traindoseindexlist[i]]
        traindataNrf2 = pd.DataFrame.from_dict(traindataNrf2)
        traindataNrf2 = np.array(traindataNrf2)
        scaler.fit(datascalerlistNrf2)
        print('check1')
        for i in range(np.shape(traindataNrf2)[1]):
            vector = traindataNrf2[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindataNrf2[:,i] = vector[:,0]
        traindataSrxn1 = {}
        for i in range(len(traindoseindexlist)):
            traindataSrxn1['trainy_Nrf2_{0}'.format(i)] = df_Srxn1[:, traindoseindexlist[i]]
        traindataSrxn1 = pd.DataFrame.from_dict(traindataSrxn1)
        traindataSrxn1 = np.array(traindataSrxn1)
        # scaler.fit(datascalerlistNrf2)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(traindataSrxn1)[1]):
            vector = traindataSrxn1[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindataSrxn1[:,i] = vector[:,0]

        # validation data
        valdataNrf2 = {}
        for i in range(len(valdoseindexlist)):
            valdataNrf2['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, valdoseindexlist[i]]
        valdataNrf2 = pd.DataFrame.from_dict(valdataNrf2)
        valdataNrf2 = np.array(valdataNrf2)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(valdataNrf2)[1]):
            vector = valdataNrf2[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            valdataNrf2[:,i] = vector[:,0]
        valdataSrxn1 = {}
        for i in range(len(valdoseindexlist)):
            valdataSrxn1['trainy_Nrf2_{0}'.format(i)] = df_Srxn1[:, valdoseindexlist[i]]
        valdataSrxn1 = pd.DataFrame.from_dict(valdataSrxn1)
        valdataSrxn1 = np.array(valdataSrxn1)
        # scaler.fit(datascalerlistNrf2)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(valdataSrxn1)[1]):
            vector = valdataSrxn1[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            valdataSrxn1[:,i] = vector[:,0]

        # Appending to training and validation data
        trainx = np.zeros((1999*np.shape(traindataNrf2)[1],1,3))
        for i in range(np.shape(traindataNrf2)[1]):
            trainx[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            trainx[int(1999*i):int(1999*(i+1)), 0, 1] = traindataNrf2[0:1999, i]
            trainx[int(1999*i):int(1999*(i+1)), 0, 2] = traindataSrxn1[0:1999, i]
        valx = np.zeros((1999*np.shape(valdataNrf2)[1],1,3))
        for i in range(np.shape(valdataNrf2)[1]):
            print('check' + str(i+1) + '.3')
            valx[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            valx[int(1999*i):int(1999*(i+1)), 0, 1] = valdataNrf2[0:1999, i]
            valx[int(1999*i):int(1999*(i+1)), 0, 2] = valdataSrxn1[0:1999, i]
            print('check' + str(i+1) + '.4')
        return trainx, valx
    # Doesn't include time as an input, just Nrf2 and Srxn1:
    def trainXfunc6_Nrf2_Srxn1_to_dSrxn1_notime(df_Nrf2, df_Srxn1, trainindex, valindex):
        traindataNrf2 = {}
        for i in range(len(traindoseindexlist)):
            traindataNrf2['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, traindoseindexlist[i]]
        traindataNrf2 = pd.DataFrame.from_dict(traindataNrf2)
        traindataNrf2 = np.array(traindataNrf2)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(traindataNrf2)[1]):
            vector = traindataNrf2[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindataNrf2[:,i] = vector[:,0]
        traindataSrxn1 = {}
        for i in range(len(traindoseindexlist)):
            traindataSrxn1['trainy_Nrf2_{0}'.format(i)] = df_Srxn1[:, traindoseindexlist[i]]
        traindataSrxn1 = pd.DataFrame.from_dict(traindataSrxn1)
        traindataSrxn1 = np.array(traindataSrxn1)
        # scaler.fit(datascalerlistNrf2)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(traindataSrxn1)[1]):
            vector = traindataSrxn1[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindataSrxn1[:,i] = vector[:,0]

        # validation data
        valdataNrf2 = {}
        for i in range(len(valdoseindexlist)):
            valdataNrf2['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, valdoseindexlist[i]]
        valdataNrf2 = pd.DataFrame.from_dict(valdataNrf2)
        valdataNrf2 = np.array(valdataNrf2)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(valdataNrf2)[1]):
            vector = valdataNrf2[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            valdataNrf2[:,i] = vector[:,0]
        valdataSrxn1 = {}
        for i in range(len(valdoseindexlist)):
            valdataSrxn1['trainy_Nrf2_{0}'.format(i)] = df_Srxn1[:, valdoseindexlist[i]]
        valdataSrxn1 = pd.DataFrame.from_dict(valdataSrxn1)
        valdataSrxn1 = np.array(valdataSrxn1)
        # scaler.fit(datascalerlistNrf2)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(valdataSrxn1)[1]):
            vector = valdataSrxn1[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            valdataSrxn1[:,i] = vector[:,0]

        # Appending to training and validation data
        trainx = np.zeros((1999*np.shape(traindataNrf2)[1],1,2))
        for i in range(np.shape(traindataNrf2)[1]):
            # trainx[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            trainx[int(1999*i):int(1999*(i+1)), 0, 0] = traindataNrf2[0:1999, i]
            trainx[int(1999*i):int(1999*(i+1)), 0, 1] = traindataSrxn1[0:1999, i]
        valx = np.zeros((1999*np.shape(valdataNrf2)[1],1,2))
        for i in range(np.shape(valdataNrf2)[1]):
            print('check' + str(i+1) + '.3')
            # valx[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            valx[int(1999*i):int(1999*(i+1)), 0, 0] = valdataNrf2[0:1999, i]
            valx[int(1999*i):int(1999*(i+1)), 0, 1] = valdataSrxn1[0:1999, i]
            print('check' + str(i+1) + '.4')
        return trainx, valx
    # Has an all zeros vector in the place of Srxn1 in the validation data, used to test the influence of Srxn1 (like a knockdown condition):
    def trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(df_Nrf2, df_Srxn1):
        traindataNrf2 = {}
        for i in range(len(traindoselist)):
            traindataNrf2['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, traindoseindexlist[i]]
        traindataNrf2 = pd.DataFrame.from_dict(traindataNrf2)
        traindataNrf2 = np.array(traindataNrf2)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(traindataNrf2)[1]):
            vector = traindataNrf2[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindataNrf2[:,i] = vector[:,0]
        traindataSrxn1 = {}
        for i in range(len(traindoselist)):
            traindataSrxn1['trainy_Nrf2_{0}'.format(i)] = df_Srxn1[:, traindoseindexlist[i]]
        traindataSrxn1 = pd.DataFrame.from_dict(traindataSrxn1)
        traindataSrxn1 = np.array(traindataSrxn1)
        # scaler.fit(datascalerlistNrf2)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(traindataSrxn1)[1]):
            vector = traindataSrxn1[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindataSrxn1[:,i] = vector[:,0]
        # validation data
        valdataNrf2 = {}
        for i in range(len(valdoseindexlist)):
            valdataNrf2['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, valdoseindexlist[i]]
        valdataNrf2 = pd.DataFrame.from_dict(valdataNrf2)
        valdataNrf2 = np.array(valdataNrf2)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(valdataNrf2)[1]):
            vector = valdataNrf2[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            valdataNrf2[:,i] = vector[:,0]
        valdataSrxn1 = {}
        for i in range(len(valdoseindexlist)):
            valdataSrxn1['trainy_Nrf2_{0}'.format(i)] = df_Srxn1[:, valdoseindexlist[i]]
        valdataSrxn1 = pd.DataFrame.from_dict(valdataSrxn1)
        valdataSrxn1 = np.array(valdataSrxn1)
        # scaler.fit(datascalerlistNrf2)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(valdataSrxn1)[1]):
            vector = valdataSrxn1[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            valdataSrxn1[:,i] = vector[:,0]
        # Appending to training and validation data
        trainx = np.zeros((1999*np.shape(traindataNrf2)[1],1,3))
        for i in range(np.shape(traindataNrf2)[1]):
            trainx[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            trainx[int(1999*i):int(1999*(i+1)), 0, 1] = traindataNrf2[0:1999, i]
            trainx[int(1999*i):int(1999*(i+1)), 0, 2] = traindataSrxn1[0:1999, i]
        valx = np.zeros((1999*np.shape(valdataNrf2)[1],1,3))
        for i in range(np.shape(valdataNrf2)[1]):
            print('check' + str(i+1) + '.3')
            valx[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            valx[int(1999*i):int(1999*(i+1)), 0, 1] = valdataNrf2[0:1999, i]
            # valx[int(1999*i):int(1999*(i+1)), 0, 2] = valdataSrxn1[0:1999, i]
            print('check' + str(i+1) + '.4')
        return trainx, valx

    # Unsmoothed dSrxn1:
    def trainYfunc6_Nrf2_Srxn1_to_dSrxn1(df_Srxn1):
        traindata = {}
        for i in range(len(traindoseindexlist)):
            traindata['trainy_Srxn1_{0}'.format(i)] = df_Srxn1[:, traindoseindexlist[i]]
        traindata = pd.DataFrame.from_dict(traindata)
        traindata = np.array(traindata)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(traindata)[1]):
            vector = traindata[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindata[:,i] = vector[:,0]
        valdata = {}
        for i in range(len(valdoseindexlist)):
            valdata['trainy_Srxn1_{0}'.format(i)] = df_Srxn1[:, valdoseindexlist[i]]
        valdata = pd.DataFrame.from_dict(valdata)
        valdata = np.array(valdata)
        # scaler.fit(datascalerlistSrxn1)
        # for i in range(np.shape(valdata)[1]):
        #     vector = valdata[:,i]
        #     vector = vector.reshape(len(vector),1)
        #     vector = scaler.transform(vector)
        #     valdata[:,i] = vector[:,0]
        trainy = np.zeros((1999*np.shape(traindata)[1],1,1))
        # if training_set == 1:
        #     diffscalerlist = np.array([-0.001,0.005])
        # if training_set == 2:
        #     diffscalerlist = np.array([-0.001,0.020])
        # diffscalerlist = diffscalerlist.reshape(2,1)
        diffscaler = scaler.fit(diffscalerlist)
        for i in range(np.shape(traindata)[1]):
            # trainy[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            # trainy[int(1999*i):int(1999*(i+1)), 0, 0] = traindata[0:1999, i]
            print('check' + str(i+1) + '.1')
            diff = traindata[: , i]
            diff = np.diff(diff)
            diff = diff.reshape(len(diff),1)
            diff = diffscaler.transform(diff)
            trainy[int(1999*i):int(1999*(i+1)), 0, 0] = diff[:,0]
            print('check' + str(i+1) + '.2')
        valy = np.zeros((1999*np.shape(valdata)[1],1,1))
        for i in range(np.shape(valdata)[1]):
            # valy[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            # valy[int(1999*i):int(1999*(i+1)), 0, 0] = valdata[0:1999, i]
            diff = valdata[: , i]
            diff = np.diff(diff)
            diff = diff.reshape(len(diff), 1)
            diff = diffscaler.transform(diff)
            valy[int(1999*i):int(1999*(i+1)), 0, 0] = diff[:,0]
            print('check' + str(i+1) + '.4')
        return trainy, valy
    # Smoothed dSrxn1 (used for final results):
    def trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(df_Srxn1, wlength, poly_order):
        traindata = {}
        for i in range(len(traindoseindexlist)):
            traindata['trainy_Srxn1_{0}'.format(i)] = df_Srxn1[:, traindoseindexlist[i]]
        traindata = pd.DataFrame.from_dict(traindata)
        traindata = np.array(traindata)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(traindata)[1]):
            vector = traindata[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindata[:,i] = vector[:,0]
        valdata = {}
        for i in range(len(valdoseindexlist)):
            valdata['trainy_Srxn1_{0}'.format(i)] = df_Srxn1[:, valdoseindexlist[i]]
        valdata = pd.DataFrame.from_dict(valdata)
        valdata = np.array(valdata)
        # scaler.fit(datascalerlistSrxn1)
        # for i in range(np.shape(valdata)[1]):
        #     vector = valdata[:,i]
        #     vector = vector.reshape(len(vector),1)
        #     vector = scaler.transform(vector)
        #     valdata[:,i] = vector[:,0]
        trainy = np.zeros((1999*np.shape(traindata)[1],1,1))
        # if training_set == 1:
        #     diffscalerlist = np.array([-0.001,0.005])
        # if training_set == 2:
        #     diffscalerlist = np.array([-0.001,0.020])
        # diffscalerlist = diffscalerlist.reshape(2,1)
        diffscaler = scaler.fit(diffscalerlist)
        for i in range(np.shape(traindata)[1]):
            # trainy[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            # trainy[int(1999*i):int(1999*(i+1)), 0, 0] = traindata[0:1999, i]
            print('check' + str(i+1) + '.1')
            diff = traindata[: , i]
            diff = np.diff(diff)
            # plt.plot(diff, label='before smoothing')
            diff = savgol_filter(diff, wlength, poly_order)
            # diff = savgol_filter(diff, 1501, 5)
            # plt.plot(diff, label='after smoothing')
            diff = diff.reshape(len(diff),1)
            diff = diffscaler.transform(diff)
            trainy[int(1999*i):int(1999*(i+1)), 0, 0] = diff[:,0]
            print('check' + str(i+1) + '.2')
        valy = np.zeros((1999*np.shape(valdata)[1],1,1))
        for i in range(np.shape(valdata)[1]):
            # valy[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            # valy[int(1999*i):int(1999*(i+1)), 0, 0] = valdata[0:1999, i]
            diff = valdata[: , i]
            diff = np.diff(diff)
            diff = diff.reshape(len(diff), 1)
            diff = diffscaler.transform(diff)
            valy[int(1999*i):int(1999*(i+1)), 0, 0] = diff[:,0]
            print('check' + str(i+1) + '.4')
        return trainy, valy

# 10 Experimental training functions
for h in range(1):
    # Training data functions
    def trainXfunc_traintest(df_Nrf2, df_Srxn1):
        scaler.fit(datascalerlistNrf2)
        if derivative_training == True:
            trainbase = pd.DataFrame(listt[0:len(listt)-1,0])
        else:
            trainbase = pd.DataFrame(listt[:,0])

        data = {}
        for i in range(len(traindoselist)):
            data['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, traindoseindexlist[i]]
        data = pd.DataFrame.from_dict(data)
        data = np.array(data)
        if derivative_training == True:
            train_Xdose = data[0:len(data)-1,0]
        else:
            train_Xdose = data[:, 0]
        # step-wise replacement of the base dose by different doses. Every xth datapoint from the base dose vector is replaced by that of a different dose vector. The length of the 'jumps', or space between the datapoints that are replaced by the same dose vector is determined by the amount of doses used.
        # Due to this jump length being represented as a multiplication of i, the length of the range has been divided by this multiplication to avoid i falling outside of the range of the loop.
        for i in range(int(len(train_Xdose) / len(traindoselist))):
            train_Xdose[i * len(traindoselist) + 1] = data[i * len(traindoselist) + 1, 1]
            if len(traindoselist) >= 3:
                train_Xdose[i * len(traindoselist) + 2] = data[i * len(traindoselist) + 2, 2]
            if len(traindoselist) >= 4:
                train_Xdose[i * len(traindoselist) + 3] = data[i * len(traindoselist) + 3, 3]
            if len(traindoselist) >= 5:
                train_Xdose[i * len(traindoselist) + 4] = data[i * len(traindoselist) + 4, 4]
            if len(traindoselist) >= 6:
                train_Xdose[i * len(traindoselist) + 5] = data[i * len(traindoselist) + 5, 5]
            if len(traindoselist) >= 7:
                train_Xdose[i * len(traindoselist) + 5] = data[i * len(traindoselist) + 6, 6]
        train_Xdose = np.array(train_Xdose)
        train_Xdose = train_Xdose.reshape(len(train_Xdose),1)
        train_Xdose = scaler.transform(train_Xdose)
        trainbase[1] = train_Xdose
        print('check Nrf2 train')

        if Srxn1_as_input == True:
            scaler.fit(datascalerlistSrxn1)
            # Srxn1
            # if derivative_training == True:
            #     trainbase = pd.DataFrame(listt[0:len(listt)-1,0])
            # else:
            #     trainbase = pd.DataFrame(listt[:,0])
            data = {}
            for i in range(len(traindoselist)):
                data['trainy_Srxn1_{0}'.format(i)] = df_Srxn1[:, traindoseindexlist[i]]
            data = pd.DataFrame.from_dict(data)
            data = np.array(data)
            if derivative_training == True:
                train_Xdose = data[0:len(data)-1,0]
            else:
                train_Xdose = data[:, 0]
            # step-wise replacement of the base dose by different doses. Every xth datapoint from the base dose vector is replaced by that of a different dose vector. The length of the 'jumps', or space between the datapoints that are replaced by the same dose vector is determined by the amount of doses used.
            # Due to this jump length being represented as a multiplication of i, the length of the range has been divided by this multiplication to avoid i falling outside of the range of the loop.
            for i in range(int(len(train_Xdose) / len(traindoselist))):
                train_Xdose[i * len(traindoselist) + 1] = data[i * len(traindoselist) + 1, 1]
                if len(traindoselist) >= 3:
                    train_Xdose[i * len(traindoselist) + 2] = data[i * len(traindoselist) + 2, 2]
                if len(traindoselist) >= 4:
                    train_Xdose[i * len(traindoselist) + 3] = data[i * len(traindoselist) + 3, 3]
                if len(traindoselist) >= 5:
                    train_Xdose[i * len(traindoselist) + 4] = data[i * len(traindoselist) + 4, 4]
                if len(traindoselist) >= 6:
                    train_Xdose[i * len(traindoselist) + 5] = data[i * len(traindoselist) + 5, 5]
                if len(traindoselist) >= 7:
                    train_Xdose[i * len(traindoselist) + 5] = data[i * len(traindoselist) + 6, 6]
            train_Xdose = np.array(train_Xdose)
            train_Xdose = train_Xdose.reshape(len(train_Xdose), 1)
            train_Xdose = scaler.transform(train_Xdose)
            trainbase[2] = train_Xdose

            print(np.shape(train_Xdose))
            print('check Srxn1 train')
            trainbase = np.array(trainbase)
            print(np.shape(trainbase))
        else:
            trainbase = np.array(trainbase)

        def test():
            scaler.fit(datascalerlistNrf2)
            if derivative_training == True:
                valbase = pd.DataFrame(listt[0:len(listt) - 1, 0])
            else:
                valbase = pd.DataFrame(listt[:, 0])
            print('check Nrf2 val 1')
            data = {}
            for i in range(len(valdoselist)):
                data['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, valdoseindexlist[i]]
            data = pd.DataFrame.from_dict(data)
            data = np.array(data)
            if derivative_training == True:
                train_Xdose = data[0:len(data) - 1, 0]
            else:
                train_Xdose = data[:, 0]
            # step-wise replacement of the base dose by different doses. Every xth datapoint from the base dose vector is replaced by that of a different dose vector. The length of the 'jumps', or space between the datapoints that are replaced by the same dose vector is determined by the amount of doses used.
            # Due to this jump length being represented as a multiplication of i, the length of the range has been divided by this multiplication to avoid i falling outside of the range of the loop.
            if len(valdoselist) > 1:
                for i in range(int(len(train_Xdose) / len(valdoselist))):
                    train_Xdose[i * len(valdoselist) + 1] = data[i * len(valdoselist) + 1, 1]
                    if len(valdoselist) >= 3:
                        train_Xdose[i * len(valdoselist) + 2] = data[i * len(valdoselist) + 2, 2]
                    if len(valdoselist) >= 4:
                        train_Xdose[i * len(valdoselist) + 3] = data[i * len(valdoselist) + 3, 3]
                    if len(valdoselist) >= 5:
                        train_Xdose[i * len(valdoselist) + 4] = data[i * len(valdoselist) + 4, 4]
                    if len(valdoselist) >= 6:
                        train_Xdose[i * len(valdoselist) + 5] = data[i * len(valdoselist) + 5, 5]
                    if len(valdoselist) >= 7:
                        train_Xdose[i * len(valdoselist) + 5] = data[i * len(valdoselist) + 6, 6]
            train_Xdose = np.array(train_Xdose)
            train_Xdose = train_Xdose.reshape(len(train_Xdose), 1)
            train_Xdose = scaler.transform(train_Xdose)
            valbase[1] = train_Xdose
            print('check Nrf2 val 2')


            if Srxn1_as_input == True:
                scaler.fit(datascalerlistSrxn1)
                data = {}
                for i in range(len(valdoselist)):
                    data['trainy_Nrf2_{0}'.format(i)] = df_Srxn1[:, valdoseindexlist[i]]
                data = pd.DataFrame.from_dict(data)
                data = np.array(data)
                if derivative_training == True:
                    train_Xdose = data[0:len(data) - 1, 0]
                else:
                    train_Xdose = data[:, 0]
                # step-wise replacement of the base dose by different doses. Every xth datapoint from the base dose vector is replaced by that of a different dose vector. The length of the 'jumps', or space between the datapoints that are replaced by the same dose vector is determined by the amount of doses used.
                # Due to this jump length being represented as a multiplication of i, the length of the range has been divided by this multiplication to avoid i falling outside of the range of the loop.
                if len(valdoselist) > 1:
                    for i in range(int(len(train_Xdose) / len(valdoselist))):
                        train_Xdose[i * len(valdoselist) + 1] = data[i * len(valdoselist) + 1, 1]
                        if len(valdoselist) >= 3:
                            train_Xdose[i * len(valdoselist) + 2] = data[i * len(valdoselist) + 2, 2]
                        if len(valdoselist) >= 4:
                            train_Xdose[i * len(valdoselist) + 3] = data[i * len(valdoselist) + 3, 3]
                        if len(valdoselist) >= 5:
                            train_Xdose[i * len(valdoselist) + 4] = data[i * len(valdoselist) + 4, 4]
                        if len(valdoselist) >= 6:
                            train_Xdose[i * len(valdoselist) + 5] = data[i * len(valdoselist) + 5, 5]
                        if len(valdoselist) >= 7:
                            train_Xdose[i * len(valdoselist) + 5] = data[i * len(valdoselist) + 6, 6]
                train_Xdose = np.array(train_Xdose)
                train_Xdose = train_Xdose.reshape(len(train_Xdose), 1)
                train_Xdose = scaler.transform(train_Xdose)
                valbase[2] = train_Xdose
                print('check Srxn1 val')
                valbase = np.array(valbase)
            else:
                valbase = np.array(valbase)
            return valbase
        valbase = test()
        return trainbase, valbase
    def trainYfunc_traintest(df):
        scaler.fit(datascalerlistSrxn1)
        if derivative_training == True:
            df = np.diff(df)
        data = {}
        for i in range(len(traindoselist)):
            data['trainy_Nrf2_{0}'.format(i)] = df[:, traindoseindexlist[i]]
        data = pd.DataFrame.from_dict(data)
        data = np.array(data)
        train_Ydose = data[:, 0]
        # step-wise replacement of the base dose by different doses. Every xth datapoint from the base dose vector is replaced by that of a different dose vector. The length of the 'jumps', or space between the datapoints that are replaced by the same dose vector is determined by the amount of doses used.
        # Due to this jump length being represented as a multiplication of i, the length of the range has been divided by this multiplication to avoid i falling outside of the range of the loop.
        for i in range(int(len(train_Ydose) / len(traindoselist))):
            train_Ydose[i * len(traindoselist) + 1] = data[i * len(traindoselist) + 1, 1]
            if len(traindoselist) >= 3:
                train_Ydose[i * len(traindoselist) + 2] = data[i * len(traindoselist) + 2, 2]
            if len(traindoselist) >= 4:
                train_Ydose[i * len(traindoselist) + 3] = data[i * len(traindoselist) + 3, 3]
            if len(traindoselist) >= 5:
                train_Ydose[i * len(traindoselist) + 4] = data[i * len(traindoselist) + 4, 4]
            if len(traindoselist) >= 6:
                train_Ydose[i * len(traindoselist) + 5] = data[i * len(traindoselist) + 5, 5]
            if len(traindoselist) >= 7:
                train_Ydose[i * len(traindoselist) + 5] = data[i * len(traindoselist) + 6, 6]
        trainbase = np.array(train_Ydose)
        trainbase = trainbase.reshape(len(trainbase),1)
        trainbase = scaler.transform(trainbase)

        print('check1')
        def test():
            data = {}
            for i in range(len(valdoselist)):
                data['trainy_Nrf2_{0}'.format(i)] = df[:, valdoseindexlist[i]]
            data = pd.DataFrame.from_dict(data)
            data = np.array(data)
            train_Ydose = data[:, 0]
            print('check2')
            # step-wise replacement of the base dose by different doses. Every xth datapoint from the base dose vector is replaced by that of a different dose vector. The length of the 'jumps', or space between the datapoints that are replaced by the same dose vector is determined by the amount of doses used.
            # Due to this jump length being represented as a multiplication of i, the length of the range has been divided by this multiplication to avoid i falling outside of the range of the loop.
            if len(valdoselist) > 1:
                for i in range(int(len(train_Ydose) / len(valdoselist))):
                    train_Ydose[i * len(valdoselist) + 1] = data[i * len(valdoselist) + 1, 1]
                    if len(valdoselist) >= 3:
                        train_Ydose[i * len(valdoselist) + 2] = data[i * len(valdoselist) + 2, 2]
                    if len(valdoselist) >= 4:
                        train_Ydose[i * len(valdoselist) + 3] = data[i * len(valdoselist) + 3, 3]
                    if len(valdoselist) >= 5:
                        train_Ydose[i * len(valdoselist) + 4] = data[i * len(valdoselist) + 4, 4]
                    if len(valdoselist) >= 6:
                        train_Ydose[i * len(valdoselist) + 5] = data[i * len(valdoselist) + 5, 5]
                    if len(valdoselist) >= 7:
                        train_Ydose[i * len(valdoselist) + 5] = data[i * len(valdoselist) + 6, 6]
            valbase = np.array(train_Ydose)
            valbase = valbase.reshape(len(valbase),1)
            valbase = scaler.transform(valbase)
            print('check3')
            return valbase
        valbase = test()
        return trainbase, valbase

        #
        # x = np.zeros((2000, 1, 7))
        # np.shape(x)
        # for i in range(np.shape(pDF_Nrf2_Rmean_36)[1]):
        #     y = pDF_Nrf2_Rmean_36[:, i]
        #     y = np.array(y)
        #     y = y.reshape(len(y), 1)
        #     x[:, :, i] = y
        # np.shape(x)f

    # Attempts at combining the conventional LSTM approach with varying shapes of input data
    # It was hypothesized that the LSTMcv approach overfitted on the training data as the training data all had the same initial time points
    # By cutting up the data into various frames this problem may be solved, yet more testing was needed
    # Dissected training (cuts up the data into even pieces)
    def trainXfunc3(df_Nrf2, df_Srxn1):
        traindata = {}
        for i in range(len(traindoselist)):
            traindata['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, traindoseindexlist[i]]
        traindata = pd.DataFrame.from_dict(traindata)
        traindata = np.array(traindata)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(traindata)[1]):
            vector = traindata[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            print(np.shape(vector))
            traindata[:,i] = vector[:,0]
        print('check1')
        valdata = {}
        for i in range(len(valdoseindexlist)):
            valdata['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, valdoseindexlist[i]]
        valdata = pd.DataFrame.from_dict(valdata)
        valdata = np.array(valdata)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(valdata)[1]):
            vector = valdata[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            print(np.shape(vector))
            valdata[:,i] = vector[:,0]
        if Srxn1_as_input == False:
            print('check3')
            trainx = np.zeros((len(traindoseindexlist), len(df_Nrf2), 2))
            print('check5')
            valx = np.zeros((int(len(valdoseindexlist)), len(df_Nrf2), 2))
            print('check4')
        if Srxn1_as_input == True:
            trainx = np.zeros((int(len(traindoseindexlist)), int((len(df_Nrf2)), 3)))
            valx = np.zeros((int(len(valdoseindexlist)), int((len(df_Nrf2)), 3)))
        time = listt[:,0]
        print('check2')
        for i in range(len(traindoseindexlist)):
            trainx[i, :, 0] = time
            trainx[i, :, 1] = traindata[:,i]
        for i in range(len(valdoseindexlist)):
            valx[i, :, 0] = time
            valx[i, :, 1] = valdata[:,i]

        wl = section_length
        mf = multiplication_factor
        traindoses = len(traindoseindexlist)
        iters = int(len(traindata) / mf)

        trainx = np.zeros((iters*traindoses, wl, 2))
        for i in range(traindoses):
            for j in range(iters):
                trainx[j+i*iters, :, 0] = time[(j*mf):((j*mf)+wl)]
                trainx[j+i*iters, :, 1] = traindata[(j*mf):((j*mf)+wl),i]

        valdoses = len(valdoseindexlist)
        itersval = int(len(valdata)/ mf)

        valx = np.zeros((itersval*valdoses, wl, 2))
        for i in range(valdoses):
            for j in range(itersval):
                valx[j+i*itersval, :, 0] = time[(j*mf):((j*mf)+wl)]
                valx[j+i*itersval, :, 1] = traindata[(j*mf):((j*mf)+wl),i]


        return trainx, valx
    def trainYfunc3(df_Srxn1):
        traindata = {}
        for i in range(len(traindoseindexlist)):
            traindata['trainy_Nrf2_{0}'.format(i)] = df_Srxn1[:, traindoseindexlist[i]]
        traindata = pd.DataFrame.from_dict(traindata)
        traindata = np.array(traindata)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(traindata)[1]):
            vector = traindata[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            print(np.shape(vector))
            traindata[:,i] = vector[:,0]
        valdata = {}
        for i in range(len(valdoseindexlist)):
            valdata['trainy_Nrf2_{0}'.format(i)] = df_Srxn1[:, valdoseindexlist[i]]
        valdata = pd.DataFrame.from_dict(valdata)
        valdata = np.array(valdata)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(valdata)[1]):
            vector = valdata[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            print(np.shape(vector))
            valdata[:,i] = vector[:,0]
        print('check1')
        print('check6')

        if Srxn1_as_input == False:
            print('check3')
            trainy = np.zeros((len(traindoseindexlist), len(df_Srxn1), 1))
            print('check5')
            valy = np.zeros((int(len(valdoseindexlist)), len(df_Srxn1), 1))
            print('check4')
        time = listt[:,0]
        print('check2')
        wl = section_length
        mf = multiplication_factor
        traindoses = len(traindoseindexlist)
        iters = int(len(traindata) / mf)

        # trainy = np.zeros(iters*len(traindoseindexlist),wl,len(traindoseindexlist))
        # for i in range(len(traindoseindexlist)):
        #     for j in range(iters - int(2)):
        #         print('check ite' + str(i))
        #         trainy[j+i*int(len(traindata)/mf), :, 0] = traindata[(j*mf):((j*mf)+wl),i]
        #
        trainy = np.zeros((iters*traindoses, wl, 1))
        for i in range(traindoses):
            for j in range(iters):
                trainy[j+i*iters, :, 0] = traindata[(j*mf):((j*mf)+wl),i]

        valdoses = len(valdoseindexlist)
        itersval = int(len(valdata)/ mf)

        valy = np.zeros((itersval*valdoses, wl, 1))
        for i in range(valdoses):
            for j in range(itersval):
                valy[j+i*itersval, :, 0] = traindata[(j*mf):((j*mf)+wl),i]

        def trainYfunc_Compression_training(df_Srxn1):
            traindata = {}
            for i in range(len(traindoseindexlist)):
                traindata['trainy_Nrf2_{0}'.format(i)] = df_Srxn1[:, traindoseindexlist[i]]
            traindata = pd.DataFrame.from_dict(traindata)
            traindata = np.array(traindata)
            scaler.fit(datascalerlistSrxn1)
            for i in range(np.shape(traindata)[1]):
                vector = traindata[:, i]
                vector = vector.reshape(len(vector), 1)
                vector = scaler.transform(vector)
                print(np.shape(vector))
                traindata[:, i] = vector[:, 0]
            valdata = {}
            for i in range(len(valdoseindexlist)):
                valdata['trainy_Nrf2_{0}'.format(i)] = df_Srxn1[:, valdoseindexlist[i]]
            valdata = pd.DataFrame.from_dict(valdata)
            valdata = np.array(valdata)
            scaler.fit(datascalerlistSrxn1)
            for i in range(np.shape(valdata)[1]):
                vector = valdata[:, i]
                vector = vector.reshape(len(vector), 1)
                vector = scaler.transform(vector)
                print(np.shape(vector))
                valdata[:, i] = vector[:, 0]
            print('check1')
            print('check6')

            if Srxn1_as_input == False:
                print('check3')
                trainy = np.zeros((len(traindoseindexlist), len(df_Srxn1), 1))
                print('check5')
                valy = np.zeros((int(len(valdoseindexlist)), len(df_Srxn1), 1))
                print('check4')
            time = listt[:, 0]
            print('check2')
            wl = section_length
            mf = multiplication_factor
            traindoses = len(traindoseindexlist)
            iters = int(len(traindata) / mf)

            # trainy = np.zeros(iters*len(traindoseindexlist),wl,len(traindoseindexlist))
            # for i in range(len(traindoseindexlist)):
            #     for j in range(iters - int(2)):
            #         print('check ite' + str(i))
            #         trainy[j+i*int(len(traindata)/mf), :, 0] = traindata[(j*mf):((j*mf)+wl),i]
            #
            trainy = np.zeros((iters * traindoses, wl, 1))
            for i in range(traindoses):
                for j in range(iters):
                    trainy[j + i * iters, :, 0] = traindata[(j * mf):((j * mf) + wl), i]

            trainy = np.zeros((iters * traindoses, wl, 1))
            for i in range(traindoses):
                for j in range(iters):
                    trainy[j + i * iters, :, 0] = traindata[(j * mf):((j * mf) + wl), i]


        # # A loop: compressed sample groups
        # al = int(len(traindata)/np.shape(trainy)[0])
        # # B loop: For making C loop
        # bl = 2000
        # # C loop: individual samples
        # # D loop: selecting datapoints for the samples
        # dl = 100
        # for k in range(len(traindoseindexlist)):
        #     for a in range(al):
        #         for b in range(bl):
        #             cl = bl-b
        #             if cl >= 100:
        #                 for c in range(cl):
        #                     list = []
        #                     for j in range(dl):
        #                         if ((cl + 100 * a) / 100) * c <= 2000:
        #                             temp = traindata[((cl + 100*a)/100) * c, 0]
        #                             list.append(temp)
        #
        #                         if ((cl + 100 * a) / 100) * c > 2000:
        #                             temp = traindata[((cl + 100 * a) / 100) * c, 0]
        #                             list.append(temp)
        #
        #                         if a >= 1:
        #                             temp = traindata[((cl / 100) * j) + 100 * a, 0]
        #
        #                             temp = traindata[(cl / (100) * j) + (100 * a) -((100 * a) * c)/cl, k]
        #

    # Compression training (generates many samples of the same size but covering varying parts of the data. Samples range from the full 0-32 hours to a fraction of that, is if with every next sample the data was being compressed.
    def trainXfunc4_compression(df_Nrf2, df_Srxn1):
        traindata = {}
        for i in range(len(traindoselist)):
            traindata['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, traindoseindexlist[i]]
        traindata = pd.DataFrame.from_dict(traindata)
        traindata = np.array(traindata)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(traindata)[1]):
            vector = traindata[:, i]
            vector = vector.reshape(len(vector), 1)
            vector = scaler.transform(vector)
            print(np.shape(vector))
            traindata[:, i] = vector[:, 0]
        print('check1')
        valdata = {}
        for i in range(len(valdoseindexlist)):
            valdata['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, valdoseindexlist[i]]
        valdata = pd.DataFrame.from_dict(valdata)
        valdata = np.array(valdata)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(valdata)[1]):
            vector = valdata[:, i]
            vector = vector.reshape(len(vector), 1)
            vector = scaler.transform(vector)
            print(np.shape(vector))
            valdata[:, i] = vector[:, 0]
        # if Srxn1_as_input == False:
        #     print('check3')
        #     trainx = np.zeros((len(traindoseindexlist), len(df_Nrf2), 2))
        #     print('check5')
        #     valx = np.zeros((int(len(valdoseindexlist)), len(df_Nrf2), 2))
        #     print('check4')
        # if Srxn1_as_input == True:
        #     trainx = np.zeros((int(len(traindoseindexlist)), int((len(df_Nrf2)), 3)))
        #     valx = np.zeros((int(len(valdoseindexlist)), int((len(df_Nrf2)), 3)))
        time = listt[:, 0]
        # print('check2')
        # for i in range(len(traindoseindexlist)):
        #     trainx[i, :, 0] = time
        #     trainx[i, :, 1] = traindata[:, i]
        # for i in range(len(valdoseindexlist)):
        #     valx[i, :, 0] = time
        #     valx[i, :, 1] = valdata[:, i]

        # A loop: compressed sample groups
        trainx = np.zeros((1900 * 20 * len(traindoseindexlist), 100, 2))
        valx = np.zeros((1900 * 20 * len(valdoseindexlist) , 100, 2))
        print('check npzeros')
        print(np.shape(trainx))
        # trainx = np.array(trainx)
        # valx = np.array(valx)
        al = 20
        # B loop: For making C loop
        bl = 1900
        # C loop: individual samples
        # D loop: selecting datapoints for the samples
        cl = 100

        # Appending time: training set
        for k in range(len(traindoseindexlist)):
        # for k in range(1):
            print(str(np.shape(trainx)) + 'loop k' + str(k))
            for a in range(al):
                print(str(np.shape(trainx)) + 'loop a' + str(a))
                for b in range(bl):
                    datalen = 2000 - b
                    if datalen >= 100:
                        list = []
                        for c in range(cl):
                            temp = time[int((datalen * c) / cl + cl * a - ((cl * a * (datalen - 100 + 100 * (datalen - 100)/1900)) / 2000))]
                            list.append(temp)
                        # list = np.array(list)
                        if len(list) != 100:
                            print('error: list not 100')
                        print('appending time Xtraining tensor: sample ' + str(b + a*1900 + 1900*20*k) + '/'+ str(1900*20*len(traindoseindexlist)))
                        trainx[int(b + a*1900 + 1900*20*k), :, 0] = list[:]
                        # trainx[int(b + a*1900), :, 0] = list[:]

        # Appending time: validation set
        for k in range(len(valdoseindexlist)):
        # for k in range(1):
            print(str(np.shape(valx)) + 'loop k' + str(k))
            for a in range(al):
                print(str(np.shape(valx)) + 'loop a' + str(a))
                for b in range(bl):
                    datalen = 2000 - b
                    if datalen >= 100:
                        list = []
                        for c in range(cl):
                            temp = time[int((datalen * c) / cl + cl * a - ((cl * a * (datalen - 100 + 100 * (datalen - 100)/1900)) / 2000))]
                            list.append(temp)
                        # list = np.array(list)
                        if len(list) != 100:
                            print('error: list not 100')
                        print('appending time Xvalidation tensor: sample ' + str(b + a*1900 + 1900*20*k) + '/'+ str(1900*20*len(valdoseindexlist)))
                        valx[int(b + a*1900 + 1900*20*k), :, 0] = list[:]
                        # valx[int(b + a*1900), :, 0] = list[:]


        for k in range(len(traindoseindexlist)):
        # for k in range(1):
            print(str(np.shape(trainx)) + 'loop k' + str(k))
            for a in range(al):
                print(str(np.shape(trainx)) + 'loop a' + str(a))
                for b in range(bl):
                    datalen = 2000 - b
                    if datalen >= 100:
                        list = []
                        for c in range(cl):
                            temp = traindata[int((datalen * c) / cl + cl * a - ((cl * a * (datalen - 100 + 100 * (datalen - 100)/1900)) / 2000)), k]
                            list.append(temp)
                        # list = np.array(list)
                        if len(list) != 100:
                            print('error: list not 100')
                        print('appending data Xtraining tensor: sample ' + str(b + a*1900 + 1900*20*k) + '/'+ str(1900*20*len(traindoseindexlist)))
                        trainx[int(b + a*1900 + 1900*20*k), :, 1] = list[:]
                        # trainx[int(b + a*1900), :, 0] = list[:]

        for k in range(len(valdoseindexlist)):
        # for k in range(1):
            print(str(np.shape(valx)) + 'loop k' + str(k))
            for a in range(al):
                print(str(np.shape(valx)) + 'loop a' + str(a))
                for b in range(bl):
                    datalen = 2000 - b
                    if datalen >= 100:
                        list = []
                        for c in range(cl):
                            temp = valdata[int((datalen * c) / cl + cl * a - ((cl * a * (datalen - 100 + 100 * (datalen - 100)/1900)) / 2000)), k]
                            list.append(temp)
                        # list = np.array(list)
                        if len(list) != 100:
                            print('error: list not 100')
                        print('appending data Xvalidation tensor: sample ' + str(b + a*1900 + 1900*20*k) + '/'+ str(1900*20*len(valdoseindexlist)))
                        valx[int(b + a*1900 + 1900*20*k), :, 1] = list[:]
                        # valx[int(b + a*1900), :, 0] = list[:]

        print('shape training output: ' + str(np.shape(trainx)))
        print('shape val output: ' + str(np.shape(valx)))

        return trainx, valx
    def trainYfunc4_compression(df_Srxn1):
        traindata = {}
        for i in range(len(traindoselist)):
            traindata['trainy_Srxn1_{0}'.format(i)] = df_Srxn1[:, traindoseindexlist[i]]
        traindata = pd.DataFrame.from_dict(traindata)
        traindata = np.array(traindata)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(traindata)[1]):
            vector = traindata[:, i]
            vector = vector.reshape(len(vector), 1)
            vector = scaler.transform(vector)
            print(np.shape(vector))
            traindata[:, i] = vector[:, 0]
        print('check1')
        valdata = {}
        for i in range(len(valdoseindexlist)):
            valdata['trainy_Srxn1_{0}'.format(i)] = df_Srxn1[:, valdoseindexlist[i]]
        valdata = pd.DataFrame.from_dict(valdata)
        valdata = np.array(valdata)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(valdata)[1]):
            vector = valdata[:, i]
            vector = vector.reshape(len(vector), 1)
            vector = scaler.transform(vector)
            print(np.shape(vector))
            valdata[:, i] = vector[:, 0]
        # if Srxn1_as_input == False:
        #     print('check3')
        #     trainy = np.zeros((len(traindoseindexlist), len(df_Nrf2), 2))
        #     print('check5')
        #     valy = np.zeros((int(len(valdoseindexlist)), len(df_Nrf2), 2))
        #     print('check4')
        # if Srxn1_as_input == True:
        #     trainy = np.zeros((int(len(traindoseindexlist)), int((len(df_Nrf2)), 3)))
        #     valy = np.zeros((int(len(valdoseindexlist)), int((len(df_Nrf2)), 3)))
        # print('check2')
        # for i in range(len(traindoseindexlist)):
        #     trainy[i, :, 0] = time
        #     trainy[i, :, 1] = traindata[:, i]
        # for i in range(len(valdoseindexlist)):
        #     valy[i, :, 0] = time
        #     valy[i, :, 1] = valdata[:, i]

        # A loop: compressed sample groups
        trainy = np.zeros((1900 * 20 * len(traindoseindexlist), 100, 1))
        valy = np.zeros((1900 * 20 * len(valdoseindexlist) , 100, 1))
        print('check npzeros')
        print(np.shape(trainy))
        # trainy = np.array(trainy)
        # valy = np.array(valy)
        al = 20
        # B loop: For making C loop
        bl = 1900
        # C loop: individual samples
        # D loop: selecting datapoints for the samples
        cl = 100

        for k in range(len(traindoseindexlist)):
        # for k in range(1):
            print(str(np.shape(trainy)) + 'loop k' + str(k))
            for a in range(al):
                print(str(np.shape(trainy)) + 'loop a' + str(a))
                for b in range(bl):
                    datalen = 2000 - b
                    if datalen >= 100:
                        list = []
                        for c in range(cl):
                            temp = traindata[int((datalen * c) / cl + cl * a - ((cl * a * (datalen - 100 + 100 * (datalen - 100)/1900)) / 2000)), k]
                            list.append(temp)
                        # list = np.array(list)
                        if len(list) != 100:
                            print('error: list not 100')
                        print('appending data Ytraining tensor: sample ' + str(b + a*1900 + 1900*20*k) + '/'+ str(1900*20*len(traindoseindexlist)))
                        trainy[int(b + a*1900 + 1900*20*k), :, 0] = list[:]
                        # trainy[int(b + a*1900), :, 0] = list[:]

        for k in range(len(valdoseindexlist)):
        # for k in range(1):
            print(str(np.shape(valy)) + 'loop k' + str(k))
            for a in range(al):
                print(str(np.shape(valy)) + 'loop a' + str(a))
                for b in range(bl):
                    datalen = 2000 - b
                    if datalen >= 100:
                        list = []
                        for c in range(cl):
                            temp = valdata[int((datalen * c) / cl + cl * a - ((cl * a * (datalen - 100 + 100 * (datalen - 100)/1900)) / 2000)), k]
                            list.append(temp)
                        # list = np.array(list)
                        if len(list) != 100:
                            print('error: list not 100')
                        print('appending data Yvalidation tensor: sample ' + str(b + a*1900 + 1900*20*k) + '/'+ str(1900*20*len(valdoseindexlist)))
                        valy[int(b + a*1900 + 1900*20*k), :, 0] = list[:]
                        # valy[int(b + a*1900), :, 0] = list[:]

        print('shape training output: ' + str(np.shape(trainy)))
        print('shape val output: ' + str(np.shape(valy)))

        return trainy, valy

    # Differentiated traning:
    def trainXfunc5_diff(df_Nrf2, df_Srxn1):
        traindata = {}
        for i in range(len(traindoselist)):
            traindata['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, traindoseindexlist[i]]
        traindata = pd.DataFrame.from_dict(traindata)
        traindata = np.array(traindata)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(traindata)[1]):
            vector = traindata[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindata[:,i] = vector[:,0]
        valdata = {}
        for i in range(len(valdoseindexlist)):
            valdata['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, valdoseindexlist[i]]
        valdata = pd.DataFrame.from_dict(valdata)
        valdata = np.array(valdata)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(valdata)[1]):
            vector = valdata[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            valdata[:,i] = vector[:,0]
        trainx = np.zeros((1999*np.shape(traindata)[1],1,3))
        diffscalerlist = np.array([-0.002,0.005])
        diffscalerlist = diffscalerlist.reshape(2,1)
        diffscaler = scaler.fit(diffscalerlist)
        for i in range(np.shape(traindata)[1]):
            trainx[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            trainx[int(1999*i):int(1999*(i+1)), 0, 1] = traindata[0:1999, i]
            print('check' + str(i+1) + '.1')
            diff = traindata[: , i]
            diff = np.diff(diff)
            diff = diff.reshape(len(diff),1)
            diff = diffscaler.transform(diff)
            trainx[int(1999*i):int(1999*(i+1)), 0, 2] = diff[:,0]
            print('check' + str(i+1) + '.2')
        valx = np.zeros((1999*np.shape(valdata)[1],1,3))
        for i in range(np.shape(valdata)[1]):
            print('check' + str(i+1) + '.3')
            valx[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            valx[int(1999*i):int(1999*(i+1)), 0, 1] = valdata[0:1999, i]
            diff = valdata[: , i]
            diff = np.diff(diff)
            diff = diff.reshape(len(diff), 1)
            diff = diffscaler.transform(diff)
            valx[int(1999*i):int(1999*(i+1)), 0, 2] = diff[:,0]
            print('check' + str(i+1) + '.4')
        return trainx, valx
    def trainYfunc5_diff(df_Srxn1):
        traindata = {}
        for i in range(len(traindoselist)):
            traindata['trainy_Srxn1_{0}'.format(i)] = df_Srxn1[:, traindoseindexlist[i]]
        traindata = pd.DataFrame.from_dict(traindata)
        traindata = np.array(traindata)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(traindata)[1]):
            vector = traindata[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindata[:,i] = vector[:,0]
        valdata = {}
        for i in range(len(valdoseindexlist)):
            valdata['trainy_Srxn1_{0}'.format(i)] = df_Srxn1[:, valdoseindexlist[i]]
        valdata = pd.DataFrame.from_dict(valdata)
        valdata = np.array(valdata)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(valdata)[1]):
            vector = valdata[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            valdata[:,i] = vector[:,0]
        trainy = np.zeros((1999*np.shape(traindata)[1],1,1))
        diffscalerlist = np.array([-0.001,0.005])
        diffscalerlist = diffscalerlist.reshape(2,1)
        diffscaler = scaler.fit(diffscalerlist)
        for i in range(np.shape(traindata)[1]):
            # trainy[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            trainy[int(1999*i):int(1999*(i+1)), 0, 0] = traindata[0:1999, i]
            print('check' + str(i+1) + '.1')
            diff = traindata[: , i]
            diff = np.diff(diff)
            diff = diff.reshape(len(diff),1)
            diff = diffscaler.transform(diff)
            # trainy[int(1999*i):int(1999*(i+1)), 0, 2] = diff[:,0]
            print('check' + str(i+1) + '.2')
        valy = np.zeros((1999*np.shape(valdata)[1],1,1))
        for i in range(np.shape(valdata)[1]):
            print('check' + str(i+1) + '.3')
            # valy[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            valy[int(1999*i):int(1999*(i+1)), 0, 0] = valdata[0:1999, i]
            diff = valdata[: , i]
            diff = np.diff(diff)
            diff = diff.reshape(len(diff), 1)
            diff = diffscaler.transform(diff)
            # valy[int(1999*i):int(1999*(i+1)), 0, 1] = diff[:,0]
            print('check' + str(i+1) + '.4')
        return trainy, valy

    # No Srxn1 input:
    def trainXfunc7_Nrf2_to_dSrxn1(df_Nrf2, df_Srxn1):
        traindataNrf2 = {}
        for i in range(len(traindoselist)):
            traindataNrf2['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, traindoseindexlist[i]]
        traindataNrf2 = pd.DataFrame.from_dict(traindataNrf2)
        traindataNrf2 = np.array(traindataNrf2)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(traindataNrf2)[1]):
            vector = traindataNrf2[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindataNrf2[:,i] = vector[:,0]
        traindataSrxn1 = {}
        for i in range(len(traindoselist)):
            traindataSrxn1['trainy_Nrf2_{0}'.format(i)] = df_Srxn1[:, traindoseindexlist[i]]
        traindataSrxn1 = pd.DataFrame.from_dict(traindataSrxn1)
        traindataSrxn1 = np.array(traindataSrxn1)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(traindataSrxn1)[1]):
            vector = traindataSrxn1[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindataSrxn1[:,i] = vector[:,0]
        # validation data
        valdataNrf2 = {}
        for i in range(len(valdoseindexlist)):
            valdataNrf2['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, valdoseindexlist[i]]
        valdataNrf2 = pd.DataFrame.from_dict(valdataNrf2)
        valdataNrf2 = np.array(valdataNrf2)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(valdataNrf2)[1]):
            vector = valdataNrf2[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            valdataNrf2[:,i] = vector[:,0]
        valdataSrxn1 = {}
        for i in range(len(valdoseindexlist)):
            valdataSrxn1['trainy_Nrf2_{0}'.format(i)] = df_Srxn1[:, valdoseindexlist[i]]
        valdataSrxn1 = pd.DataFrame.from_dict(valdataSrxn1)
        valdataSrxn1 = np.array(valdataSrxn1)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(valdataSrxn1)[1]):
            vector = valdataSrxn1[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            valdataSrxn1[:,i] = vector[:,0]
        # Appending to training and validation data
        trainx = np.zeros((1999*np.shape(traindataNrf2)[1],1,2))
        for i in range(np.shape(traindataNrf2)[1]):
            trainx[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            trainx[int(1999*i):int(1999*(i+1)), 0, 1] = traindataNrf2[0:1999, i]
            # trainx[int(1999*i):int(1999*(i+1)), 0, 2] = traindataSrxn1[0:1999, i]
        valx = np.zeros((1999*np.shape(valdataNrf2)[1],1,2))
        for i in range(np.shape(valdataNrf2)[1]):
            print('check' + str(i+1) + '.3')
            valx[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            valx[int(1999*i):int(1999*(i+1)), 0, 1] = valdataNrf2[0:1999, i]
            # valx[int(1999*i):int(1999*(i+1)), 0, 2] = valdataSrxn1[0:1999, i]
            print('check' + str(i+1) + '.4')
        return trainx, valx
    def trainYfunc7_Nrf2_to_dSrxn1(df_Srxn1):
        traindata = {}
        for i in range(len(traindoselist)):
            traindata['trainy_Srxn1_{0}'.format(i)] = df_Srxn1[:, traindoseindexlist[i]]
        traindata = pd.DataFrame.from_dict(traindata)
        traindata = np.array(traindata)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(traindata)[1]):
            vector = traindata[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindata[:,i] = vector[:,0]
        valdata = {}
        for i in range(len(valdoseindexlist)):
            valdata['trainy_Srxn1_{0}'.format(i)] = df_Srxn1[:, valdoseindexlist[i]]
        valdata = pd.DataFrame.from_dict(valdata)
        valdata = np.array(valdata)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(valdata)[1]):
            vector = valdata[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            valdata[:,i] = vector[:,0]
        trainy = np.zeros((1999*np.shape(traindata)[1],1,1))
        # if training_set == 1:
        #     diffscalerlist = np.array([-0.001,0.005])
        # if training_set == 2:
        #     diffscalerlist = np.array([-0.001,0.020])
        # diffscalerlist = diffscalerlist.reshape(2,1)
        diffscaler = scaler.fit(diffscalerlist)
        for i in range(np.shape(traindata)[1]):
            # trainy[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            # trainy[int(1999*i):int(1999*(i+1)), 0, 0] = traindata[0:1999, i]
            print('check' + str(i+1) + '.1')
            diff = traindata[: , i]
            diff = np.diff(diff)
            diff = diff.reshape(len(diff),1)
            diff = diffscaler.transform(diff)
            trainy[int(1999*i):int(1999*(i+1)), 0, 0] = diff[:,0]
            print('check' + str(i+1) + '.2')
        valy = np.zeros((1999*np.shape(valdata)[1],1,1))
        for i in range(np.shape(valdata)[1]):
            # valy[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            # valy[int(1999*i):int(1999*(i+1)), 0, 0] = valdata[0:1999, i]
            diff = valdata[: , i]
            diff = np.diff(diff)
            diff = diff.reshape(len(diff), 1)
            diff = diffscaler.transform(diff)
            valy[int(1999*i):int(1999*(i+1)), 0, 0] = diff[:,0]
            print('check' + str(i+1) + '.4')
        return trainy, valy

    # dNrf2 as an additional output:
    def trainXfunc8_Nrf2_Srxn1_to_dNrf2_dSrxn1(df_Nrf2, df_Srxn1):
        traindataNrf2 = {}
        for i in range(len(traindoselist)):
            traindataNrf2['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, traindoseindexlist[i]]
        traindataNrf2 = pd.DataFrame.from_dict(traindataNrf2)
        traindataNrf2 = np.array(traindataNrf2)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(traindataNrf2)[1]):
            vector = traindataNrf2[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindataNrf2[:,i] = vector[:,0]
        print('check1')
        traindataSrxn1 = {}
        for i in range(len(traindoselist)):
            traindataSrxn1['trainy_Nrf2_{0}'.format(i)] = df_Srxn1[:, traindoseindexlist[i]]
        traindataSrxn1 = pd.DataFrame.from_dict(traindataSrxn1)
        traindataSrxn1 = np.array(traindataSrxn1)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(traindataSrxn1)[1]):
            vector = traindataSrxn1[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindataSrxn1[:,i] = vector[:,0]
        print('check2')

        # validation data
        valdataNrf2 = {}
        for i in range(len(valdoseindexlist)):
            valdataNrf2['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, valdoseindexlist[i]]
        valdataNrf2 = pd.DataFrame.from_dict(valdataNrf2)
        valdataNrf2 = np.array(valdataNrf2)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(valdataNrf2)[1]):
            vector = valdataNrf2[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            valdataNrf2[:,i] = vector[:,0]
        print('check3')

        valdataSrxn1 = {}
        for i in range(len(valdoseindexlist)):
            valdataSrxn1['trainy_Nrf2_{0}'.format(i)] = df_Srxn1[:, valdoseindexlist[i]]
        valdataSrxn1 = pd.DataFrame.from_dict(valdataSrxn1)
        valdataSrxn1 = np.array(valdataSrxn1)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(valdataSrxn1)[1]):
            vector = valdataSrxn1[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            valdataSrxn1[:,i] = vector[:,0]
        print('check4')

        # Appending to training and validation data
        trainx = np.zeros((1999*np.shape(traindataNrf2)[1],1,3))
        for i in range(np.shape(traindataNrf2)[1]):
            trainx[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            trainx[int(1999*i):int(1999*(i+1)), 0, 1] = traindataNrf2[0:1999, i]
            trainx[int(1999*i):int(1999*(i+1)), 0, 2] = traindataSrxn1[0:1999, i]
        valx = np.zeros((1999*np.shape(valdataNrf2)[1],1,3))
        for i in range(np.shape(valdataNrf2)[1]):
            print('check' + str(i+1) + '.3')
            valx[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            valx[int(1999*i):int(1999*(i+1)), 0, 1] = valdataNrf2[0:1999, i]
            valx[int(1999*i):int(1999*(i+1)), 0, 2] = valdataSrxn1[0:1999, i]
            print('check' + str(i+1) + '.4')
        return trainx, valx
    def trainYfunc8_Nrf2_Srxn1_to_dNrf2_dSrxn1(df_Nrf2, df_Srxn1):
        traindataSrxn1 = {}
        for i in range(len(traindoselist)):
            traindataSrxn1['trainy_Srxn1_{0}'.format(i)] = df_Srxn1[:, traindoseindexlist[i]]
        traindataSrxn1 = pd.DataFrame.from_dict(traindataSrxn1)
        traindataSrxn1 = np.array(traindataSrxn1)

        traindataNrf2 = {}
        for i in range(len(traindoselist)):
            traindataNrf2['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, traindoseindexlist[i]]
        traindataNrf2 = pd.DataFrame.from_dict(traindataNrf2)
        traindataNrf2 = np.array(traindataNrf2)

        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(traindataSrxn1)[1]):
            vector = traindataSrxn1[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindataSrxn1[:,i] = vector[:,0]

        valdataSrxn1 = {}
        for i in range(len(valdoseindexlist)):
            valdataSrxn1['trainy_Srxn1_{0}'.format(i)] = df_Srxn1[:, valdoseindexlist[i]]
        valdataSrxn1 = pd.DataFrame.from_dict(valdataSrxn1)
        valdataSrxn1 = np.array(valdataSrxn1)

        valdataNrf2 = {}
        for i in range(len(valdoseindexlist)):
            valdataNrf2['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, valdoseindexlist[i]]
        valdataNrf2 = pd.DataFrame.from_dict(valdataNrf2)
        valdataNrf2 = np.array(valdataNrf2)

        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(valdataSrxn1)[1]):
            vector = valdataSrxn1[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            valdataSrxn1[:,i] = vector[:,0]
        trainy = np.zeros((1999*np.shape(traindataSrxn1)[1], 1, 2))
        valy =   np.zeros((1999 * np.shape(valdataSrxn1)[1], 1, 2))
        # if training_set == 1:
        #     diffscalerlist = np.array([-0.001,0.005])
        # if training_set == 2:
        #     diffscalerlist = np.array([-0.001,0.020])
        # diffscalerlist = diffscalerlist.reshape(2,1)

        # Nrf2 train and val
        diffscaler = scaler.fit(diffscalerlistNrf2)
        for i in range(np.shape(traindataNrf2)[1]):
            # trainy[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            # trainy[int(1999*i):int(1999*(i+1)), 0, 0] = traindataNrf2[0:1999, i]
            print('check' + str(i+1) + '.1')
            diff = traindataNrf2[: , i]
            diff = np.diff(diff)
            diff = diff.reshape(len(diff),1)
            diff = diffscaler.transform(diff)
            trainy[int(1999*i):int(1999*(i+1)), 0, 0] = diff[:,0]
            print('check' + str(i+1) + '.2')
        for i in range(np.shape(valdataNrf2)[1]):
            print(valdataNrf2)
            # valy[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            # valy[int(1999*i):int(1999*(i+1)), 0, 0] = valdataNrf2[0:1999, i]
            diff = valdataNrf2[:, i]
            diff = np.diff(diff)
            diff = diff.reshape(len(diff), 1)
            diff = diffscaler.transform(diff)
            valy[int(1999 * i):int(1999 * (i + 1)), 0, 0] = diff[:, 0]
            print('check' + str(i + 1) + '.4')
        # Srxn1 train and val
        diffscaler = scaler.fit(diffscalerlistSrxn1)
        for i in range(np.shape(traindataSrxn1)[1]):
            # trainy[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            # trainy[int(1999*i):int(1999*(i+1)), 0, 0] = traindataSrxn1[0:1999, i]
            print('check' + str(i+1) + '.1')
            diff = traindataSrxn1[: , i]
            diff = np.diff(diff)
            diff = diff.reshape(len(diff),1)
            diff = diffscaler.transform(diff)
            trainy[int(1999*i):int(1999*(i+1)), 0, 1] = diff[:,0]
            print('check' + str(i+1) + '.2')
        for i in range(np.shape(valdataSrxn1)[1]):
            # valy[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            # valy[int(1999*i):int(1999*(i+1)), 0, 0] = valdataSrxn1[0:1999, i]
            diff = valdataSrxn1[: , i]
            diff = np.diff(diff)
            diff = diff.reshape(len(diff), 1)
            diff = diffscaler.transform(diff)
            valy[int(1999*i):int(1999*(i+1)), 0, 1] = diff[:,0]
            print('check' + str(i+1) + '.4')
        return trainy, valy

    # Multiple exposure conditions
    def trainXfunc12_Nrf2_Srxn1_to_dSrxn1_ME(df_Nrf2, df_Srxn1, trainindex, valindex, include_baseline=False, include_time=False):
        traindataNrf2 = {}
        for i in range(len(traindoseindexlist)):
            traindataNrf2['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, traindoseindexlist[i]]
        traindataNrf2 = pd.DataFrame.from_dict(traindataNrf2)
        traindataNrf2 = np.array(traindataNrf2)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(traindataNrf2)[1]):
            vector = traindataNrf2[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindataNrf2[:,i] = vector[:,0]
        traindataSrxn1 = {}
        for i in range(len(traindoseindexlist)):
            traindataSrxn1['trainy_Nrf2_{0}'.format(i)] = df_Srxn1[:, traindoseindexlist[i]]
        traindataSrxn1 = pd.DataFrame.from_dict(traindataSrxn1)
        traindataSrxn1 = np.array(traindataSrxn1)
        # scaler.fit(datascalerlistNrf2)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(traindataSrxn1)[1]):
            vector = traindataSrxn1[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindataSrxn1[:,i] = vector[:,0]

        # validation data
        valdataNrf2 = {}
        for i in range(len(valdoseindexlist)):
            valdataNrf2['trainy_Nrf2_{0}'.format(i)] = df_Nrf2[:, valdoseindexlist[i]]
        valdataNrf2 = pd.DataFrame.from_dict(valdataNrf2)
        valdataNrf2 = np.array(valdataNrf2)
        scaler.fit(datascalerlistNrf2)
        for i in range(np.shape(valdataNrf2)[1]):
            vector = valdataNrf2[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            valdataNrf2[:,i] = vector[:,0]
        valdataSrxn1 = {}
        for i in range(len(valdoseindexlist)):
            valdataSrxn1['trainy_Nrf2_{0}'.format(i)] = df_Srxn1[:, valdoseindexlist[i]]
        valdataSrxn1 = pd.DataFrame.from_dict(valdataSrxn1)
        valdataSrxn1 = np.array(valdataSrxn1)
        # scaler.fit(datascalerlistNrf2)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(valdataSrxn1)[1]):
            vector = valdataSrxn1[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            valdataSrxn1[:,i] = vector[:,0]

        if include_time == True:
            # Appending to training and validation data
            if include_baseline == False:
                trainx = np.zeros((1999*np.shape(traindataNrf2)[1],1,3))
                valx = np.zeros((1999 * np.shape(valdataNrf2)[1], 1, 3))
            # If a baseline is included the empty vector is extended for this baseline (only for training, useless for validation(?)
            if include_baseline == True:
                trainx = np.zeros((1999 * np.shape(traindataNrf2)[1] + 1999, 1, 3))
                valx = np.zeros((1999 * np.shape(valdataNrf2)[1], 1, 3))

            for i in range(np.shape(traindataNrf2)[1]):
            # for i in range(2):
                if include_baseline == False:
                    trainx[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
                if include_baseline == True:
                    trainx[int(1999 * i):int(1999 * (i + 1)), 0, 0] = listt[:-1, 0]
                    # baseline value calculation and 0-1 normalization
                    scaler.fit(datascalerlistNrf2)
                    Nrf2baseline = np.zeros((1999,1))
                    BLvalue = np.amin(df_Nrf2)
                    BLvalue = BLvalue.reshape(1,1)
                    BLvalue = scaler.transform(BLvalue)
                    for j in range(len(Nrf2baseline)):
                        Nrf2baseline[j,0] = BLvalue
                    scaler.fit(datascalerlistSrxn1)
                    Srxn1baseline = np.zeros((1999,1))
                    BLvalue = np.amin(df_Srxn1)
                    BLvalue = BLvalue.reshape(1,1)
                    BLvalue = scaler.transform(BLvalue)
                    for j in range(len(Srxn1baseline)):
                        Srxn1baseline[j,0] = BLvalue
                    # Appending baseline value to the last 1999 datapoints of the tensor
                    trainx[int(1999 * np.shape(traindataNrf2)[1]):int(1999 * (np.shape(traindataNrf2)[1] + 1)), 0, 0] = listt[0:1999, 0]
                    trainx[int(1999 * np.shape(traindataNrf2)[1]):int(1999 * (np.shape(traindataNrf2)[1] + 1)), 0, 1] = Nrf2baseline[:, 0]
                    trainx[int(1999 * np.shape(traindataNrf2)[1]):int(1999 * (np.shape(traindataNrf2)[1] + 1)), 0, 2] = Srxn1baseline[:, 0]
                # trainx[0:1999, 0, 1] = traindataNrf2[0:1999, 2]
                trainx[int(1999*i):int(1999*(i+1)), 0, 1] = traindataNrf2[0:1999, i]
                trainx[int(1999*i):int(1999*(i+1)), 0, 2] = traindataSrxn1[0:1999, i]
            for i in range(np.shape(valdataNrf2)[1]):
                print('check' + str(i+1) + '.3')
                valx[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
                valx[int(1999*i):int(1999*(i+1)), 0, 1] = valdataNrf2[0:1999, i]
                valx[int(1999*i):int(1999*(i+1)), 0, 2] = valdataSrxn1[0:1999, i]
                print('check' + str(i+1) + '.4')
        if include_time == False:
            # Appending to training and validation data
            if include_baseline == False:
                trainx = np.zeros((1999 * np.shape(traindataNrf2)[1], 1, 2))
                valx = np.zeros((1999 * np.shape(valdataNrf2)[1], 1, 2))
            # If a baseline is included the empty vector is extended for this baseline (only for training, useless for validation(?)
            if include_baseline == True:
                trainx = np.zeros((1999 * np.shape(traindataNrf2)[1] + 1999, 1, 2))
                valx = np.zeros((1999 * np.shape(valdataNrf2)[1], 1, 2))
            for i in range(np.shape(traindataNrf2)[1]):
                # for i in range(2):
                if include_baseline == True:
                    # trainx[int(1999 * i):int(1999 * (i + 1)), 0, 0] = listt[:-1, 0]
                    # baseline value calculation and 0-1 normalization
                    scaler.fit(datascalerlistNrf2)
                    Nrf2baseline = np.zeros((1999, 1))
                    BLvalue = np.amin(df_Nrf2)
                    BLvalue = BLvalue.reshape(1, 1)
                    BLvalue = scaler.transform(BLvalue)
                    for j in range(len(Nrf2baseline)):
                        Nrf2baseline[j, 0] = BLvalue
                    scaler.fit(datascalerlistSrxn1)
                    Srxn1baseline = np.zeros((1999, 1))
                    BLvalue = np.amin(df_Srxn1)
                    BLvalue = BLvalue.reshape(1, 1)
                    BLvalue = scaler.transform(BLvalue)
                    for j in range(len(Srxn1baseline)):
                        Srxn1baseline[j, 0] = BLvalue
                    # Appending baseline value to the last 1999 datapoints of the tensor
                    # trainx[int(1999 * np.shape(traindataNrf2)[1]):int(1999 * (np.shape(traindataNrf2)[1] + 1)), 0, 0] = listt[
                    #                                                                                                     0:1999,
                    #                                                                                                     0]
                    trainx[int(1999 * np.shape(traindataNrf2)[1]):int(1999 * (np.shape(traindataNrf2)[1] + 1)), 0,
                    0] = Nrf2baseline[:, 0]
                    trainx[int(1999 * np.shape(traindataNrf2)[1]):int(1999 * (np.shape(traindataNrf2)[1] + 1)), 0,
                    1] = Srxn1baseline[:, 0]
                # trainx[0:1999, 0, 1] = traindataNrf2[0:1999, 2]
                trainx[int(1999 * i):int(1999 * (i + 1)), 0, 0] = traindataNrf2[0:1999, i]
                trainx[int(1999 * i):int(1999 * (i + 1)), 0, 1] = traindataSrxn1[0:1999, i]
            for i in range(np.shape(valdataNrf2)[1]):
                print('check' + str(i + 1) + '.3')

                # valx[int(1999 * i):int(1999 * (i + 1)), 0, 0] = listt[:-1, 0]
            valx[int(1999 * i):int(1999 * (i + 1)), 0, 0] = valdataNrf2[0:1999, i]
            valx[int(1999 * i):int(1999 * (i + 1)), 0, 1] = valdataSrxn1[0:1999, i]
            print('check' + str(i + 1) + '.4')
        return trainx, valx
    def trainYfunc12_Nrf2_Srxn1_to_smoothed_dSrxn1_ME(df_Srxn1, wlength, poly_order, include_baseline=False, include_time=False):
        traindata = {}
        for i in range(len(traindoseindexlist)):
            traindata['trainy_Srxn1_{0}'.format(i)] = df_Srxn1[:, traindoseindexlist[i]]
        traindata = pd.DataFrame.from_dict(traindata)
        traindata = np.array(traindata)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(traindata)[1]):
            vector = traindata[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindata[:,i] = vector[:,0]
        valdata = {}
        for i in range(len(valdoseindexlist)):
            valdata['trainy_Srxn1_{0}'.format(i)] = df_Srxn1[:, valdoseindexlist[i]]
        valdata = pd.DataFrame.from_dict(valdata)
        valdata = np.array(valdata)
        # scaler.fit(datascalerlistSrxn1)
        # for i in range(np.shape(valdata)[1]):
        #     vector = valdata[:,i]
        #     vector = vector.reshape(len(vector),1)
        #     vector = scaler.transform(vector)
        #     valdata[:,i] = vector[:,0]
        if include_baseline == False:
            trainy = np.zeros((1999*np.shape(traindata)[1],1,1))
        if include_baseline == True:
            trainy = np.zeros((1999*np.shape(traindata)[1] +1999,1,1))

        # if training_set == 1:
        #     diffscalerlist = np.array([-0.001,0.005])
        # if training_set == 2:
        #     diffscalerlist = np.array([-0.001,0.020])
        # diffscalerlist = diffscalerlist.reshape(2,1)
        diffscaler = scaler.fit(diffscalerlist)
        for i in range(np.shape(traindata)[1]):
            # trainy[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            # trainy[int(1999*i):int(1999*(i+1)), 0, 0] = traindata[0:1999, i]
            print('check' + str(i+1) + '.1')
            diff = traindata[: , i]
            diff = np.diff(diff)
            # plt.plot(diff, label='before smoothing')
            diff = savgol_filter(diff, wlength, poly_order)
            # diff = savgol_filter(diff, 1501, 5)
            # plt.plot(diff, label='after smoothing')
            diff = diff.reshape(len(diff),1)
            diff = diffscaler.transform(diff)
            trainy[int(1999*i):int(1999*(i+1)), 0, 0] = diff[:,0]
            print('check' + str(i+1) + '.2')
        zerodifflist = np.zeros((1999,1))
        for i in range(1999):
            zerodifflist[i,0] = 0
        zerodifflist = diffscaler.transform(zerodifflist)
        if include_baseline == True:
            trainy[int(1999 * np.shape(traindata)[1]):int(1999 * (np.shape(traindata)[1] + 1)), 0, 0] = zerodifflist[0:1999, 0]
        valy = np.zeros((1999*np.shape(valdata)[1],1,1))
        for i in range(np.shape(valdata)[1]):
            # valy[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            # valy[int(1999*i):int(1999*(i+1)), 0, 0] = valdata[0:1999, i]
            diff = valdata[: , i]
            diff = np.diff(diff)
            diff = diff.reshape(len(diff), 1)
            diff = diffscaler.transform(diff)
            valy[int(1999*i):int(1999*(i+1)), 0, 0] = diff[:,0]
            print('check' + str(i+1) + '.4')
        return trainy, valy

    def trainYfunc13_Nrf2_Srxn1_to_smoothed_dSrxn1(df_Srxn1, wlength, poly_order):
        traindata = {}
        for i in range(len(traindoseindexlist)):
            traindata['trainy_Srxn1_{0}'.format(i)] = df_Srxn1[:, traindoseindexlist[i]]
        traindata = pd.DataFrame.from_dict(traindata)
        traindata = np.array(traindata)
        scaler.fit(datascalerlistSrxn1)
        for i in range(np.shape(traindata)[1]):
            vector = traindata[:,i]
            vector = vector.reshape(len(vector),1)
            vector = scaler.transform(vector)
            traindata[:,i] = vector[:,0]
        valdata = {}
        for i in range(len(valdoseindexlist)):
            valdata['trainy_Srxn1_{0}'.format(i)] = df_Srxn1[:, valdoseindexlist[i]]
        valdata = pd.DataFrame.from_dict(valdata)
        valdata = np.array(valdata)
        # scaler.fit(datascalerlistSrxn1)
        # for i in range(np.shape(valdata)[1]):
        #     vector = valdata[:,i]
        #     vector = vector.reshape(len(vector),1)
        #     vector = scaler.transform(vector)
        #     valdata[:,i] = vector[:,0]
        trainy = np.zeros((1999*np.shape(traindata)[1],1,1))
        # if training_set == 1:
        #     diffscalerlist = np.array([-0.001,0.005])
        # if training_set == 2:
        #     diffscalerlist = np.array([-0.001,0.020])
        # diffscalerlist = diffscalerlist.reshape(2,1)
        diffscaler = scaler.fit(diffscalerlist)
        for i in range(np.shape(traindata)[1]):
            # trainy[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            # trainy[int(1999*i):int(1999*(i+1)), 0, 0] = traindata[0:1999, i]
            print('check' + str(i+1) + '.1')
            diff = traindata[: , i]
            diff = np.diff(diff)
            # plt.plot(diff, label='before smoothing')
            diff = savgol_filter(diff, wlength, poly_order)
            # diff = savgol_filter(diff, 1501, 5)
            # plt.plot(diff, label='after smoothing')
            diff = diff.reshape(len(diff),1)
            diff = diffscaler.transform(diff)
            trainy[int(1999*i):int(1999*(i+1)), 0, 0] = diff[:,0]
            print('check' + str(i+1) + '.2')
        valy = np.zeros((1999*np.shape(valdata)[1],1,1))
        for i in range(np.shape(valdata)[1]):
            # valy[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
            # valy[int(1999*i):int(1999*(i+1)), 0, 0] = valdata[0:1999, i]
            diff = valdata[: , i]
            diff = np.diff(diff)
            diff = diff.reshape(len(diff), 1)
            diff = diffscaler.transform(diff)
            valy[int(1999*i):int(1999*(i+1)), 0, 0] = diff[:,0]
            print('check' + str(i+1) + '.4')
        return trainy, valy

# 11. Other functions
for h in range(1):
    # training statistics, saving model
    # posttraining: generates a plot of the training loss and validation loss and saves it to a folder
    # Additionally it saves all loss values as .txt files so a loss value at a specific epoch can be looked up
    def posttraining(model, history):
        for h in range(1):
            lossdict = {}
            val_lossdict = {}
            loss = np.around(np.log10(history.history['loss'][-1]), 3)
            model.save(
                str(directory) + 'Versions/NN2/run' + str(run) + ' seed=' + str(
                    seed) + '.h5')
            # model.save(
            #     str(directory) + 'Versions/NN2/run' + str(run) + ' seed=' + str(
            #         seed) + ' loss=10^' + str(loss) + '.h5')
            tf.compat.v1.reset_default_graph()

            # Training loss
            for k in range(1):
                history_dict = history.history
                loss_values = np.log10(history_dict['loss'])
                val_loss_values = np.log10(history_dict['val_loss'])
                epochs = range(1, epch + 1)
                plt.plot(epochs, loss_values, 'r', label='Training loss')
                plt.plot(epochs, val_loss_values, 'g', label='Validation loss')
                plt.title(' Training and Validation loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss (log10)')
                plt.legend()
            plt.savefig('E:/BFW/Master BPS/RP1/Technical/Causal model tests/Training loss and acc/NN2/run' + str(
                run) + ' seed=' + str(seed) + ' loss=10^' + str(loss) + '.png')
            plt.clf()

            lossdict['loss_seed{0}'.format(seed)] = history.history['loss']
            lossdict = pd.DataFrame(lossdict)
            lossdict.to_csv(str(directory) + 'Loss dicts/NN2/loss run' + str(run) + ' seed=' + str(seed) + '.txt')
            val_lossdict['val_loss_seed{0}'.format(seed)] = history.history['val_loss']
            val_lossdict = pd.DataFrame(val_lossdict)
            val_lossdict.to_csv(str(directory) + 'Loss dicts/NN2/val_loss run' + str(run) + ' seed=' + str(seed) + '.txt')

    # Plotting for training on Equation set 5
    # NNconfig: 1 for LSTMcv, 2 for LSTMdtp
    # trainnr/testnr = number of train and test sets used respectively
    # inputnr = number of input variables like time, Nrf2, Srxn1. Use 2 or 3 depending on whether Srxn1 is included as an input
    def plottingES5(trainX, trainY, NNconfig, trainnr, testnr, inputnr):
        colors = ['red', 'green', 'blue', 'purple']
        #NNconfig: 1 = LSTM, 2 = ODE
        if NNconfig == 1:
            for i in range(4):
                indexnr = i
                predic = trainX[i, :, :]
                predic = predic.reshape(1, 2000, 2)
                predic = model.predict(predic)
                print(np.shape(predic))
                plt.plot(predic[0, :, 0], label='train set ' + str(indexnr + 1) + ' prediction',
                         color=str(colors[i]), linestyle='dashed')
                plt.plot(trainY[i, :, 0], label='train set' + str(i + 1), color=str(colors[i]))
                plt.xlabel('datapoints')
                plt.ylabel('value')
                plt.title('Run' + str(run) + ' (LSTM conf.) equation set 1 training prediction')
                plt.legend()

            for i in range(2):
                indexnr = i
                print(i)
                predic = testX[i, :, :]
                predic = predic.reshape(1, 2000, 2)
                predic = model.predict(predic)
                plt.plot(predic[0, :, 0], label='test set ' + str(indexnr + 1) + ' prediction', linestyle='dashed',
                         color=str(colors[i]))
                plt.plot(testY[i, :, 0], label='test set' + str(i + 1), color=str(colors[i]))
                plt.xlabel('datapoints')
                plt.ylabel('value')
                plt.title('Run' + str(run) + ' (LSTM conf.) equation set 1 test prediction')
                plt.legend()
            plt.clf()
        if NNconfig == 2:
            for i in range(trainnr):
                indexnr = i
                predic = trainX[int(1999 * indexnr):int(1999 * (indexnr + 1)), :, :]
                if inputnr == 2:
                    predic = predic.reshape(1999, 1, 2)
                if inputnr == 3:
                    predic = predic.reshape(1999, 1, 3)
                predic = model.predict(predic)
                print(np.shape(predic))

                # predicmse = np.log10(mean_squared_error((trainY[int(1999 * indexnr):int(1999 * (indexnr + 1)), 0], predic1[:, 0])))

                plt.plot(predic[:, 0], label='train set ' + str(indexnr + 1) + ' prediction, mse=', color=str(colors[i]),
                         linestyle='dashed')
                plt.plot(trainY[int(1999 * indexnr):int(1999 * (indexnr + 1)), 0], label='train set' + str(i + 1),
                         color=str(colors[i]))
                plt.xlabel('datapoints')
                plt.ylabel('value')
                plt.title('Run' + str(run) + ' (ODE conf.) equation set 1 train prediction')
                plt.legend()
                plt.savefig('E:/BFW/Master BPS/RP1/Technical/ODE learning tests/run' + str(
                    run) + ' seed=' + str(seed) + ' ES1 train.png')
            plt.clf()

            for i in range(testnr):
                indexnr = i
                predic = testX[int(1999 * indexnr):int(1999 * (indexnr + 1)), :, :]
                if inputnr == 2:
                    predic = predic.reshape(1999, 1, 2)
                if inputnr == 3:
                    predic = predic.reshape(1999, 1, 3)
                predic = model.predict(predic)
                plt.plot(predic[:, 0], label='test set ' + str(indexnr + 1) + ' prediction', color=str(colors[i]),
                         linestyle='dashed')
                plt.plot(testY[int(1999 * indexnr):int(1999 * (indexnr + 1)), 0], color=str(colors[i]),
                         label='test set' + str(i + 1))
                plt.xlabel('datapoints')
                plt.ylabel('value')
                plt.title('Run' + str(run) + ' (ODE conf.) equation set 1 test prediction')
                plt.legend()
                plt.savefig('E:/BFW/Master BPS/RP1/Technical/ODE learning tests/run' + str(
                    run) + ' seed=' + str(seed) + ' ES1 test.png')
            plt.clf()
            
            for i in range(1):
                output, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_ES5[:, 5], pDF_Srxn1_Rmean_ES5[0, 5],1998)

                plt.plot(output, label='integrated test dose 1', color='green', linestyle='dashed')
                plt.plot(pDF_Srxn1_Rmean_ES5[:, 5], label='test dose 1', color='green')
                output, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_ES5[:, 6], pDF_Srxn1_Rmean_ES5[0, 6],1998)
                plt.plot(output, label='integrated test dose 1', color='red', linestyle='dashed')
                plt.plot(pDF_Srxn1_Rmean_ES5[:, 6], label='test dose 1', color='red')
                plt.legend()
                plt.xlabel('data points')
                plt.ylabel('0-1 normalized value')
                plt.title('run' + str(run) + ' seed=' + str(seed) + ' integrated fit')
                plt.savefig('E:/BFW/Master BPS/RP1/Technical/ODE learning tests/run' + str(
                    run) + ' seed=' + str(seed) + ' integrated test.png')
            plt.clf()

    # Plotting for training on Equation set 1
    # NNconfig: 1 for LSTMcv, 2 for LSTMdtp
    # trainnr/testnr = number of train and test sets used respectively
    # inputnr = number of input variables like time, Nrf2, Srxn1. Use 2 or 3 depending on whether Srxn1 is included as an input
    def plottingES1(trainX, trainY, NNconfig, trainnr, testnr, inputnr):
        colors = ['red', 'green', 'blue', 'purple']

        # NNconfig: 1 = LSTM, 2 = ODE
        if NNconfig == 1:
            for i in range(4):
                indexnr = i
                predic = trainX[i, :, :]
                predic = predic.reshape(1, 2000, 2)
                predic = model.predict(predic)
                print(np.shape(predic))
                plt.plot(predic[0, :, 0], label='train set ' + str(indexnr + 1) + ' prediction',
                         color=str(colors[i]), linestyle='dashed')
                plt.plot(trainY[i, :, 0], label='train set' + str(i + 1), color=str(colors[i]))
                plt.xlabel('datapoints')
                plt.ylabel('value')
                plt.title('Run' + str(run) + ' (LSTM conf.) equation set 1 training prediction')
                plt.legend()

            for i in range(2):
                indexnr = i
                print(i)
                predic = testX[i, :, :]
                predic = predic.reshape(1, 2000, 2)
                predic = model.predict(predic)
                plt.plot(predic[0, :, 0], label='test set ' + str(indexnr + 1) + ' prediction', linestyle='dashed',
                         color=str(colors[i]))
                plt.plot(testY[i, :, 0], label='test set' + str(i + 1), color=str(colors[i]))
                plt.xlabel('datapoints')
                plt.ylabel('value')
                plt.title('Run' + str(run) + ' (LSTM conf.) equation set 1 test prediction')
                plt.legend()
            plt.clf()
        if NNconfig == 2:
            for i in range(trainnr):
                indexnr = i
                predic = trainX[int(1999 * indexnr):int(1999 * (indexnr + 1)), :, :]
                if inputnr == 2:
                    predic = predic.reshape(1999, 1, 2)
                if inputnr == 3:
                    predic = predic.reshape(1999, 1, 3)
                predic = model.predict(predic)
                print(np.shape(predic))

                # predicmse = np.log10(mean_squared_error((trainY[int(1999 * indexnr):int(1999 * (indexnr + 1)), 0], predic1[:, 0])))

                plt.plot(predic[:, 0], label='train set ' + str(indexnr + 1) + ' prediction, mse=',
                         color=str(colors[i]),
                         linestyle='dashed')
                plt.plot(trainY[int(1999 * indexnr):int(1999 * (indexnr + 1)), 0], label='train set' + str(i + 1),
                         color=str(colors[i]))
                plt.xlabel('datapoints')
                plt.ylabel('value')
                plt.title('Run' + str(run) + ' (ODE conf.) equation set 1 train prediction')
                plt.legend()
                plt.savefig('E:/BFW/Master BPS/RP1/Technical/ODE learning tests/run' + str(
                    run) + ' seed=' + str(seed) + ' ES1 train.png')
            plt.clf()

            for i in range(testnr):
                indexnr = i
                predic = testX[int(1999 * indexnr):int(1999 * (indexnr + 1)), :, :]
                if inputnr == 2:
                    predic = predic.reshape(1999, 1, 2)
                if inputnr == 3:
                    predic = predic.reshape(1999, 1, 3)
                predic = model.predict(predic)
                plt.plot(predic[:, 0], label='test set ' + str(indexnr + 1) + ' prediction', color=str(colors[i]),
                         linestyle='dashed')
                plt.plot(testY[int(1999 * indexnr):int(1999 * (indexnr + 1)), 0], color=str(colors[i]),
                         label='test set' + str(i + 1))
                plt.xlabel('datapoints')
                plt.ylabel('value')
                plt.title('Run' + str(run) + ' (ODE conf.) equation set 1 test prediction')
                plt.legend()
                plt.savefig('E:/BFW/Master BPS/RP1/Technical/ODE learning tests/run' + str(
                    run) + ' seed=' + str(seed) + ' ES1 test.png')
            plt.clf()

            for i in range(1):
                output, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_ES1[:, 4], pDF_Srxn1_Rmean_ES1[0, 4],
                                                                   1998)

                plt.plot(output, label='integrated test dose 1', color='green', linestyle='dashed')
                plt.plot(pDF_Srxn1_Rmean_ES1[:, 4], label='test dose 1', color='green')
                output, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_ES1[:, 5], pDF_Srxn1_Rmean_ES1[0, 5],
                                                                   1998)
                plt.plot(output, label='integrated test dose 1', color='red', linestyle='dashed')
                plt.plot(pDF_Srxn1_Rmean_ES1[:, 5], label='test dose 1', color='red')
                plt.legend()
                plt.xlabel('data points')
                plt.ylabel('0-1 normalized value')
                plt.title('run' + str(run) + ' seed=' + str(seed) + ' integrated fit')
                plt.savefig('E:/BFW/Master BPS/RP1/Technical/ODE learning tests/run' + str(
                    run) + ' seed=' + str(seed) + ' integrated test.png')
            plt.clf()

    # Plotting for Nrf2 > dSrxn1 (not very relevant)
    def plotting_Nrf2_to_dSrxn1(trainX, trainY, NNconfig):
        colors = ['red', 'green', 'blue', 'purple']
        # NNconfig: 1 = LSTM, 2 = ODE
        if NNconfig == 1:
            for i in range(4):
                indexnr = i
                predic = trainX[i, :, :]
                predic = predic.reshape(1, 2000, 2)
                predic = model.predict(predic)
                print(np.shape(predic))
                plt.plot(predic[0, :, 0], label='train set ' + str(indexnr + 1) + ' prediction',
                         color=str(colors[i]), linestyle='dashed')
                plt.plot(trainY[i, :, 0], label='train set' + str(i + 1), color=str(colors[i]))
                plt.xlabel('datapoints')
                plt.ylabel('value')
                plt.title('Run' + str(run) + ' (LSTM conf.) equation set 1 training prediction')
                plt.legend()

            for i in range(2):
                indexnr = i
                print(i)
                predic = testX[i, :, :]
                predic = predic.reshape(1, 2000, 2)
                predic = model.predict(predic)
                plt.plot(predic[0, :, 0], label='test set ' + str(indexnr + 1) + ' prediction', linestyle='dashed',
                         color=str(colors[i]))
                plt.plot(testY[i, :, 0], label='test set' + str(i + 1), color=str(colors[i]))
                plt.xlabel('datapoints')
                plt.ylabel('value')
                plt.title('Run' + str(run) + ' (LSTM conf.) equation set 1 test prediction')
                plt.legend()
            plt.clf()
        if NNconfig == 2:
            for i in range(4):
                indexnr = i
                predic = trainX[int(1999 * indexnr):int(1999 * (indexnr + 1)), :, :]
                predic = predic.reshape(1999, 1, 2)
                predic = model.predict(predic)
                print(np.shape(predic))
                plt.plot(predic[:, 0], label='train set ' + str(indexnr + 1) + ' prediction', color=str(colors[i]),
                         linestyle='dashed')
                plt.plot(trainY[int(1999 * indexnr):int(1999 * (indexnr + 1)), 0], label='train set' + str(i + 1),
                         color=str(colors[i]))
                plt.xlabel('datapoints')
                plt.ylabel('value')
                plt.title('Run' + str(run) + ' (ODE conf.) equation set 1 train prediction')
                plt.legend()
                plt.savefig('E:/BFW/Master BPS/RP1/Technical/ODE learning tests/run' + str(
                    run) + ' seed=' + str(seed) + ' ES1 train.png')
            plt.clf()

            for i in range(2):
                indexnr = i
                predic = testX[int(1999 * indexnr):int(1999 * (indexnr + 1)), :, :]
                predic = predic.reshape(1999, 1, 2)
                plt.plot(predic[:,0,0], label=':,0,0')
                plt.plot(predic[:,0,1], label=':,0,1')
            #     predic = model.predict(predic)
            #     plt.plot(predic[:, 0], label='test set ' + str(indexnr + 1) + ' prediction', color=str(colors[i]),
            #              linestyle='dashed')
            #     plt.plot(testY[int(1999 * indexnr):int(1999 * (indexnr + 1)), 0], color=str(colors[i]),
            #              label='test set' + str(i + 1))
            #     plt.xlabel('datapoints')
            #     plt.ylabel('value')
            #     plt.title('Run' + str(run) + ' (ODE conf.) equation set 1 test prediction')
            #     plt.legend()
            #     plt.savefig('E:/BFW/Master BPS/RP1/Technical/ODE learning tests/run' + str(
            #         run) + ' seed=' + str(seed) + ' ES1 test.png')
            # plt.clf()

    # Despite its name not an actual ODE integrator, just integrates the derivative trained neural network output (LSTMdtp)
    # Nrf2_data = Nrf2 pDF, y0Srxn1 = initial Srxn1 value, timesteps = amount of iterations, usually 1998 or 1999
    def ODE_integrator(Nrf2_data, y0Srxn1, timesteps):
        timelist = [listt[0, 0]]
        Nrf2list = [Nrf2_data[0]]
        Srxn1list = [y0Srxn1]
        dSrxn1list = []
        for i in range(timesteps):
            print('Integrating ' + str(i) + '/' + str(timesteps))
            Nrf2 = Nrf2list[-1]
            Nrf2 = Nrf2.reshape(1, 1)
            scaler.fit(datascalerlistNrf2)
            Nrf2 = scaler.transform(Nrf2)
            Srxn1 = Srxn1list[-1]
            Srxn1 = Srxn1.reshape(1, 1)
            scaler.fit(datascalerlistSrxn1)
            Srxn1 = scaler.transform(Srxn1)
            tensor = np.zeros((1, 1, 3))
            tensor[0, 0, 0] = timelist[-1]
            tensor[0, 0, 1] = Nrf2[0, 0]
            tensor[0, 0, 2] = Srxn1[0, 0]
            print(tensor)
            predic = model.predict(tensor)
            dSrxn1list.append(predic[0, 0])
            scaler.fit(diffscalerlist)
            predic = scaler.inverse_transform(predic)
            # scaler.fit(datascalerlistSrxn1)
            # predic = scaler.inverse_transform(predic)
            # predic = 10**predic
            nextSrxn1 = Srxn1list[-1]
            scaler.fit(datascalerlistSrxn1)
            nextSrxn1 = nextSrxn1.reshape(1, 1)
            nextSrxn1 = scaler.transform(nextSrxn1)
            nextSrxn1 = nextSrxn1[0, 0] + predic[0]
            nextSrxn1 = nextSrxn1.reshape(1, 1)
            nextSrxn1 = scaler.inverse_transform(nextSrxn1)
            Srxn1list.append(nextSrxn1[0])
            print(predic)
            timelist.append(listt[(i + 1), 0])
            Nrf2list.append(Nrf2_data[(i + 1)])
        # def integration_loss():
        #     integratedSrxn1_diff = np.diff()
        return Srxn1list, dSrxn1list, Nrf2list, timelist

    # gives the index value corresponding to a certain time in the data
    # starttime = desired time, datalen = length of the data, datahours = length of the time course of the data (32 hours)
    def initial_time_index_generator(starttime, datalen, datahours):
        init = int((datalen/datahours) * starttime)
        return init

    # practically the same as the above but with a different name
    # endtime = desired time, datalen = length of the data, datahours = length of the time course of the data (32 hours)
    def final_time_index_generator(endtime, datalen, datahours):
        init = int((datalen/datahours) * endtime)
        return init

    # LSTMdtp integrator with time component
    # Nrf2_data = Nrf2 pDF, y0Srxn1 = initial Srxn1 value, starttime = time (hours) from which Nrf2 data will be read, endtime = time (hours) at which the integration stops
    def ODE_integrator_wtime(Nrf2_data, y0Srxn1, starttime, endtime):
        timelist = [listt[int((2000/32) * starttime) , 0]]
        Nrf2list = [Nrf2_data[int((2000/32) * starttime)]]
        Srxn1list = [y0Srxn1]
        dSrxn1list = []
        for i in range(int((2000/32) * (endtime-starttime))-1):
            print('Integrating ' + str(i) + '/' + str((2000/32) * (endtime-starttime)))
            Nrf2 = Nrf2list[-1]
            Nrf2 = Nrf2.reshape(1, 1)
            scaler.fit(datascalerlistNrf2)
            Nrf2 = scaler.transform(Nrf2)
            Srxn1 = Srxn1list[-1]
            Srxn1 = Srxn1.reshape(1, 1)
            scaler.fit(datascalerlistSrxn1)
            Srxn1 = scaler.transform(Srxn1)
            tensor = np.zeros((1, 1, 3))
            tensor[0, 0, 0] = timelist[-1]
            tensor[0, 0, 1] = Nrf2[0, 0]
            tensor[0, 0, 2] = Srxn1[0, 0]
            print(tensor)
            predic = model.predict(tensor)
            dSrxn1list.append(predic[0, 0])
            scaler.fit(diffscalerlist)
            predic = scaler.inverse_transform(predic)
            # scaler.fit(datascalerlistSrxn1)
            # predic = scaler.inverse_transform(predic)
            # predic = 10**predic
            nextSrxn1 = Srxn1list[-1]
            scaler.fit(datascalerlistSrxn1)
            nextSrxn1 = nextSrxn1.reshape(1, 1)
            nextSrxn1 = scaler.transform(nextSrxn1)
            nextSrxn1 = nextSrxn1[0, 0] + predic[0]
            nextSrxn1 = nextSrxn1.reshape(1, 1)
            nextSrxn1 = scaler.inverse_transform(nextSrxn1)
            Srxn1list.append(nextSrxn1[0])
            print(predic)
            timelist.append(listt[(i + 1 + int((2000/32) * starttime)), 0])
            Nrf2list.append(Nrf2_data[(i + 1 + int((2000/32) * starttime))])
        # def integration_loss():
        #     integratedSrxn1_diff = np.diff()
        return Srxn1list, Nrf2list

    # LSTMdtp integrator without time component, might be the same as ODE_integrator
    def ODE_integrator_notime(Nrf2_data, y0Srxn1, timesteps):
        timelist = [listt[0, 0]]
        Nrf2list = [Nrf2_data[0]]
        Srxn1list = [y0Srxn1]
        dSrxn1list = []
        for i in range(timesteps):
            print('Integrating ' + str(i) + '/' + str(timesteps))
            Nrf2 = Nrf2list[-1]
            Nrf2 = Nrf2.reshape(1, 1)
            scaler.fit(datascalerlistNrf2)
            Nrf2 = scaler.transform(Nrf2)
            Srxn1 = Srxn1list[-1]
            Srxn1 = Srxn1.reshape(1, 1)
            scaler.fit(datascalerlistSrxn1)
            Srxn1 = scaler.transform(Srxn1)
            tensor = np.zeros((1, 1, 2))
            # tensor[0,0,0] = timelist[-1]
            tensor[0, 0, 0] = Nrf2[0, 0]
            tensor[0, 0, 1] = Srxn1[0, 0]
            print(tensor)
            predic = model.predict(tensor)
            dSrxn1list.append(predic[0, 0])
            scaler.fit(diffscalerlist)
            predic = scaler.inverse_transform(predic)
            # scaler.fit(datascalerlistSrxn1)
            # predic = scaler.inverse_transform(predic)
            # predic = 10**predic
            nextSrxn1 = Srxn1list[-1]
            scaler.fit(datascalerlistSrxn1)
            nextSrxn1 = nextSrxn1.reshape(1, 1)
            nextSrxn1 = scaler.transform(nextSrxn1)
            nextSrxn1 = nextSrxn1[0, 0] + predic[0]
            nextSrxn1 = nextSrxn1.reshape(1, 1)
            nextSrxn1 = scaler.inverse_transform(nextSrxn1)
            Srxn1list.append(nextSrxn1[0])
            print(predic)
            timelist.append(listt[(i + 1), 0])
            Nrf2list.append(Nrf2_data[(i + 1)])
        # def integration_loss():
        #     integratedSrxn1_diff = np.diff()
        return Srxn1list, dSrxn1list, Nrf2list, timelist

    # automated plotting for SUL LSTMdtp training
    # collectMSE = True or False, returns list of MSE values if true
    def SUL_integration_plotting(collectMSE):
        # SULintegrated0, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL32[:, 0],
        #                                                            pDF_Srxn1_Rmean_SUL32[0, 0], 1998)
        # SULintegrated1, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL32[:, 1],
        #                                                            pDF_Srxn1_Rmean_SUL32[0, 1], 1998)
        # SULintegrated2, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL32[:, 2],
        #                                                            pDF_Srxn1_Rmean_SUL32[0, 2], 1998)
        # SULintegrated3, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL32[:, 3],
        #                                                            pDF_Srxn1_Rmean_SUL32[0, 3], 1998)
        # SULintegrated4, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL32[:, 4],
        #                                                            pDF_Srxn1_Rmean_SUL32[0, 4], 1998)
        # SULintegrated5, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL32[:, 5],
        #                                                            pDF_Srxn1_Rmean_SUL32[0, 5], 1998)

        SULintegrated0 = ODE_integrator(pDF_Nrf2_Rmean_SUL32[:, 0],
                                                                   pDF_Srxn1_Rmean_SUL32[0, 0], 1998)
        SULintegrated1 = ODE_integrator(pDF_Nrf2_Rmean_SUL32[:, 1],
                                                                   pDF_Srxn1_Rmean_SUL32[0, 1], 1998)
        SULintegrated2 = ODE_integrator(pDF_Nrf2_Rmean_SUL32[:, 2],
                                                                   pDF_Srxn1_Rmean_SUL32[0, 2], 1998)
        SULintegrated3 = ODE_integrator(pDF_Nrf2_Rmean_SUL32[:, 3],
                                                                   pDF_Srxn1_Rmean_SUL32[0, 3], 1998)
        SULintegrated4 = ODE_integrator(pDF_Nrf2_Rmean_SUL32[:, 4],
                                                                   pDF_Srxn1_Rmean_SUL32[0, 4], 1998)
        SULintegrated5 = ODE_integrator(pDF_Nrf2_Rmean_SUL32[:, 5],
                                                                   pDF_Srxn1_Rmean_SUL32[0, 5], 1998)

        plt.plot(SULintegrated0, label='SUL 0.35uM integrated prediction', linestyle='dashed', color='yellow')
        plt.plot(SULintegrated1, label='SUL 0.75uM integrated prediction', linestyle='dashed', color='grey')
        plt.plot(SULintegrated2, label='SUL 1.62uM integrated prediction', linestyle='dashed', color='blue')
        plt.plot(SULintegrated3, label='SUL 3.5uM integrated prediction', linestyle='dashed', color='red')
        plt.plot(SULintegrated4, label='SUL 7.54uM integrated prediction', linestyle='dashed', color='green')
        plt.plot(SULintegrated5, label='SUL 16.25uM integrated prediction', linestyle='dashed', color='purple')

        plt.plot(pDF_Srxn1_Rmean_SUL32[:, 0], label='0.35uM SUL Srxn1 data', color='yellow')
        plt.plot(pDF_Srxn1_Rmean_SUL32[:, 1], label='0.75uM SUL Srxn1 data', color='grey')
        plt.plot(pDF_Srxn1_Rmean_SUL32[:, 2], label='1.62uM SUL Srxn1 data', color='blue')
        plt.plot(pDF_Srxn1_Rmean_SUL32[:, 3], label='3.5uM SUL Srxn1 data', color='red')
        plt.plot(pDF_Srxn1_Rmean_SUL32[:, 4], label='7.54uM SUL Srxn1 data', color='green')
        plt.plot(pDF_Srxn1_Rmean_SUL32[:, 5], label='16.25uM SUL Srxn1 data', color='purple')

        plt.xlabel('datapoints')
        plt.ylabel('value')
        plt.title('Run' + str(run) + ' dydt smoothing test')
        plt.legend()
        plt.savefig('E:/BFW/Master BPS/RP1/Technical/ODE learning tests/run' + str(
            run) + ' seed=' + str(seed) + ' SUL dydt smoothing test.png')
        plt.clf()

        if collectMSE == True: 
            MSElist = np.zeros((6,2))
            MSElist[0,0] = '0.35'
            MSElist[0,1] = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[0:1999, 0]))
            MSElist[1,0] = '0.75'
            MSElist[1,1] = np.log10(mean_squared_error(SULintegrated1, pDF_Srxn1_Rmean_SUL32[0:1999, 1]))
            MSElist[2,0] = '1.62'
            MSElist[2,1] = np.log10(mean_squared_error(SULintegrated2, pDF_Srxn1_Rmean_SUL32[0:1999, 2]))
            MSElist[3,0] = '3.5'
            MSElist[3,1] = np.log10(mean_squared_error(SULintegrated3, pDF_Srxn1_Rmean_SUL32[0:1999, 3]))
            MSElist[4,0] = '7.54'
            MSElist[4,1] = np.log10(mean_squared_error(SULintegrated4, pDF_Srxn1_Rmean_SUL32[0:1999, 4]))
            MSElist[5,0] = '16.25'
            MSElist[5,1] = np.log10(mean_squared_error(SULintegrated5, pDF_Srxn1_Rmean_SUL32[0:1999, 5]))

            # MSE0 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 0]))
            # MSE1 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 1]))
            # MSE2 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 2]))
            # MSE3 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 3]))
            # MSE4 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 4]))
            # MSE5 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 5]))
        return MSElist

    # automated plotting for a SUL trained model with CDDO data as input
    # collectMSE = True or False, returns list of MSE values if true
    def CDDOtoSUL_integration_plotting(collectMSE):
        CDDOintegrated0, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_CDDO32[:, 0],
                                                                   pDF_Srxn1_Rmean_SUL32[0, 0], 1998)
        CDDOintegrated1, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_CDDO32[:, 1],
                                                                   pDF_Srxn1_Rmean_SUL32[0, 1], 1998)
        CDDOintegrated2, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_CDDO32[:, 2],
                                                                   pDF_Srxn1_Rmean_SUL32[0, 2], 1998)
        CDDOintegrated3, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_CDDO32[:, 3],
                                                                   pDF_Srxn1_Rmean_SUL32[0, 3], 1998)
        CDDOintegrated4, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_CDDO32[:, 4],
                                                                   pDF_Srxn1_Rmean_SUL32[0, 4], 1998)
        CDDOintegrated5, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_CDDO32[:, 5],
                                                                   pDF_Srxn1_Rmean_SUL32[0, 5], 1998)

        plt.plot(CDDOintegrated0, label='CDDO 0.01uM integrated prediction', linestyle='dashed', color='yellow')
        plt.plot(CDDOintegrated1, label='CDDO 0.02uM integrated prediction', linestyle='dashed', color='grey')
        plt.plot(CDDOintegrated2, label='CDDO 0.05uM integrated prediction', linestyle='dashed', color='blue')
        plt.plot(CDDOintegrated3, label='CDDO 0.1uM integrated prediction', linestyle='dashed', color='red')
        plt.plot(CDDOintegrated4, label='CDDO 0.22uM integrated prediction', linestyle='dashed', color='green')
        plt.plot(CDDOintegrated5, label='CDDO 0.46uM integrated prediction', linestyle='dashed', color='purple')

        plt.plot(pDF_Srxn1_Rmean_CDDO32[:, 0], label='0.01uM CDDO Srxn1 data', color='yellow')
        plt.plot(pDF_Srxn1_Rmean_CDDO32[:, 1], label='0.02uM CDDO Srxn1 data', color='grey')
        plt.plot(pDF_Srxn1_Rmean_CDDO32[:, 2], label='0.05uM CDDO Srxn1 data', color='blue')
        plt.plot(pDF_Srxn1_Rmean_CDDO32[:, 3], label='0.1uM CDDO Srxn1 data', color='red')
        plt.plot(pDF_Srxn1_Rmean_CDDO32[:, 4], label='0.22uM CDDO Srxn1 data', color='green')
        plt.plot(pDF_Srxn1_Rmean_CDDO32[:, 5], label='0.46uM CDDO Srxn1 data', color='purple')

        plt.xlabel('datapoints')
        plt.ylabel('value')
        plt.title('Run' + str(run) + ' dydt smoothing test')
        plt.legend()
        plt.savefig('E:/BFW/Master BPS/RP1/Technical/ODE learning tests/run' + str(
            run) + ' seed=' + str(seed) + ' CDDO dydt smoothing test.png')
        plt.clf()

        if collectMSE == True: 
            MSElist = np.zeros((6,2))
            MSElist[0,0] = '0.35'
            MSElist[0,1] = np.log10(mean_squared_error(CDDOintegrated0, pDF_Srxn1_Rmean_SUL32[0:1999, 0]))
            MSElist[1,0] = '0.75'
            MSElist[1,1] = np.log10(mean_squared_error(CDDOintegrated1, pDF_Srxn1_Rmean_SUL32[0:1999, 1]))
            MSElist[2,0] = '1.62'
            MSElist[2,1] = np.log10(mean_squared_error(CDDOintegrated2, pDF_Srxn1_Rmean_SUL32[0:1999, 2]))
            MSElist[3,0] = '3.5'
            MSElist[3,1] = np.log10(mean_squared_error(CDDOintegrated3, pDF_Srxn1_Rmean_SUL32[0:1999, 3]))
            MSElist[4,0] = '7.54'
            MSElist[4,1] = np.log10(mean_squared_error(CDDOintegrated4, pDF_Srxn1_Rmean_SUL32[0:1999, 4]))
            MSElist[5,0] = '16.25'
            MSElist[5,1] = np.log10(mean_squared_error(CDDOintegrated5, pDF_Srxn1_Rmean_SUL32[0:1999, 5]))

            # MSE0 = np.log10(mean_squared_error(CDDOintegrated0, pDF_Srxn1_Rmean_SUL32[:, 0]))
            # MSE1 = np.log10(mean_squared_error(CDDOintegrated0, pDF_Srxn1_Rmean_SUL32[:, 1]))
            # MSE2 = np.log10(mean_squared_error(CDDOintegrated0, pDF_Srxn1_Rmean_SUL32[:, 2]))
            # MSE3 = np.log10(mean_squared_error(CDDOintegrated0, pDF_Srxn1_Rmean_SUL32[:, 3]))
            # MSE4 = np.log10(mean_squared_error(CDDOintegrated0, pDF_Srxn1_Rmean_SUL32[:, 4]))
            # MSE5 = np.log10(mean_squared_error(CDDOintegrated0, pDF_Srxn1_Rmean_SUL32[:, 5]))
        return MSElist

    # Automated plotting for ES1 LSTMdtp training
    # collectMSE = True or False, returns list of MSE values if true
    def ES1_integration_plotting(collectMSE):
        integrated0, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_ES1[:, 0],
                                                                   pDF_Srxn1_Rmean_ES1[0, 0], 1998)
        integrated1, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_ES1[:, 1],
                                                                   pDF_Srxn1_Rmean_ES1[0, 1], 1998)
        integrated2, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_ES1[:, 2],
                                                                   pDF_Srxn1_Rmean_ES1[0, 2], 1998)
        integrated3, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_ES1[:, 3],
                                                                   pDF_Srxn1_Rmean_ES1[0, 3], 1998)
        integrated4, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_ES1[:, 4],
                                                                   pDF_Srxn1_Rmean_ES1[0, 4], 1998)
        integrated5, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_ES1[:, 5],
                                                                   pDF_Srxn1_Rmean_ES1[0, 5], 1998)


        plt.plot(integrated0, label='ES1 artificial dose 1 integrated prediction', linestyle='dashed', color='yellow')
        plt.plot(integrated1, label='ES1 artificial dose 2 integrated prediction', linestyle='dashed', color='grey')
        plt.plot(integrated2, label='ES1 artificial dose 3 integrated prediction', linestyle='dashed', color='blue')
        plt.plot(integrated3, label='ES1 artificial dose 4 integrated prediction', linestyle='dashed', color='red')
        plt.plot(integrated4, label='ES1 artificial dose 5 integrated prediction', linestyle='dashed', color='green')
        plt.plot(integrated5, label='ES1 artificial dose 6 integrated prediction', linestyle='dashed', color='purple')

        plt.plot(pDF_Srxn1_Rmean_ES1[:, 0], label='ES1 artificial dose 1 Srxn1 data', color='yellow')
        plt.plot(pDF_Srxn1_Rmean_ES1[:, 1], label='ES1 artificial dose 2 Srxn1 data', color='grey')
        plt.plot(pDF_Srxn1_Rmean_ES1[:, 2], label='ES1 artificial dose 3 Srxn1 data', color='blue')
        plt.plot(pDF_Srxn1_Rmean_ES1[:, 3], label='ES1 artificial dose 4 Srxn1 data', color='red')
        plt.plot(pDF_Srxn1_Rmean_ES1[:, 4], label='ES1 artificial dose 5 Srxn1 data', color='green')
        plt.plot(pDF_Srxn1_Rmean_ES1[:, 5], label='ES1 artificial dose 6 Srxn1 data', color='purple')

        plt.xlabel('datapoints')
        plt.ylabel('value')
        plt.title('Run' + str(run) + ' dydt smoothing test')
        plt.legend()
        plt.savefig('E:/BFW/Master BPS/RP1/Technical/ODE learning tests/run' + str(
            run) + ' seed=' + str(seed) + ' SUL dydt smoothing test.png')
        plt.clf()

        if collectMSE == True: 
            MSElist = np.zeros((6,2))
            MSElist[0,0] = '1'
            MSElist[0,1] = np.log10(mean_squared_error(integrated0, pDF_Srxn1_Rmean_SUL32[0:1999, 0]))
            MSElist[1,0] = '2'
            MSElist[1,1] = np.log10(mean_squared_error(integrated1, pDF_Srxn1_Rmean_SUL32[0:1999, 1]))
            MSElist[2,0] = '3'
            MSElist[2,1] = np.log10(mean_squared_error(integrated2, pDF_Srxn1_Rmean_SUL32[0:1999, 2]))
            MSElist[3,0] = '4'
            MSElist[3,1] = np.log10(mean_squared_error(integrated3, pDF_Srxn1_Rmean_SUL32[0:1999, 3]))
            MSElist[4,0] = '6'
            MSElist[4,1] = np.log10(mean_squared_error(integrated4, pDF_Srxn1_Rmean_SUL32[0:1999, 4]))
            MSElist[5,0] = '7'
            MSElist[5,1] = np.log10(mean_squared_error(integrated5, pDF_Srxn1_Rmean_SUL32[0:1999, 5]))

            # MSE0 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 0]))
            # MSE1 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 1]))
            # MSE2 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 2]))
            # MSE3 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 3]))
            # MSE4 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 4]))
            # MSE5 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 5]))
        return MSElist

    # Automated plotting for ES5 LSTMdtp training
    # collectMSE = True or False, returns list of MSE values if true
    def ES5_integration_plotting(collectMSE):
        integrated0, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_ES5[:, 0],
                                                                   pDF_Srxn1_Rmean_ES5[0, 0], 1998)
        integrated1, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_ES5[:, 1],
                                                                   pDF_Srxn1_Rmean_ES5[0, 1], 1998)
        integrated2, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_ES5[:, 2],
                                                                   pDF_Srxn1_Rmean_ES5[0, 2], 1998)
        integrated3, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_ES5[:, 3],
                                                                   pDF_Srxn1_Rmean_ES5[0, 3], 1998)
        integrated4, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_ES5[:, 5],
                                                                   pDF_Srxn1_Rmean_ES5[0, 5], 1998)
        integrated5, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_ES5[:, 6],
                                                                   pDF_Srxn1_Rmean_ES5[0, 6], 1998)

        plt.plot(integrated0, label='ES5 artificial dose 1 integrated prediction', linestyle='dashed', color='yellow')
        plt.plot(integrated1, label='ES5 artificial dose 2 integrated prediction', linestyle='dashed', color='grey')
        plt.plot(integrated2, label='ES5 artificial dose 3 integrated prediction', linestyle='dashed', color='blue')
        plt.plot(integrated3, label='ES5 artificial dose 4 integrated prediction', linestyle='dashed', color='red')
        plt.plot(integrated4, label='ES5 artificial dose 6 integrated prediction', linestyle='dashed', color='green')
        plt.plot(integrated5, label='ES5 artificial dose 7 integrated prediction', linestyle='dashed', color='purple')

        plt.plot(pDF_Srxn1_Rmean_ES5[:, 0], label='ES5 artificial dose 1 Srxn1 data', color='yellow')
        plt.plot(pDF_Srxn1_Rmean_ES5[:, 1], label='ES5 artificial dose 2 Srxn1 data', color='grey')
        plt.plot(pDF_Srxn1_Rmean_ES5[:, 2], label='ES5 artificial dose 3 Srxn1 data', color='blue')
        plt.plot(pDF_Srxn1_Rmean_ES5[:, 3], label='ES5 artificial dose 4 Srxn1 data', color='red')
        plt.plot(pDF_Srxn1_Rmean_ES5[:, 5], label='ES5 artificial dose 6 Srxn1 data', color='green')
        plt.plot(pDF_Srxn1_Rmean_ES5[:, 6], label='ES5 artificial dose 7 Srxn1 data', color='purple')

        plt.xlabel('datapoints')
        plt.ylabel('value')
        plt.title('Run' + str(run) + ' dydt smoothing test')
        plt.legend()
        plt.savefig('E:/BFW/Master BPS/RP1/Technical/ODE learning tests/run' + str(
            run) + ' seed=' + str(seed) + ' SUL dydt smoothing test.png')
        plt.clf()

        if collectMSE == True: 
            MSElist = np.zeros((6,2))
            MSElist[0,0] = '1'
            MSElist[0,1] = np.log10(mean_squared_error(integrated0, pDF_Srxn1_Rmean_SUL32[0:1999, 0]))
            MSElist[1,0] = '2'
            MSElist[1,1] = np.log10(mean_squared_error(integrated1, pDF_Srxn1_Rmean_SUL32[0:1999, 1]))
            MSElist[2,0] = '3'
            MSElist[2,1] = np.log10(mean_squared_error(integrated2, pDF_Srxn1_Rmean_SUL32[0:1999, 2]))
            MSElist[3,0] = '4'
            MSElist[3,1] = np.log10(mean_squared_error(integrated3, pDF_Srxn1_Rmean_SUL32[0:1999, 3]))
            MSElist[4,0] = '6'
            MSElist[4,1] = np.log10(mean_squared_error(integrated4, pDF_Srxn1_Rmean_SUL32[0:1999, 4]))
            MSElist[5,0] = '7'
            MSElist[5,1] = np.log10(mean_squared_error(integrated5, pDF_Srxn1_Rmean_SUL32[0:1999, 5]))

            # MSE0 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 0]))
            # MSE1 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 1]))
            # MSE2 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 2]))
            # MSE3 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 3]))
            # MSE4 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 4]))
            # MSE5 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 5]))
        return MSElist

    # Finding validation loss minimum (best fit on validation data)
    # run = run number, seeds = how many seeds did the run have, min_epochs = start of epoch range to search in, max_epoch = end of epoch range to search in
    def min_val_loss(run, seeds, min_epochs, max_epochs):
        vallossmatrix = np.zeros((seeds, 2))
        for i in range(seeds):
            epch = int(max_epochs)
            read_run = int(run)
            read_loss = pd.read_csv(
                str(directory) + 'Loss dicts/NN2/val_loss run' + str(
                    read_run) + ' seed=' + str(i) + '.txt')
            np.shape(read_loss)
            read_loss = np.array(read_loss)
            print('seed=' + str(i))
            print(np.amin(read_loss[min_epochs:int(epch - 1), :]))
            print(np.where(read_loss == np.amin(read_loss[min_epochs:int(epch - 1), :])))
            # minlossvalue = np.amin(read_loss[min_epochs:int(epch - 1), 1][0])
            minlossvalue = np.amin(read_loss[min_epochs:int(epch - 1), :])
            print(minlossvalue)
            minlossloc = int(np.where(read_loss == np.amin(read_loss[min_epochs:int(epch - 1), :]))[0])
            vallossmatrix[i, 0] = minlossloc
            vallossmatrix[i, 1] = minlossvalue
            vallossmatrixexport = pd.DataFrame(vallossmatrix)
            # vallossmatrixexport.to_csv(str(directory) + 'validation loss matrix/NN2 loss matrix/run' + str(
            #     run) + '.txt')
        return vallossmatrix

    # Comparing minimum validation losses between runs, used for picking optimal model
    # runlist = list with run numbers that are to be compared (like: [210,211], max = 6),
    # seeds = how many seeds each of these runs had
    # min_epochs = start of epoch range to search in, max_epoch = end of epoch range to search in
    # plot = type of plotting to use, options include 'X', 'Y' and 'Z'
    # 'X'
    def run_comparison(runlist, seeds, minepochs, maxepochs, plot):
        losslistrun1 = min_val_loss(runlist[0], seeds, minepochs, maxepochs)
        losslistrun2 = min_val_loss(runlist[1], seeds, minepochs, maxepochs)
        if len(runlist) >= 3:
            losslistrun3 = min_val_loss(runlist[2], seeds, minepochs, maxepochs)
        if len(runlist) >= 4:
            losslistrun4 = min_val_loss(runlist[3], seeds, minepochs, maxepochs)
        if len(runlist) >= 5:
            losslistrun5 = min_val_loss(runlist[4], seeds, minepochs, maxepochs)
        if len(runlist) >= 6:
            losslistrun6 = min_val_loss(runlist[5], seeds, minepochs, maxepochs)

        collectedY = np.zeros((seeds, len(runlist)))
        collectedY[:, 0] = losslistrun1[:, 1]
        collectedY[:, 1] = losslistrun2[:, 1]
        if len(runlist) >= 3:
            collectedY[:, 2] = losslistrun3[:, 1]
        if len(runlist) >= 4:
            collectedY[:, 3] = losslistrun4[:, 1]
        if len(runlist) >= 5:
            collectedY[:, 4] = losslistrun5[:, 1]

        collectedX = np.zeros((seeds, len(runlist)))
        collectedX[:, 0] = losslistrun1[:, 0]
        collectedX[:, 1] = losslistrun2[:, 0]
        if len(runlist) >= 3:
            collectedX[:, 2] = losslistrun3[:, 0]
        if len(runlist) >= 4:
            collectedX[:, 3] = losslistrun4[:, 0]
        if len(runlist) >= 5:
            collectedX[:, 3] = losslistrun5[:, 0]

        if plot == 'Z':
            labelname = ['run131', 'run132', 'run133', 'run134']
            plotcolors = ['b.', 'r.', 'g.', 'y.']
            for i in range(len(runlist)):
                y = collectedY[:, i]
                x = collectedX[:, i]
                P.plot(x, y, plotcolors[i])
            # plt.legend()
            # plt.ylabel('Lowest validation MAE loss')
            # plt.xlabel('epochs')
            # plt.title('Lowest validation loss comparison run 128, 129 and 130')
            # plt.clf()

        if plot == 'Y':
            plt.boxplot(collectedY, labels=np.array(runlist), manage_ticks=True)
            for i in range(len(runlist)):
                y = collectedY[:, i]
                x = []
                for j in range(int(np.shape(collectedY)[0])):
                    x.append(1 + i)
                P.plot(x, y, 'b.')
            plt.xlabel('Run nr.')
            plt.ylabel('MAE validation loss value')
            # plt.title('Lowest validation loss per seed, run 128, 129 and 130')

        if plot == 'X':
            plt.boxplot(collectedX, labels=np.array(runlist), manage_ticks=True)
            for i in range(len(runlist)):
                y = collectedX[:, i]
                x = []
                for j in range(int(np.shape(collectedX)[0])):
                    x.append(1 + i)
                P.plot(x, y, 'b.')
                plt.xlabel('Run nr.')
                plt.ylabel('epochs')


    # Plotting MSE matrices
    # In case a run saved an 'MSEmatrix'; a matrix with MSE values of multiple dose predictions for multiple different seeds, these matrices can be plotted as a boxplot to compare the performance of different configurations
    #
    def MSEmatrix_comparison(runlist, datatype, nseeds, doselist):
        boxplotlist = np.zeros((nseeds * 6, len(runlist)))
        for k in range(len(runlist)):
            MSEmatrix = pd.read_csv(
                str(directory) + 'MSE matrix/run' + str(runlist[k]) + str(
                    datatype[k]) + ' .txt')
            MSEmatrix = np.array(MSEmatrix)
            MSEmatrix = MSEmatrix[:, 1:]
            print(MSEmatrix)
            for i in range(len(doselist)):
                x = []
                for j in range(nseeds):
                    # x.append(runlist[k])
                    x.append(1 + k)
                y = MSEmatrix[i + 1, 1:(nseeds + 1)]
                # P.plot(x,y, colorlist[i], label=str(doselist[i]) + 'uM SUL')
                if datatype[k] == 'SUL':
                    if doselist[i] == 3.5:
                        plt.scatter(x, y, color=lighten_color('orange', 0 + 0.13 * (i + 1)),
                                    label=str(doselist[i]) + 'uM SUL')
                    else:
                        plt.scatter(x, y, color=lighten_color('g', 0 + 0.13 * (i + 1)),
                                    label=str(doselist[i]) + 'uM SUL')
                if datatype[k] == 'CDDO':
                    plt.scatter(x, y, color=lighten_color('b', 0 + 0.13 * (i + 1)), label=str(doselist[i]) + 'uM SUL')
                if datatype[k] == 'ANDR':
                    plt.scatter(x, y, color=lighten_color('purple', 0 + 0.13 * (i + 1)),
                                label=str(doselist[i]) + 'uM SUL')
                #
                # plt.legend(handles=[plt.patches.Patch(color='#A70022', label='1'),
                #                     plt.patches.Patch(color='#303297', label='5')],
                #            loc='lower right', edgecolor='black', framealpha=1)

            # boxplotlist = []
            # for i in range(3):
            #     for j in range(6):
            #         boxplotlist.append(MSEmatrix[j+1, i+1])
            for i in range(nseeds):
                for j in range(6):
                    boxplotlist[j + 6 * i, k] = (MSEmatrix[j + 1, i + 1])
            print('check1')
        print(np.shape(boxplotlist))
        print(boxplotlist)
        plt.boxplot(boxplotlist, manage_ticks=True, labels=np.array(datatype))
        # plt.text(0.5, -0.02, str(tti(boxplotlist[:,0], boxplotlist[:,1])))
        plt.text(1, -0.5, '* (p-value =' + str(np.around(tti(boxplotlist[:, 0], boxplotlist[:, 1])[1], 3)) + ')')
        if len(runlist) > 2:
            plt.text(0.5, -0.5, '* (pvalue =' + str(tti(boxplotlist[:, 2], boxplotlist[:, 3])[1]) + ')')
        # plt.boxplot(boxplotlist, manage_ticks=True, labels=np.array([k]))
        # plt.colorbar()
        # fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        # plt.legend()
        return

# actual training starts here, don't run these folded sections as a whole!
# 12. Current runs (most recent runs, after a while the most recent runs are moved to the 'run history' section (15))
for h in range(1):
    # SUL dydt smoothing tests
    run = 201
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32, 1501, 5)
    NNconfig = 2
    plt.clf()
    for i in range(3):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        SUL_integration_plotting()
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL dydt smoothing tests
    run = 201
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32, 1501, 5)
    NNconfig = 2
    plt.clf()
    for i in range(3):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        SUL_integration_plotting()
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)

    # ES5 tests
    run = 202
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32, 1501, 5)
    NNconfig = 2
    MSEmatrix = np.zeros((7, 4))
    plt.clf()
    for i in range(3):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(30, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(30, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(30, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(30, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        seedMSE = SUL_integration_plotting(True)
        if i == 0:
            MSEmatrix[1:7, i] = seedMSE[:,0]
        MSEmatrix[0,i+1] = seed
        MSEmatrix[1:7,i+1] = seedMSE[:,1]

    # ES5 tests
    run = 203
    traindoseindexlist = trainindex([1,2,3,4], 'ES5')
    valdoseindexlist = valindex([6,7], 'ES5')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(5.4)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES5, pDF_Srxn1_Rmean_ES5,
                                                     traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES5)
    NNconfig = 2
    # MSEmatrix = np.zeros((7, 4))
    MSEmatrix = np.zeros((7, 4))
    # MSEmatrix = np.zeros((len(traindoseindexlist + valdoseindexlist + 1), 4))
    plt.clf()
    for i in range(3):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        seedMSE = ES5_integration_plotting(True)
        if i == 0:
            MSEmatrix[1:7, i] = seedMSE[:, 0]
        MSEmatrix[0, i + 1] = seed
        MSEmatrix[1:7, i + 1] = seedMSE[:, 1]

    # ES5 tests
    run = 204
    traindoseindexlist = trainindex([1,2,3,4], 'ES5')
    valdoseindexlist = valindex([6,7], 'ES5')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(5.2)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES5, pDF_Srxn1_Rmean_ES5,
                                                     traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES5)
    NNconfig = 2
    # MSEmatrix = np.zeros((7, 4))
    MSEmatrix = np.zeros((7, 4))
    # MSEmatrix = np.zeros((len(traindoseindexlist + valdoseindexlist + 1), 4))
    plt.clf()
    for i in range(3):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        seedMSE = ES5_integration_plotting(True)
        if i == 0:
            MSEmatrix[1:7, i] = seedMSE[:, 0]
        MSEmatrix[0, i + 1] = seed
        MSEmatrix[1:7, i + 1] = seedMSE[:, 1]

    # ES5 tests
    run = 205
    traindoseindexlist = trainindex([1,2,3,4], 'ES5')
    valdoseindexlist = valindex([6,7], 'ES5')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(5.4)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES5, pDF_Srxn1_Rmean_ES5,
                                                     traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES5)
    NNconfig = 2
    # MSEmatrix = np.zeros((7, 4))
    MSEmatrix = np.zeros((7, 4))
    # MSEmatrix = np.zeros((len(traindoseindexlist + valdoseindexlist + 1), 4))
    plt.clf()
    for i in range(3):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        seedMSE = ES5_integration_plotting(True)
        if i == 0:
            MSEmatrix[1:7, i] = seedMSE[:, 0]
        MSEmatrix[0, i + 1] = seed
        MSEmatrix[1:7, i + 1] = seedMSE[:, 1]

    # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # ES5 tests
    run = 206
    traindoseindexlist = trainindex([1,2,3,4], 'ES5')
    valdoseindexlist = valindex([6,7], 'ES5')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(5.2)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES5, pDF_Srxn1_Rmean_ES5,
                                                     traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES5)
    plt.plot(testY[:,0,:])
    plt.clf()
    NNconfig = 2
    # MSEmatrix = np.zeros((7, 4))
    MSEmatrix = np.zeros((7, 4))
    # MSEmatrix = np.zeros((len(traindoseindexlist + valdoseindexlist + 1), 4))
    for i in range(3):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        seedMSE = ES5_integration_plotting(True)
        if i == 0:
            MSEmatrix[1:7, i] = seedMSE[:, 0]
        MSEmatrix[0, i + 1] = seed
        MSEmatrix[1:7, i + 1] = seedMSE[:, 1]

    # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)

    # ES5 tests
    run = 207
    traindoseindexlist = trainindex([1,2,3,4], 'ES1')
    valdoseindexlist = valindex([5,6], 'ES1')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
    plt.plot(testY[:,0,:])
    plt.clf()
    NNconfig = 2
    # MSEmatrix = np.zeros((7, 4))
    MSEmatrix = np.zeros((7, 4))
    # MSEmatrix = np.zeros((len(traindoseindexlist + valdoseindexlist + 1), 4))
    for i in range(3):
        print('seed = ' + str(i+i))
        seed = i+i
        tf.random.set_seed(seed=i+i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        seedMSE = ES1_integration_plotting(True)
        if i == 0:
            MSEmatrix[1:7, i] = seedMSE[:, 0]
        MSEmatrix[0, i + 1] = seed
        MSEmatrix[1:7, i + 1] = seedMSE[:, 1]

    # data training
    run = 208.2
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32,1501, 5)
    plt.clf()
    NNconfig = 2
    # MSEmatrix = np.zeros((7, 4))
    MSEmatrix1 = np.zeros((7, 4))
    MSEmatrix2 = np.zeros((7, 4))

    # MSEmatrix = np.zeros((len(traindoseindexlist + valdoseindexlist + 1), 4))
    for i in range(3):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)

        # MSEmatrix1 = np.array(MSEmatrix1)
        posttraining(model, history)
        seedMSE = SUL_integration_plotting(True)
        if i == 0:
            MSEmatrix1[1:7, i] = seedMSE[:, 0]
        MSEmatrix1[0, i + 1] = seed
        MSEmatrix1[1:7, i + 1] = seedMSE[:, 1]
        np.shape(MSEmatrix1)


        # MSEmatrix2 = np.array(MSEmatrix2)
        seedMSE = CDDOtoSUL_integration_plotting(True)
        if i == 0:
            MSEmatrix2[1:7, i] = seedMSE[:, 0]
        MSEmatrix2[0, i + 1] = seed
        MSEmatrix2[1:7, i + 1] = seedMSE[:, 1]

    MSEmatrix1 = pd.DataFrame(MSEmatrix1)
    MSEmatrix1.to_csv(str(directory) + 'MSE matrix/run' + str(run) + 'SUL .txt')

    MSEmatrix2 = pd.DataFrame(MSEmatrix2)
    MSEmatrix2.to_csv(str(directory) + 'MSE matrix/run' +str(run) + 'CDDO .txt')



    # data training
    run = 209
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32,1501, 5)
    plt.clf()
    NNconfig = 2
    # MSEmatrix = np.zeros((7, 4))
    MSEmatrix1 = np.zeros((7, 16))
    MSEmatrix2 = np.zeros((7, 16))

    # MSEmatrix = np.zeros((len(traindoseindexlist + valdoseindexlist + 1), 4))
    for i in range(15):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)

        # MSEmatrix1 = np.array(MSEmatrix1)
        posttraining(model, history)
        seedMSE = SUL_integration_plotting(True)
        if i == 0:
            MSEmatrix1[1:7, i] = seedMSE[:, 0]
        MSEmatrix1[0, i + 1] = seed
        MSEmatrix1[1:7, i + 1] = seedMSE[:, 1]
        np.shape(MSEmatrix1)


        # MSEmatrix2 = np.array(MSEmatrix2)
        seedMSE = CDDOtoSUL_integration_plotting(True)
        if i == 0:
            MSEmatrix2[1:7, i] = seedMSE[:, 0]
        MSEmatrix2[0, i + 1] = seed
        MSEmatrix2[1:7, i + 1] = seedMSE[:, 1]

    MSEmatrix1 = pd.DataFrame(MSEmatrix1)
    MSEmatrix1.to_csv(str(directory) + 'MSE matrix/run' + str(run) + 'SUL .txt')

    MSEmatrix2 = pd.DataFrame(MSEmatrix2)
    MSEmatrix2.to_csv(str(directory) + 'MSE matrix/run' +str(run) + 'CDDO .txt')



    # data training
    run = 210
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32,1501, 5)
    plt.clf()
    NNconfig = 2
    # MSEmatrix = np.zeros((7, 4))
    MSEmatrix1 = np.zeros((7, 16))
    MSEmatrix2 = np.zeros((7, 16))

    # MSEmatrix = np.zeros((len(traindoseindexlist + valdoseindexlist + 1), 4))
    for i in range(15):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)

        # MSEmatrix1 = np.array(MSEmatrix1)
        posttraining(model, history)
        seedMSE = SUL_integration_plotting(True)
        if i == 0:
            MSEmatrix1[1:7, i] = seedMSE[:, 0]
        MSEmatrix1[0, i + 1] = seed
        MSEmatrix1[1:7, i + 1] = seedMSE[:, 1]
        np.shape(MSEmatrix1)


        # MSEmatrix2 = np.array(MSEmatrix2)
        seedMSE = CDDOtoSUL_integration_plotting(True)
        if i == 0:
            MSEmatrix2[1:7, i] = seedMSE[:, 0]
        MSEmatrix2[0, i + 1] = seed
        MSEmatrix2[1:7, i + 1] = seedMSE[:, 1]

    MSEmatrix1 = pd.DataFrame(MSEmatrix1)
    MSEmatrix1.to_csv(str(directory) + 'MSE matrix/run' + str(run) + 'SUL .txt')

    MSEmatrix2 = pd.DataFrame(MSEmatrix2)
    MSEmatrix2.to_csv(str(directory) + 'MSE matrix/run' +str(run) + 'CDDO .txt')


    # data training
    run = 211
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32,1501, 5)
    plt.clf()
    NNconfig = 2
    # MSEmatrix = np.zeros((7, 4))
    MSEmatrix1 = np.zeros((7, 16))
    MSEmatrix2 = np.zeros((7, 16))

    # MSEmatrix = np.zeros((len(traindoseindexlist + valdoseindexlist + 1), 4))
    for i in range(15):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 2000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)

        # MSEmatrix1 = np.array(MSEmatrix1)
        posttraining(model, history)
        seedMSE = SUL_integration_plotting(True)
        if i == 0:
            MSEmatrix1[1:7, i] = seedMSE[:, 0]
        MSEmatrix1[0, i + 1] = seed
        MSEmatrix1[1:7, i + 1] = seedMSE[:, 1]
        np.shape(MSEmatrix1)


        # MSEmatrix2 = np.array(MSEmatrix2)
        seedMSE = CDDOtoSUL_integration_plotting(True)
        if i == 0:
            MSEmatrix2[1:7, i] = seedMSE[:, 0]
        MSEmatrix2[0, i + 1] = seed
        MSEmatrix2[1:7, i + 1] = seedMSE[:, 1]

    MSEmatrix1 = pd.DataFrame(MSEmatrix1)
    MSEmatrix1.to_csv(str(directory) + 'MSE matrix/run' + str(run) + 'SUL .txt')

    MSEmatrix2 = pd.DataFrame(MSEmatrix2)
    MSEmatrix2.to_csv(str(directory) + 'MSE matrix/run' +str(run) + 'CDDO .txt')


    # data training
    #
    run = 212.1
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'ES1')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.15)
    trainX, testX = trainXfunc2(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32)
    trainY, testY = trainYfunc2(pDF_Srxn1_Rmean_SUL32)
    datascalerlistSrxn1
    NNconfig = 1
    # MSEmatrix = np.zeros((7, 4))
    MSEmatrix1 = np.zeros((7, 16))
    MSEmatrix2 = np.zeros((7, 16))

    # MSEmatrix = np.zeros((len(traindoseindexlist + valdoseindexlist + 1), 4))
    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 2000

        model = Sequential()
        model.add(
            LSTM(20, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(10, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(
            LSTM(1, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(
            LSTM(1, return_sequences=True, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        posttraining(model, history)



        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)

        # MSEmatrix1 = np.array(MSEmatrix1)
        seedMSE = SUL_integration_plotting(True)
        if i == 0:
            MSEmatrix1[1:7, i] = seedMSE[:, 0]
        MSEmatrix1[0, i + 1] = seed
        MSEmatrix1[1:7, i + 1] = seedMSE[:, 1]
        np.shape(MSEmatrix1)


        # MSEmatrix2 = np.array(MSEmatrix2)
        seedMSE = CDDOtoSUL_integration_plotting(True)
        if i == 0:
            MSEmatrix2[1:7, i] = seedMSE[:, 0]
        MSEmatrix2[0, i + 1] = seed
        MSEmatrix2[1:7, i + 1] = seedMSE[:, 1]

    MSEmatrix1 = pd.DataFrame(MSEmatrix1)
    MSEmatrix1.to_csv('E:/BFW/Master BPS/RP1/Technical/thesis plots/NN2/run' + str(run) + 'SUL .txt')

    #
    # MSEmatrix2 = pd.DataFrame(MSEmatrix2)
    # MSEmatrix2.to_csv(str(directory) + 'MSE matrix/run' +str(run) + 'CDDO .txt')
    #

    predictionlist = [0.35, 0.75, 1.62, 3.5, 7.54, 16.25]
    scaler.fit(scalerlist)
    time = scaler.inverse_transform(listt[:, 0].reshape(len(listt), 1))

    testlist = []
    for i in range(2000):
        if time[i] <= 1:
            testlist.append(i)
    print('index=' + str(testlist[-1]) + ' ' + 'time=' +  str(time[testlist[-1]]))


    traindoseindexlist = trainindex(predictionlist, 'SUL')
    trainX, testX = trainXfunc2(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32)
    MSElist = np.zeros((6,2))
    for i in range(len(predictionlist)):
        predic = trainX[i,:,:]
        predic = predic.reshape(1,np.shape(predic)[0], np.shape(predic)[1])
        print(np.shape(predic))
        predic = model.predict(predic)
        predic = predic.reshape(2000,1)
        scaler.fit(datascalerlistSrxn1)
        predic = scaler.inverse_transform(predic)
        if predictionlist[i] != 1.62:
            if predictionlist[i] == 3.5:
                plt.plot(time, predic, label='test prediction ' + str(predictionlist[i]) + 'uM SUL', color=lighten_color('r',0+0.13*(i+1)), linestyle='dashed')
            else:
                plt.plot(time, predic, label='training prediction ' + str(predictionlist[i]) + 'uM SUL', color=lighten_color('g',0+0.13*(i+1)), linestyle='dashed')
        MSE = np.log10(mean_squared_error(predic, data))
        MSElist[i,0] = i+1
        MSElist[i,1] = MSE
        predic = pd.DataFrame(predic)
        # predic.to_csv('E:/BFW/Master BPS/RP1/Technical/thesis plots/NN2/run' + str(run) + ' ' + str(predictionlist[i]) +  'SUL prediction .txt')
        data = pDF_Srxn1_Rmean_SUL32[:,i]
        if predictionlist[i] != 1.62:
            if predictionlist[i] == 3.5:
                plt.plot(time, data, label='test data ' + str(predictionlist[i]) + 'uM SUL', color=lighten_color('r',0+0.13*(i+1)))
            else:
                plt.plot(time, data, label='training data ' + str(predictionlist[i]) + 'uM SUL', color=lighten_color('g',0+0.13*(i+1)))
        data = pd.DataFrame(data)

        # data.to_csv('E:/BFW/Master BPS/RP1/Technical/thesis plots/NN2/run' + str(run) + ' ' + str(predictionlist[i]) +  'SUL data.txt')
    MSElist = pd.DataFrame(MSElist)
    MSElist.to_csv('E:/BFW/Master BPS/RP1/Technical/thesis plots/NN2/1/a/run' + str(run) + ' MSElist.txt')


    # data training
    run = 213
    traindoseindexlist = trainindex([1,2,3,4], 'ES1')
    valdoseindexlist = valindex([5,6], 'ES1')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc2(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc2(pDF_Srxn1_Rmean_ES1)

    datascalerlistSrxn1
    NNconfig = 1
    # MSEmatrix = np.zeros((7, 4))
    MSEmatrix1 = np.zeros((7, 16))
    MSEmatrix2 = np.zeros((7, 16))

    # MSEmatrix = np.zeros((len(traindoseindexlist + valdoseindexlist + 1), 4))
    for i in range(3):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 4000

        model = Sequential()
        model.add(
            LSTM(20, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(10, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(
            LSTM(1, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(
            LSTM(1, return_sequences=True, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        posttraining(model, history)


 # data training
    run = 214
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32,1501, 5)
    plt.clf()
    NNconfig = 2
    # MSEmatrix = np.zeros((7, 4))
    MSEmatrix1 = np.zeros((7, 16))
    MSEmatrix2 = np.zeros((7, 16))

    # MSEmatrix = np.zeros((len(traindoseindexlist + valdoseindexlist + 1), 4))
    for i in range(15):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)

        # MSEmatrix1 = np.array(MSEmatrix1)
        posttraining(model, history)
        seedMSE = SUL_integration_plotting(True)
        if i == 0:
            MSEmatrix1[1:7, i] = seedMSE[:, 0]
        MSEmatrix1[0, i + 1] = seed
        MSEmatrix1[1:7, i + 1] = seedMSE[:, 1]
        np.shape(MSEmatrix1)


        # MSEmatrix2 = np.array(MSEmatrix2)
        seedMSE = CDDOtoSUL_integration_plotting(True)
        if i == 0:
            MSEmatrix2[1:7, i] = seedMSE[:, 0]
        MSEmatrix2[0, i + 1] = seed
        MSEmatrix2[1:7, i + 1] = seedMSE[:, 1]

    MSEmatrix1 = pd.DataFrame(MSEmatrix1)
    MSEmatrix1.to_csv(str(directory) + 'MSE matrix/run' + str(run) + 'SUL .txt')

    MSEmatrix2 = pd.DataFrame(MSEmatrix2)
    MSEmatrix2.to_csv(str(directory) + 'MSE matrix/run' +str(run) + 'CDDO .txt')


 # data training
    run = 215
    traindoseindexlist = trainindex([1,2,3,4], 'ES1')
    valdoseindexlist = valindex([5,6], 'ES1')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc9_Nrf2_to_Srxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc9_Nrf2_to_Srxn1(pDF_Srxn1_Rmean_ES1, traindoseindexlist, valdoseindexlist)
    plt.clf()
    NNconfig = 2
    # MSEmatrix = np.zeros((7, 4))
    MSEmatrix1 = np.zeros((7, 16))
    MSEmatrix2 = np.zeros((7, 16))



    # MSEmatrix = np.zeros((len(traindoseindexlist + valdoseindexlist + 1), 4))
    for i in range(5):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)

        # MSEmatrix1 = np.array(MSEmatrix1)
        posttraining(model, history)
        seedMSE = SUL_integration_plotting(True)
        if i == 0:
            MSEmatrix1[1:7, i] = seedMSE[:, 0]
        MSEmatrix1[0, i + 1] = seed
        MSEmatrix1[1:7, i + 1] = seedMSE[:, 1]
        np.shape(MSEmatrix1)


        # MSEmatrix2 = np.array(MSEmatrix2)
        seedMSE = CDDOtoSUL_integration_plotting(True)
        if i == 0:
            MSEmatrix2[1:7, i] = seedMSE[:, 0]
        MSEmatrix2[0, i + 1] = seed
        MSEmatrix2[1:7, i + 1] = seedMSE[:, 1]

    MSEmatrix1 = pd.DataFrame(MSEmatrix1)
    MSEmatrix1.to_csv(str(directory) + 'MSE matrix/run' + str(run) + 'SUL .txt')

    MSEmatrix2 = pd.DataFrame(MSEmatrix2)
    MSEmatrix2.to_csv(str(directory) + 'MSE matrix/run' +str(run) + 'CDDO .txt')

    # NN1 training
    def NN1_LSTM_plotting(traindoseindexlist, valdoseindexlist):

        figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
        scaler.fit(scalerlist)
        time = scaler.inverse_transform(listt[:, 0].reshape(len(listt), 1))

        predicdf = np.zeros((2000, 6))
        predicdoselist = [0.35, 0.75, 1.62, 3.5, 7.54, 16.25]
        for i in range(6):
            predic, empty = trainXfunc11_NN1_convLSTM([predicdoselist[i]], [0.1])
            predic = model.predict(predic)
            predic = predic.reshape(2000,1)
            scaler.fit(datascalerlistNrf2)
            predic = scaler.inverse_transform(predic)
            MSE = np.around(np.log10(mean_squared_error(predic, pDF_Nrf2_Rmean_SUL32[:,i])),2)
            predicdf[:, i] = predic[:,0]
            print('check1')
            trainIncl = False
            valIncl = False
            for j in range(len(traindoseindexlist)):
                if i == traindoseindexlist[j]:
                    plt.plot(time, predic,
                             color=lighten_color('green', (1 / 6) * (i + 1)),
                             label=str(predicdoselist[i]) + 'uM SUL prediction, log MSE=' + str(MSE),
                             linestyle='dashed')
                    plt.plot(time, pDF_Nrf2_Rmean_SUL32[:, i],
                             color=lighten_color('green', (1 / 6) * (i + 1)),
                             label=str(predicdoselist[i]) + 'uM SUL data')
                    trainIncl = True

            for j in range(len(valdoseindexlist)):
                if i == valdoseindexlist[j]:
                    plt.plot(time, predic,
                             color=lighten_color('orange', (1 / 6) * (i + 1)),
                             label=str(predicdoselist[i]) + 'uM SUL prediction, log MSE=' + str(MSE),
                             linestyle='dashed')
                    plt.plot(time, pDF_Nrf2_Rmean_SUL32[:, i],
                             color=lighten_color('orange', (1 / 6) * (i + 1)),
                             label=str(predicdoselist[i]) + 'uM SUL data')
                    valIncl = True

            if trainIncl == False and valIncl == False:
                plt.plot(time, predic,
                         color=lighten_color('red', (1 / 6) * (i + 1)),
                         label=str(predicdoselist[i]) + 'uM SUL prediction, log MSE=' + str(MSE),
                         linestyle='dashed')
                plt.plot(time, pDF_Nrf2_Rmean_SUL32[:, i],
                         color=lighten_color('red', (1 / 6) * (i + 1)),
                         label=str(predicdoselist[i]) + 'uM SUL data')
        # plt.ylim(0, 0.8)
        plt.xlim(0, 50)
        plt.xlabel('Time (h)')
        plt.ylabel('Log10 value (AU)')
        plt.legend(loc=4)
        plt.tight_layout()
        plt.savefig('E:/BFW/Master BPS/RP1/Technical/final NN1 tests/run' + str(run) + ' seed=' + str(seed) + ' LSTM SUL.png')


    run = 216
    traindoselist = [0.35, 1.62, 7.54]
    valdoselist = [3.5]
    traindoseindexlist = trainindex(traindoselist, 'SUL')
    valdoseindexlist = valindex(valdoselist, 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.16)

    trainX, testX = trainXfunc11_NN1_convLSTM(traindoselist, valdoselist)
    trainY, testY = trainYfunc11_NN1_convLSTM(pDF_Nrf2_Rmean_SUL32, traindoseindexlist, valdoseindexlist)

    for i in range(4):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 4000

        model = Sequential()
        model.add(
            LSTM(20, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(10, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(
            LSTM(1, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(
            LSTM(1, return_sequences=True, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        posttraining(model, history)
        NN1_LSTM_plotting(traindoseindexlist, valdoseindexlist)



    run = 216.1
    traindoselist = [0.35, 1.62, 7.54]
    valdoselist = [3.5]
    traindoseindexlist = trainindex(traindoselist, 'SUL')
    valdoseindexlist = valindex(valdoselist, 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)

    trainX, testX = trainXfunc11_NN1_convLSTM(traindoselist, valdoselist)
    trainY, testY = trainYfunc11_NN1_convLSTM(pDF_Nrf2_Rmean_SUL32, traindoseindexlist, valdoseindexlist)

    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(20, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(10, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(
            LSTM(1, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(
            LSTM(1, return_sequences=True, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        posttraining(model, history)
        NN1_LSTM_plotting(traindoseindexlist, valdoseindexlist)


    run = 216.2
    traindoselist = [0.35, 1.62, 7.54]
    valdoselist = [3.5]
    traindoseindexlist = trainindex(traindoselist, 'SUL')
    valdoseindexlist = valindex(valdoselist, 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.16)

    trainX, testX = trainXfunc11_NN1_convLSTM(traindoselist, valdoselist)
    trainY, testY = trainYfunc11_NN1_convLSTM(pDF_Nrf2_Rmean_SUL32, traindoseindexlist, valdoseindexlist)

    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(20, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(10, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(
            LSTM(1, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(
            LSTM(1, return_sequences=True, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        posttraining(model, history)
        NN1_LSTM_plotting(traindoseindexlist, valdoseindexlist)


    run = 216.4
    traindoselist = [0.35, 1.62, 7.54]
    valdoselist = [3.5]
    traindoseindexlist = trainindex(traindoselist, 'SUL')
    valdoseindexlist = valindex(valdoselist, 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.17)

    trainX, testX = trainXfunc11_NN1_convLSTM(traindoselist, valdoselist)
    trainY, testY = trainYfunc11_NN1_convLSTM(pDF_Nrf2_Rmean_SUL32, traindoseindexlist, valdoseindexlist)

    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        # 216.3: 500, 216.4: 500, 216.5: 4000 (saved as 216.4)
        # total = 5000
        epch = 4000

        model = Sequential()
        model.add(
            LSTM(20, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(10, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(
            LSTM(1, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(
            LSTM(1, return_sequences=True, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        posttraining(model, history)
        NN1_LSTM_plotting(traindoseindexlist, valdoseindexlist)


    run = 217.2
    traindoselist = [0.35, 1.62, 7.54]
    valdoselist = [0.75, 16.25]
    traindoseindexlist = trainindex(traindoselist, 'SUL')
    valdoseindexlist = valindex(valdoselist, 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.17)

    trainX, testX = trainXfunc11_NN1_convLSTM(traindoselist, valdoselist)
    trainY, testY = trainYfunc11_NN1_convLSTM(pDF_Nrf2_Rmean_SUL32, traindoseindexlist, valdoseindexlist)

    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        # 217: 1000, 217.1: 1000 217.2: 1000
        # total = 5000
        epch = 1000

        model = Sequential()
        model.add(
            LSTM(20, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(10, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(
            LSTM(1, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(
            LSTM(1, return_sequences=True, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        posttraining(model, history)
        NN1_LSTM_plotting(traindoseindexlist, valdoseindexlist)


    run = 218.3
    traindoselist = [0.75, 3.5]
    valdoselist = [0.35, 7.54]
    traindoseindexlist = trainindex(traindoselist, 'SUL')
    valdoseindexlist = valindex(valdoselist, 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.17)

    trainX, testX = trainXfunc11_NN1_convLSTM(traindoselist, valdoselist)
    trainY, testY = trainYfunc11_NN1_convLSTM(pDF_Nrf2_Rmean_SUL32, traindoseindexlist, valdoseindexlist)

    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        # 218: 1000, 218.1: 1000 218.2: 1000, 218.3: 2000
        # total = 5000
        epch = 2000

        model = Sequential()
        model.add(
            LSTM(20, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(10, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(
            LSTM(1, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(
            LSTM(1, return_sequences=True, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        posttraining(model, history)
        NN1_LSTM_plotting(traindoseindexlist, valdoseindexlist)

    run = 219
    traindoselist = [0.75, 3.5]
    valdoselist = [0.35, 7.54]
    traindoseindexlist = trainindex(traindoselist, 'SUL')
    valdoseindexlist = valindex(valdoselist, 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.17)

    trainX, testX = trainXfunc11_NN1_convLSTM(traindoselist, valdoselist)
    trainY, testY = trainYfunc11_NN1_convLSTM(pDF_Nrf2_Rmean_SUL32, traindoseindexlist, valdoseindexlist)

    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        # 218: 1000, 218.1: 1000 218.2: 1000, 218.3: 2000
        # total = 5000
        epch = 1000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(
            LSTM(50, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(
            LSTM(1, return_sequences=True, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        posttraining(model, history)
        NN1_LSTM_plotting(traindoseindexlist, valdoseindexlist)


    run = 220
    traindoselist = [1.62, 7.54]
    valdoselist = [0.35]
    traindoseindexlist = trainindex(traindoselist, 'SUL')
    valdoseindexlist = valindex(valdoselist, 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.17)

    trainX, testX = trainXfunc11_NN1_convLSTM(traindoselist, valdoselist)
    trainY, testY = trainYfunc11_NN1_convLSTM(pDF_Nrf2_Rmean_SUL32, traindoseindexlist, valdoseindexlist)

    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        # 218: 1000, 218.1: 1000 218.2: 1000, 218.3: 2000
        # total = 5000
        epch = 5000

        model = Sequential()
        model.add(
            LSTM(20, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(10, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(
            LSTM(1, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(
            LSTM(1, return_sequences=True, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        posttraining(model, history)
        NN1_LSTM_plotting(traindoseindexlist, valdoseindexlist)




    run = 221
    traindoselist = [0.35, 1.62]
    valdoselist = []
    traindoseindexlist = trainindex(traindoselist, 'SUL')
    valdoseindexlist = valindex(valdoselist, 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.17)

    trainX, testX = trainXfunc11_NN1_convLSTM(traindoselist, valdoselist)
    trainY, testY = trainYfunc11_NN1_convLSTM(pDF_Nrf2_Rmean_SUL32, traindoseindexlist, valdoseindexlist)

    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        # 218: 1000, 218.1: 1000 218.2: 1000, 218.3: 2000
        # total = 5000
        epch = 5000

        model = Sequential()
        model.add(
            LSTM(20, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(10, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(
            LSTM(1, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.1))
        model.add(
            LSTM(1, return_sequences=True, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        posttraining(model, history)
        NN1_LSTM_plotting(traindoseindexlist, valdoseindexlist)

    def NN1_tp_plotting(traindoseindexlist, valdoseindexlist):

        figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
        scaler.fit(scalerlist)
        time = scaler.inverse_transform(listt[:, 0].reshape(len(listt), 1))

        predicdf = np.zeros((2000, 6))
        predicdoselist = [0.35, 0.75, 1.62, 3.5, 7.54, 16.25]
        for i in range(6):
            predic, empty = trainXfunc10_NN1_tp([predicdoselist[i]], [0.1])
            predic = model.predict(predic)
            predic = predic.reshape(2000,1)
            scaler.fit(datascalerlistNrf2)
            predic = scaler.inverse_transform(predic)
            MSE = np.around(np.log10(mean_squared_error(predic, pDF_Nrf2_Rmean_SUL32[:,i])),2)
            predicdf[:, i] = predic[:,0]
            print('check1')
            trainIncl = False
            valIncl = False
            for j in range(len(traindoseindexlist)):
                if i == traindoseindexlist[j]:
                    plt.plot(time, predic,
                             color=lighten_color('green', (1 / 6) * (i + 1)),
                             label=str(predicdoselist[i]) + 'uM SUL prediction, log MSE=' + str(MSE),
                             linestyle='dashed')
                    plt.plot(time, pDF_Nrf2_Rmean_SUL32[:, i],
                             color=lighten_color('green', (1 / 6) * (i + 1)),
                             label=str(predicdoselist[i]) + 'uM SUL data')
                    trainIncl = True

            for j in range(len(valdoseindexlist)):
                if i == valdoseindexlist[j]:
                    plt.plot(time, predic,
                             color=lighten_color('orange', (1 / 6) * (i + 1)),
                             label=str(predicdoselist[i]) + 'uM SUL prediction, log MSE=' + str(MSE),
                             linestyle='dashed')
                    plt.plot(time, pDF_Nrf2_Rmean_SUL32[:, i],
                             color=lighten_color('orange', (1 / 6) * (i + 1)),
                             label=str(predicdoselist[i]) + 'uM SUL data')
                    valIncl = True

            if trainIncl == False and valIncl == False:
                plt.plot(time, predic,
                         color=lighten_color('red', (1 / 6) * (i + 1)),
                         label=str(predicdoselist[i]) + 'uM SUL prediction, log MSE=' + str(MSE),
                         linestyle='dashed')
                plt.plot(time, pDF_Nrf2_Rmean_SUL32[:, i],
                         color=lighten_color('red', (1 / 6) * (i + 1)),
                         label=str(predicdoselist[i]) + 'uM SUL data')
        # plt.ylim(0, 0.8)
        plt.xlim(0, 50)
        plt.xlabel('Time (h)')
        plt.ylabel('Log10 value (AU)')
        plt.legend(loc=4)
        plt.tight_layout()
        plt.savefig('E:/BFW/Master BPS/RP1/Technical/final NN1 tests/run' + str(run) + ' seed=' + str(seed) + ' tp SUL.png')


    run = 222
    traindoselist = [0.35, 1.62]
    valdoselist = [0.75]
    traindoseindexlist = trainindex(traindoselist, 'SUL')
    valdoseindexlist = valindex(valdoselist, 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.17)

    trainX, testX = trainXfunc10_NN1_tp(traindoselist, valdoselist)
    trainY, testY = trainYfunc10_NN1_tp(pDF_Nrf2_Rmean_SUL32, traindoseindexlist, valdoseindexlist)

    for i in range(3):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 5000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        posttraining(model, history)
        NN1_tp_plotting(traindoseindexlist, valdoseindexlist)


    run = 222
    traindoselist = [0.35, 1.62]
    valdoselist = [7.54]
    traindoseindexlist = trainindex(traindoselist, 'SUL')
    valdoseindexlist = valindex(valdoselist, 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.17)

    trainX, testX = trainXfunc10_NN1_tp(traindoselist, valdoselist)
    trainY, testY = trainYfunc10_NN1_tp(pDF_Nrf2_Rmean_SUL32, traindoseindexlist, valdoseindexlist)

    for i in range(3):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 3000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        posttraining(model, history)
        NN1_tp_plotting(traindoseindexlist, valdoseindexlist)


    run = 223
    traindoselist = [0.75, 3.5]
    valdoselist = [7.54]
    traindoseindexlist = trainindex(traindoselist, 'SUL')
    valdoseindexlist = valindex(valdoselist, 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.17)

    trainX, testX = trainXfunc10_NN1_tp(traindoselist, valdoselist)
    trainY, testY = trainYfunc10_NN1_tp(pDF_Nrf2_Rmean_SUL32, traindoseindexlist, valdoseindexlist)

    for i in range(3):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 3000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        posttraining(model, history)
        NN1_tp_plotting(traindoseindexlist, valdoseindexlist)


    run = 224
    traindoselist = [1.62, 7.54]
    valdoselist = [3.5]
    traindoseindexlist = trainindex(traindoselist, 'SUL')
    valdoseindexlist = valindex(valdoselist, 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.17)

    trainX, testX = trainXfunc10_NN1_tp(traindoselist, valdoselist)
    trainY, testY = trainYfunc10_NN1_tp(pDF_Nrf2_Rmean_SUL32, traindoseindexlist, valdoseindexlist)

    for i in range(3):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 3000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        posttraining(model, history)
        NN1_tp_plotting(traindoseindexlist, valdoseindexlist)

# 13. Reverse differentiation tests
for h in range(1):
    def trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(df_Srxn1, wlength, poly_order):
            traindata = {}
            for i in range(len(traindoseindexlist)):
                traindata['trainy_Srxn1_{0}'.format(i)] = df_Srxn1[:, traindoseindexlist[i]]
            traindata = pd.DataFrame.from_dict(traindata)
            traindata = np.array(traindata)
            scaler.fit(datascalerlistSrxn1)
            for i in range(np.shape(traindata)[1]):
                vector = traindata[:,i]
                vector = vector.reshape(len(vector),1)
                vector = scaler.transform(vector)
                traindata[:,i] = vector[:,0]
            valdata = {}
            for i in range(len(valdoseindexlist)):
                valdata['trainy_Srxn1_{0}'.format(i)] = df_Srxn1[:, valdoseindexlist[i]]
            valdata = pd.DataFrame.from_dict(valdata)
            valdata = np.array(valdata)
            # scaler.fit(datascalerlistSrxn1)
            # for i in range(np.shape(valdata)[1]):
            #     vector = valdata[:,i]
            #     vector = vector.reshape(len(vector),1)
            #     vector = scaler.transform(vector)
            #     valdata[:,i] = vector[:,0]
            trainy = np.zeros((1999*np.shape(traindata)[1],1,1))
            # if training_set == 1:
            #     diffscalerlist = np.array([-0.001,0.005])
            # if training_set == 2:
            #     diffscalerlist = np.array([-0.001,0.020])
            # diffscalerlist = diffscalerlist.reshape(2,1)
            diffscaler = scaler.fit(diffscalerlist)
            for i in range(np.shape(traindata)[1]):
                # trainy[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
                # trainy[int(1999*i):int(1999*(i+1)), 0, 0] = traindata[0:1999, i]
                print('check' + str(i+1) + '.1')
                diff = traindata[: , i]
                diff = np.diff(diff)
                # plt.plot(diff, label='before smoothing')
                diff = savgol_filter(diff, wlength, poly_order)
                # diff = savgol_filter(diff, 1501, 5)
                # plt.plot(diff, label='after smoothing')
                diff = diff.reshape(len(diff),1)
                diff = diffscaler.transform(diff)
                trainy[int(1999*i):int(1999*(i+1)), 0, 0] = diff[:,0]
                print('check' + str(i+1) + '.2')
            valy = np.zeros((1999*np.shape(valdata)[1],1,1))
            for i in range(np.shape(valdata)[1]):
                # valy[int(1999*i):int(1999*(i+1)), 0, 0] = listt[:-1, 0]
                # valy[int(1999*i):int(1999*(i+1)), 0, 0] = valdata[0:1999, i]
                diff = valdata[: , i]
                diff = np.diff(diff)
                diff = diff.reshape(len(diff), 1)
                diff = diffscaler.transform(diff)
                valy[int(1999*i):int(1999*(i+1)), 0, 0] = diff[:,0]
                print('check' + str(i+1) + '.4')
            return trainy, valy

# 14. Final results plotting (results as shown in thesis)
for h in range(1):
    # Tests
    # NN1 LSTMcv tests (fig 3)
    for h in range(1):
        model = load_model(str(directory) + 'Versions/NN2/run216 seed=0.h5')
        run = 216
        seed = 0
        traindoselist = [0.35, 1.62, 7.54]
        valdoselist = [3.5]
        traindoseindexlist = trainindex(traindoselist, 'SUL')
        valdoseindexlist = valindex(valdoselist, 'SUL')
        NN1_LSTM_plotting(traindoseindexlist, valdoseindexlist)

    # NN1 LSTMtp tests (fig 4)
    for h in range(1):
        def NN1_tp_plotting_seplegend(traindoseindexlist, valdoseindexlist):

            figure(num=None, figsize=(7, 6), dpi=80, facecolor='w', edgecolor='k')
            scaler.fit(scalerlist)
            time = scaler.inverse_transform(listt[:, 0].reshape(len(listt), 1))

            predicdf = np.zeros((2000, 6))
            predicdoselist = [0.35, 0.75, 1.62, 3.5, 7.54, 16.25]
            for i in range(6):
                predic, empty = trainXfunc10_NN1_tp([predicdoselist[i]], [0.1])
                predic = model.predict(predic)
                predic = predic.reshape(2000,1)
                scaler.fit(datascalerlistNrf2)
                predic = scaler.inverse_transform(predic)
                MSE = np.around(np.log10(mean_squared_error(predic, pDF_Nrf2_Rmean_SUL32[:,i])),2)
                predicdf[:, i] = predic[:,0]
                print('check1')
                trainIncl = False
                valIncl = False
                for j in range(len(traindoseindexlist)):
                    if i == traindoseindexlist[j]:
                        plt.plot(time, predic,
                                 color=lighten_color('green', (1 / 6) * (i + 1)),
                                 label=str(predicdoselist[i]) + 'uM SUL prediction, log MSE=' + str(MSE),
                                 linestyle='dashed')
                        plt.plot(time, pDF_Nrf2_Rmean_SUL32[:, i],
                                 color=lighten_color('green', (1 / 6) * (i + 1)),
                                 label=str(predicdoselist[i]) + 'uM SUL data')
                        trainIncl = True

                for j in range(len(valdoseindexlist)):
                    if i == valdoseindexlist[j]:
                        plt.plot(time, predic,
                                 color=lighten_color('orange', (1 / 6) * (i + 1)),
                                 label=str(predicdoselist[i]) + 'uM SUL prediction, log MSE=' + str(MSE),
                                 linestyle='dashed')
                        plt.plot(time, pDF_Nrf2_Rmean_SUL32[:, i],
                                 color=lighten_color('orange', (1 / 6) * (i + 1)),
                                 label=str(predicdoselist[i]) + 'uM SUL data')
                        valIncl = True

                if trainIncl == False and valIncl == False:
                    plt.plot(time, predic,
                             color=lighten_color('red', (1 / 6) * (i + 1)),
                             label=str(predicdoselist[i]) + 'uM SUL prediction, log MSE=' + str(MSE),
                             linestyle='dashed')
                    plt.plot(time, pDF_Nrf2_Rmean_SUL32[:, i],
                             color=lighten_color('red', (1 / 6) * (i + 1)),
                             label=str(predicdoselist[i]) + 'uM SUL data')
            # plt.ylim(0, 0.8)
            # plt.xlim(0, 50)
            plt.xlabel('Time (h)')
            plt.ylabel('Log10 value (AU)')
            plt.tight_layout()
            plt.savefig('E:/BFW/Master BPS/RP1/Technical/final NN1 tests/seperate legend/run' + str(run) + ' seed=' + str(seed) + ' tp SUL.png')

            figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
            for i in range(6):
                predic, empty = trainXfunc10_NN1_tp([predicdoselist[i]], [0.1])
                predic = model.predict(predic)
                predic = predic.reshape(2000, 1)
                scaler.fit(datascalerlistNrf2)
                predic = scaler.inverse_transform(predic)
                MSE = np.around(np.log10(mean_squared_error(predic, pDF_Nrf2_Rmean_SUL32[:, i])), 2)
                predicdf[:, i] = predic[:, 0]
                print('check1')
                trainIncl = False
                valIncl = False
                for j in range(len(traindoseindexlist)):
                    if i == traindoseindexlist[j]:
                        plt.plot(time, predic,
                                 color=lighten_color('green', (1 / 6) * (i + 1)),
                                 label=str(predicdoselist[i]) + 'uM SUL prediction',
                                 linestyle='dashed')
                        plt.plot(time, pDF_Nrf2_Rmean_SUL32[:, i],
                                 color=lighten_color('green', (1 / 6) * (i + 1)),
                                 label=str(predicdoselist[i]) + 'uM SUL data')
                        trainIncl = True

                for j in range(len(valdoseindexlist)):
                    if i == valdoseindexlist[j]:
                        plt.plot(time, predic,
                                 color=lighten_color('orange', (1 / 6) * (i + 1)),
                                 label=str(predicdoselist[i]) + 'uM SUL prediction',
                                 linestyle='dashed')
                        plt.plot(time, pDF_Nrf2_Rmean_SUL32[:, i],
                                 color=lighten_color('orange', (1 / 6) * (i + 1)),
                                 label=str(predicdoselist[i]) + 'uM SUL data')
                        valIncl = True

                if trainIncl == False and valIncl == False:
                    plt.plot(time, predic,
                             color=lighten_color('red', (1 / 6) * (i + 1)),
                             label=str(predicdoselist[i]) + 'uM SUL prediction',
                             linestyle='dashed')
                    plt.plot(time, pDF_Nrf2_Rmean_SUL32[:, i],
                             color=lighten_color('red', (1 / 6) * (i + 1)),
                             label=str(predicdoselist[i]) + 'uM SUL data')
            plt.xlim(0, 80)
            plt.legend(loc=4)
            plt.savefig('E:/BFW/Master BPS/RP1/Technical/final NN1 tests/seperate legend/run' + str(run) + ' seed=' + str(seed) + ' tp SUL legend.png')

        datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.17)
        scaler.fit(scalerlist)
        time = scaler.inverse_transform(listt[:, 0].reshape(len(listt), 1))

        model = load_model(str(directory) + 'Versions/NN2/run222 seed=2 loss=10^-2.432.h5')
        run = 222
        seed = 2
        traindoselist = [0.35, 1.62]
        valdoselist = []
        traindoseindexlist = trainindex(traindoselist, 'SUL')
        valdoseindexlist = valindex(valdoselist, 'SUL')
        NN1_tp_plotting_seplegend(traindoseindexlist, valdoseindexlist)

        model = load_model(str(directory) + 'Versions/NN2/run223 seed=2 loss=10^-2.449.h5')
        run = 223
        seed = 2
        traindoselist = [0.75, 3.5]
        valdoselist = []
        traindoseindexlist = trainindex(traindoselist, 'SUL')
        valdoseindexlist = valindex(valdoselist, 'SUL')
        NN1_tp_plotting_seplegend(traindoseindexlist, valdoseindexlist)

        model = load_model(str(directory) + 'Versions/NN2/run224 seed=2 loss=10^-1.792.h5')
        run = 224
        seed = 2
        traindoselist = [1.62, 7.54]
        valdoselist = []
        traindoseindexlist = trainindex(traindoselist, 'SUL')
        valdoseindexlist = valindex(valdoselist, 'SUL')
        NN1_tp_plotting_seplegend(traindoseindexlist, valdoseindexlist)

    #NN1 limited data tests (fig 5)
    for h in range(1):
        def limitedp_tests(tplist, dose, doseindex):
            predictions = np.zeros((len(tplist), 3))
            for i in range(len(tplist)):
                predicdoselist = [0.35, 0.75, 1.62, 3.5, 7.54, 16.25]
                print('check1')
                trainX, empty = trainXfunc10_NN1_tp([dose], [0.1])
                tpindex = np.where(np.around(time, 1) == tplist[i])[0][0]
                print('check2')
                print(tpindex)
                # tp = listt[tpindex,0]
                tp = trainX[tpindex, 0, 0]
                concdp = trainX[tpindex, 0, 1]
                print('iter' + str(i) + ', ' + str(tplist[i]) + 'h')
                input = np.zeros((1, 1, 2))
                input[0, 0, 0] = tp
                input[0, 0, 1] = concdp
                predic = model.predict(input)
                scaler.fit(datascalerlistNrf2)
                predic = scaler.inverse_transform(predic)
                predictions[i, 0] = tplist[i]
                predictions[i, 1] = predic
                predictions[i, 2] = pDF_Nrf2_Rmean_ES1[tpindex, doseindex]
            return predictions, trainX


        tppredics, trainX = limitedp_tests([7.5, 15.5, 23.5, 31], 3.5, 3)

        # Dose 4 ES1 prediction
        predic = model.predict(trainX)
        scaler.fit(datascalerlistNrf2)
        predic = scaler.inverse_transform(predic)
        # MSE = np.around(np.log10(mean_squared_error(trainY[(0+(doseindex*2000)):(2000+ 2000*doseindex),0,0], predic[:,0])), 2)
        # MSE = np.around(np.log10(mean_squared_error(pDF_Srxn1_Rmean_ES1[:, i + 4], predic[:, 0])), 2)

        MSE = np.around(np.log10(mean_squared_error(pDF_Srxn1_Rmean_ES1[:, 3], predic[:, 0])), 2)
        figure(num=None, figsize=(7, 6), dpi=80, facecolor='w', edgecolor='k')
        plt.scatter(tppredics[:, 0], tppredics[:, 1], color=lighten_color('red', 1),
                    label='Single time point predictions')
        plt.plot(time, pDF_Nrf2_Rmean_SUL32[:, 3],
                 # label='Dose ' + str(4) + ' Srxn1 data',
                 label='Nrf2 data',
                 color=lighten_color('blue', 1))
        plt.plot(time, predic[:, 0],
                 # label='Dose ' + str(4) + ' Srxn1 prediction, log MSE=' + str(MSE),
                 label='Nrf2 prediction, log MSE=' + str(MSE),
                 color=lighten_color('blue', 1),
                 linestyle='dashed')
        # plt.ylim(-1.5, 4)
        # plt.xlim(0, 32)
        plt.xlabel('Time (h)')
        plt.ylabel('Log10 value (AU)')
        plt.legend(loc=4)
        plt.tight_layout()

    #NN2 LSTMcv test (fig 6):
    for h in range(1):
        run = 212.1
        traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'ES1')
        valdoseindexlist = valindex([3.5], 'SUL')
        datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.15)
        trainX, testX = trainXfunc2(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32)
        trainY, testY = trainYfunc2(pDF_Srxn1_Rmean_SUL32)
        NNconfig = 1
        seedMSE = SUL_integration_plotting(True)

    #NN2 LSTMcv Equation set 1/ODE data test (fig 8)
    for h in range(1):
        scaler.fit(scalerlist)
        time = scaler.inverse_transform(listt[:, 0].reshape(2000, 1))
        # figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
        figure(num=None, figsize=(8, 7), dpi=80, facecolor='w', edgecolor='k')
        for i in range(np.shape(pDF_Nrf2_Rmean_ES1)[1]):
            if i <4:
                plt.plot(time, pDF_Nrf2_Rmean_ES1[:, i], label='dose ' + str(i+1) + ' Nrf2 data',color=lighten_color('green', (1 / 4) * (i+1)))
            else:
                plt.plot(time, pDF_Nrf2_Rmean_ES1[:, i], label='dose ' + str(i+1) + ' Nrf2 data',color=lighten_color('orange', (1 / 2) * (i-3)))
        plt.xlabel('Time (h)')
        plt.ylabel('Value (AU)')
        plt.tight_layout()
        plt.legend(loc=1)
        plt.ylim(0,1500)

        figure(num=None, figsize=(8, 7), dpi=80, facecolor='w', edgecolor='k')
        for i in range(np.shape(pDF_Srxn1_Rmean_ES1)[1]):
            if i <4:
                plt.plot(time, pDF_Srxn1_Rmean_ES1[:, i], label='dose ' + str(i+1) + ' Srxn1 data',color=lighten_color('green', (1 / 4) * (i+1)))
            else:
                plt.plot(time, pDF_Srxn1_Rmean_ES1[:, i], label='dose ' + str(i+1) + ' Srxn1 data',color=lighten_color('orange', (1 / 2) * (i-3)))
        plt.xlabel('Time (h)')
        plt.ylabel('Value (AU)')
        plt.tight_layout()
        plt.legend(loc=1)
        plt.ylim(-1,5)
    #
        figure(num=None, figsize=(8, 7), dpi=80, facecolor='w', edgecolor='k')
        for i in range(np.shape(pDF_Srxn1_Rmean_ES1)[1]):
            if i <4:
                predic = trainX[i,:,:]
                predic = predic.reshape(1,2000,2)
                predic = model.predict(predic)
                predic = predic.reshape(2000,1)
                scaler.fit(datascalerlistSrxn1)
                predic = scaler.inverse_transform(predic)
                plt.plot(time, pDF_Srxn1_Rmean_ES1[:, i], label='dose ' + str(i+1) + ' Srxn1 data',color=lighten_color('green', (1 / 4) * (i+1)))
                plt.plot(time, predic[:,:], label='dose ' + str(i+1) + ' Srxn1 prediction',color=lighten_color('green', (1 / 4) * (i+1)), linestyle ='dashed')
        plt.xlabel('Time (h)')
        plt.ylabel('Value (AU)')
        plt.legend(loc=1)
        plt.ylim(-1,5.5)
        plt.tight_layout()


        figure(num=None, figsize=(8, 7), dpi=80, facecolor='w', edgecolor='k')
        for i in range(2):
            predic = testX[i,:,:]
            predic = predic.reshape(1,2000,2)
            predic = model.predict(predic)
            predic = predic.reshape(2000,1)
            scaler.fit(datascalerlistSrxn1)
            predic = scaler.inverse_transform(predic)
            plt.plot(time, pDF_Srxn1_Rmean_ES1[:, i+4], label='dose ' + str(i + 5) + ' Srxn1 data',
                     color=lighten_color('orange', (1 / 2) * (i+1)))
            plt.plot(time, predic[:,:], label='dose ' + str(i+5) + ' Srxn1 prediction', color=lighten_color('orange', (1 / 2) * (i+1)), linestyle ='dashed')

        plt.xlabel('Time (h)')
        plt.ylabel('Value (AU)')
        plt.legend(loc=1)
        plt.ylim(-1,4.5)
        plt.tight_layout()

    #NN2 LSTMcv time testing (fig 9)
    for h in range(1):
        model = load_model(str(directory) + 'Versions/NN2/run213 seed=2.h5')
        scaler.fit(scalerlist)
        time = scaler.inverse_transform(listt[:, 0].reshape(len(listt), 1))
        def convlstm_timetesting2(starttime, endtime, dose, prediccolor):
            for i in range(len(starttime)):
                print(starttime)
                firstindex = np.where(time >= starttime[i])[0][0]
                lastindex = np.where(time <= endtime[i])[0][-1]
                ttvector = listt[firstindex:(lastindex+1), 0]
                ttdatalen = len(listt[firstindex:(lastindex+1), 0])
                datavector = trainX[(dose-1), firstindex:(lastindex+1),1]
                tensor = np.zeros((1, ttdatalen, 2))
                tensor[:,:,0] = ttvector
                tensor[:,:,1] = datavector
                # plt.plot(datavector)
                predic = model.predict(tensor)
                predic = predic.reshape(ttdatalen, 1)
                scaler.fit(datascalerlistSrxn1)
                predic = scaler.inverse_transform(predic)
                data = pDF_Srxn1_Rmean_ES1[firstindex:(lastindex+1), (dose-1)]
                plt.plot(time[firstindex:(lastindex+1), 0], data, label='ES1 dose ' + str(dose) + ' data ' + str(starttime[i]) + '-' + str(endtime[i]) + 'h',
                         color=lighten_color('blue', 0 + 1))
                plt.plot(time[firstindex:(lastindex+1), 0], predic, label='ES1 dose ' + str(dose) + ' prediction ' + str(starttime[i]) + '-' + str(endtime[i]) + 'h',
                         color=lighten_color(prediccolor, (1/len(starttime))*(i+1)), linestyle='dashed')
            return
        def convlstm_timetesting3(starttime, endtime, dose, plottype):
            firstindex = np.where(time >= starttime)[0][0]
            lastindex = np.where(time <= endtime)[0][-1]
            ttvector = listt[firstindex:(lastindex + 1), 0]
            ttdatalen = len(listt[firstindex:(lastindex + 1), 0])
            datavector = trainX[(dose - 1), firstindex:(lastindex + 1), 1]
            tensor = np.zeros((1, ttdatalen, 2))
            tensor[:, :, 0] = ttvector
            tensor[:, :, 1] = datavector
            # plt.plot(datavector)
            predic = model.predict(tensor)
            predic = predic.reshape(ttdatalen, 1)
            scaler.fit(datascalerlistSrxn1)
            predic = scaler.inverse_transform(predic)
            data = pDF_Srxn1_Rmean_ES1[firstindex:(lastindex + 1), (dose - 1)]
            if plottype == 'data':
                plt.plot(time[firstindex:(lastindex + 1), 0], data,
                         label='Data ' + str(starttime) + '-' + str(endtime) + 'h',
                         color=lighten_color('blue', 0 + 1))
            if plottype == 'predic':
                plt.plot(time[firstindex:(lastindex + 1), 0], predic,
                         label='Prediction ' + str(starttime) + '-' + str(endtime) + 'h',
                         color=lighten_color('red', 0 + 1), linestyle='dashed')
            return
        def convlstm_timetesting4(starttime, endtime, dose, plottype,  prediccolor='purple', datacolvalue=0.5, linewidth=[0.5], linestyle=['dashed']):
            for i in range(len(starttime)):
                print(starttime)
                firstindex = np.where(time >= starttime[i])[0][0]
                lastindex = np.where(time <= endtime[i])[0][-1]
                ttvector = listt[firstindex:(lastindex+1), 0]
                ttdatalen = len(listt[firstindex:(lastindex+1), 0])
                datavector = trainX[(dose-1), firstindex:(lastindex+1),1]
                tensor = np.zeros((1, ttdatalen, 2))
                tensor[:,:,0] = ttvector
                tensor[:,:,1] = datavector
                # plt.plot(datavector)
                predic = model.predict(tensor)
                predic = predic.reshape(ttdatalen, 1)
                scaler.fit(datascalerlistSrxn1)
                predic = scaler.inverse_transform(predic)
                data = pDF_Srxn1_Rmean_ES1[firstindex:(lastindex+1), (dose-1)]
                if len(linestyle) >1:
                    if plottype == 'data':
                        plt.plot(time[firstindex:(lastindex + 1), 0], data,
                                 label='Data ' + str(starttime[i]) + '-' + str(endtime[i]) + 'h',
                                 color=lighten_color('blue', datacolvalue), linewidth=linewidth)
                    if plottype == 'predic':
                        plt.plot(time[firstindex:(lastindex + 1), 0], predic,
                                 label='Prediction ' + str(starttime[i]) + '-' + str(endtime[i]) + 'h',
                                 color=lighten_color(prediccolor, (1 / len(starttime)) * (i + 1)),linestyle=linestyle[i], linewidth=linewidth[i])
                else:
                    if plottype == 'data':
                        plt.plot(time[firstindex:(lastindex + 1), 0], data,
                                 label='Data ' + str(starttime[i]) + '-' + str(endtime[i]) + 'h',
                                 color=lighten_color('blue', datacolvalue), linewidth=linewidth)
                    if plottype == 'predic':
                        plt.plot(time[firstindex:(lastindex + 1), 0], predic,
                                 label='Prediction ' + str(starttime[i]) + '-' + str(endtime[i]) + 'h',
                                 color=lighten_color(prediccolor, (1 / len(starttime)) * (i + 1)),linestyle=linestyle[0], linewidth=linewidth[i])
            return

        traindoseindexlist = trainindex([1,2,3,4], 'ES1')
        valdoseindexlist = valindex([5,6], 'ES1')
        datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
        trainX, testX = trainXfunc2(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        trainY, testY = trainYfunc2(pDF_Srxn1_Rmean_ES1)

        # figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
        figure(num=None, figsize=(7, 6), dpi=80, facecolor='w', edgecolor='k')
        convlstm_timetesting4([0],[32],3,'data', datacolvalue=0.4, linewidth=8)
        # convlstm_timetesting4([0,0,0],[32,16,8],3, 'predic', linewidth=[3,3,3], linestyle=['dashdot','dashed','dotted'])
        linetype = 3
        convlstm_timetesting4([0,0,0],[32,16,8],3, 'predic', linewidth=[3,3,3], linestyle=[(0,(linetype,1)),(0,(linetype,1.5)),(0,(linetype,2))],prediccolor='purple')
        plt.ylim(-1.5,4)
        plt.xlabel('Time (h)')
        plt.ylabel('Log10 value (AU)')
        plt.legend(loc=4)
        plt.tight_layout()

        # figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
        figure(num=None, figsize=(7, 6), dpi=80, facecolor='w', edgecolor='k')
        convlstm_timetesting4([0],[32],3,'data', datacolvalue=0.4, linewidth=8)
        # convlstm_timetesting4([0,0,0],[32,16,8],3, 'predic', linewidth=[3,3,3], linestyle=['dashdot','dashed','dotted'])
        linetype = 3
        convlstm_timetesting4([0,3,8,16],[32,32,32,32],3, 'predic', linewidth=[3,3,3,3], linestyle=[(0,(linetype,1)),(0,(linetype,1.5)),(0,(linetype,2)),(0,(linetype,2.5))], prediccolor='purple')
        plt.ylim(-1.5,4)
        plt.xlabel('Time (h)')
        plt.ylabel('Log10 value (AU)')
        plt.legend(loc=4)
        plt.tight_layout()


        # figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
        figure(num=None, figsize=(7, 6), dpi=80, facecolor='w', edgecolor='k')
        convlstm_timetesting4([0],[32],3,'data', datacolvalue=0.4, linewidth=10)
        convlstm_timetesting4([0,0,0],[32,16,8],3, 'predic', linewidth=[3,2,1])
        plt.ylim(-1.5,4)
        plt.xlabel('Time (h)')
        plt.ylabel('Log10 value (AU)')
        plt.legend(loc=4)
        plt.tight_layout()

        convlstm_timetesting3([0,0,0],[32,16,8],3, 'purple')
        convlstm_timetesting2([0,3,8,16],[32,32,32,32],3, 'purple')
        convlstm_timetesting2([0,3,8,16],[32,32,32,32],3, 'purple')

    #NN2 LSTMtp test (time, Nrf2 > Srxn1) (fig 10)
    for h in range(1):
        model = load_model(str(directory) + 'Versions/NN2/run215 seed=4.h5')
        scaler.fit(scalerlist)
        time = scaler.inverse_transform(listt[:, 0].reshape(len(listt), 1))

        traindoseindexlist = trainindex([1,2,3,4], 'ES1')
        valdoseindexlist = valindex([5,6], 'ES1')
        datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
        trainX, testX = trainXfunc9_Nrf2_to_Srxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1, traindoseindexlist, valdoseindexlist)


        # Run 215 last seed
        figure(num=None, figsize=(7, 6), dpi=80, facecolor='w', edgecolor='k')
        for i in range(4):
            datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
            doseindex = i
            predic = model.predict(trainX[(0+(doseindex*2000)):(2000+ 2000*doseindex),:,:])
            scaler.fit(datascalerlistSrxn1)
            predic = scaler.inverse_transform(predic)
            # MSE = np.around(np.log10(mean_squared_error(trainY[(0+(doseindex*2000)):(2000+ 2000*doseindex),0,0], predic[:,0])), 2)
            MSE = np.around(np.log10(mean_squared_error(pDF_Srxn1_Rmean_ES1[:,i], predic[:,0])), 2)

            plt.plot(time, predic[:,0],
                     label='Dose ' + str(i+1) + ' Srxn1 prediction, log MSE=' + str(MSE),
                     color=lighten_color('green', (1/4)*(i+1)),
                     linestyle='dashed')
            # plt.plot(time, trainY[(0+(doseindex*2000)):(2000+ 2000*doseindex),0,0],
            #          label='Dose ' + str(i + 1) + ' Srxn1 data',
            #          color=lighten_color('green', (1 / 4) * (i + 1)),)
            plt.plot(time, pDF_Srxn1_Rmean_ES1[:,i],
                     label='Dose ' + str(i + 1) + ' Srxn1 data',
                     color=lighten_color('green', (1 / 4) * (i + 1)),)

            # plt.ylim(-0.1, 2.5)
        plt.xlabel('Time (h)')
        plt.ylabel('Log10 value (AU)')
        plt.legend(loc=4)
        plt.tight_layout()


        figure(num=None, figsize=(7, 6), dpi=80, facecolor='w', edgecolor='k')
        for i in range(2):
            datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
            doseindex = i
            predic = model.predict(testX[(0+(doseindex*2000)):(2000+ 2000*doseindex),:,:])
            scaler.fit(datascalerlistSrxn1)
            predic = scaler.inverse_transform(predic)
            # MSE = np.around(np.log10(mean_squared_error(trainY[(0+(doseindex*2000)):(2000+ 2000*doseindex),0,0], predic[:,0])), 2)
            MSE = np.around(np.log10(mean_squared_error(pDF_Srxn1_Rmean_ES1[:,i+4], predic[:,0])), 2)

            plt.plot(time, predic[:,0],
                     label='Dose ' + str(i+5) + ' Srxn1 prediction, log MSE=' + str(MSE),
                     color=lighten_color('orange', (1/2)*(i+1)),
                     linestyle='dashed')
            # plt.plot(time, trainY[(0+(doseindex*2000)):(2000+ 2000*doseindex),0,0],
            #          label='Dose ' + str(i + 1) + ' Srxn1 data',
            #          color=lighten_color('green', (1 / 4) * (i + 1)),)
            plt.plot(time, pDF_Srxn1_Rmean_ES1[:,i+4],
                     label='Dose ' + str(i + 5) + ' Srxn1 data',
                     color=lighten_color('orange', (1 / 2) * (i + 1)),)

            # plt.ylim(-0.1, 2.5)
        plt.xlabel('Time (h)')
        plt.ylabel('Log10 value (AU)')
        plt.legend(loc=4)
        plt.tight_layout()

        # Limited data points tests, ES1 dose 4,  15.5, 23.5, 31 hours (fig 11)
        for h in range(1):
            def limitedp_tests(tplist, dose):
                predictions = np.zeros((len(tplist),3))
                for i in range(len(tplist)):
                    tpindex = np.where(np.around(time,1) == tplist[i])[0][0]
                    print(tpindex)
                    # tp = listt[tpindex,0]
                    tp = trainX[tpindex + 2000*(dose-1), 0, 0]
                    Nrf2dp = trainX[tpindex + 2000*(dose-1), 0, 1]
                    print('iter' + str(i) + ', ' + str(tplist[i]) + 'h')
                    input = np.zeros((1,1,2))
                    input[0,0,0] = tp
                    input[0,0,1] = Nrf2dp
                    predic = model.predict(input)
                    scaler.fit(datascalerlistSrxn1)
                    predic = scaler.inverse_transform(predic)
                    predictions[i,0] = tplist[i]
                    predictions[i,1] = predic
                    predictions[i,2] = pDF_Srxn1_Rmean_ES1[tpindex,(dose-1)]
                return predictions
            tppredics = limitedp_tests([7.5, 15.5, 23.5, 31],4)

            # Dose 4 ES1 prediction
            predic = model.predict(trainX[(0 + (3 * 2000)):(2000 + 2000 * 3), :, :])
            scaler.fit(datascalerlistSrxn1)
            predic = scaler.inverse_transform(predic)
            # MSE = np.around(np.log10(mean_squared_error(trainY[(0+(doseindex*2000)):(2000+ 2000*doseindex),0,0], predic[:,0])), 2)
            # MSE = np.around(np.log10(mean_squared_error(pDF_Srxn1_Rmean_ES1[:, i + 4], predic[:, 0])), 2)


            MSE = np.around(np.log10(mean_squared_error(pDF_Srxn1_Rmean_ES1[:, 3], predic[:, 0])), 2)
            figure(num=None, figsize=(7, 6), dpi=80, facecolor='w', edgecolor='k')
            plt.scatter(tppredics[:,0], tppredics[:,1], color=lighten_color('red',1),
                        label='Single time point predictions')
            plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 3],
                     # label='Dose ' + str(4) + ' Srxn1 data',
                     label='Srxn1 data',
                     color=lighten_color('green', 1))
            plt.plot(time, predic[:, 0],
                     # label='Dose ' + str(4) + ' Srxn1 prediction, log MSE=' + str(MSE),
                     label='Srxn1 prediction, log MSE=' + str(MSE),
                     color=lighten_color('green', 1),
                     linestyle='dashed')
            plt.ylim(-1.5, 4)
            plt.xlim(0, 32)
            plt.xlabel('Time (h)')
            plt.ylabel('Log10 value (AU)')
            plt.legend(loc=4)
            plt.tight_layout()


    # LSTMdtp tests (time, Nrf2, Srxn1 > dSrxn1 (fig 12)
    for h in range(1):
        model = load_model(str(directory) + 'Versions/NN2/run207 seed=1.h5')

        scaler.fit(scalerlist)
        time = scaler.inverse_transform(listt[0:1999, 0].reshape(1999, 1))
        time
        traindoseindexlist = trainindex([1,2,3,4], 'ES1')
        valdoseindexlist = valindex([5,6], 'ES1')
        datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
        trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1, traindoseindexlist, valdoseindexlist)
        trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)

        def ODE_integrator(Nrf2_data, y0Srxn1, timesteps):
                timelist = [listt[0, 0]]
                Nrf2list = [Nrf2_data[0]]
                Srxn1list = [y0Srxn1]
                dSrxn1list = []
                for i in range(timesteps):
                    print('Integrating ' + str(i) + '/' + str(timesteps))
                    Nrf2 = Nrf2list[-1]
                    Nrf2 = Nrf2.reshape(1, 1)
                    scaler.fit(datascalerlistNrf2)
                    Nrf2 = scaler.transform(Nrf2)
                    Srxn1 = Srxn1list[-1]
                    Srxn1 = Srxn1.reshape(1, 1)
                    scaler.fit(datascalerlistSrxn1)
                    Srxn1 = scaler.transform(Srxn1)
                    tensor = np.zeros((1, 1, 3))
                    tensor[0, 0, 0] = timelist[-1]
                    tensor[0, 0, 1] = Nrf2[0, 0]
                    tensor[0, 0, 2] = Srxn1[0, 0]
                    print(tensor)
                    predic = model.predict(tensor)
                    dSrxn1list.append(predic[0, 0])
                    scaler.fit(diffscalerlist)
                    predic = scaler.inverse_transform(predic)
                    # scaler.fit(datascalerlistSrxn1)
                    # predic = scaler.inverse_transform(predic)
                    # predic = 10**predic
                    nextSrxn1 = Srxn1list[-1]
                    scaler.fit(datascalerlistSrxn1)
                    nextSrxn1 = nextSrxn1.reshape(1, 1)
                    nextSrxn1 = scaler.transform(nextSrxn1)
                    nextSrxn1 = nextSrxn1[0, 0] + predic[0]
                    nextSrxn1 = nextSrxn1.reshape(1, 1)
                    nextSrxn1 = scaler.inverse_transform(nextSrxn1)
                    Srxn1list.append(nextSrxn1[0])
                    print(predic)
                    timelist.append(listt[(i + 1), 0])
                    Nrf2list.append(Nrf2_data[(i + 1)])
                # def integration_loss():
                #     integratedSrxn1_diff = np.diff()
                return Srxn1list, dSrxn1list, Nrf2list, timelist

        integrated0, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_ES1[:, 0],
                                                                pDF_Srxn1_Rmean_ES1[0, 0], 1998)
        integrated1, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_ES1[:, 1],
                                                                pDF_Srxn1_Rmean_ES1[0, 1], 1998)
        integrated2, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_ES1[:, 2],
                                                                pDF_Srxn1_Rmean_ES1[0, 2], 1998)
        integrated3, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_ES1[:, 3],
                                                                pDF_Srxn1_Rmean_ES1[0, 3], 1998)
        integrated4, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_ES1[:, 4],
                                                                pDF_Srxn1_Rmean_ES1[0, 4], 1998)
        integrated5, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_ES1[:, 5],
                                                                pDF_Srxn1_Rmean_ES1[0, 5], 1998)

        def ES1_integration_plotting(collectMSE, plotlist):
            MSE1 = np.around(np.log10(mean_squared_error(integrated0, pDF_Srxn1_Rmean_ES1[:,0])), 2)
            MSE2 = np.around(np.log10(mean_squared_error(integrated1, pDF_Srxn1_Rmean_ES1[:,1])), 2)
            MSE3 = np.around(np.log10(mean_squared_error(integrated2, pDF_Srxn1_Rmean_ES1[:,2])), 2)
            MSE4 = np.around(np.log10(mean_squared_error(integrated3, pDF_Srxn1_Rmean_ES1[:,3])), 2)
            MSE5 = np.around(np.log10(mean_squared_error(integrated4, pDF_Srxn1_Rmean_ES1[:,4])), 2)
            MSE6 = np.around(np.log10(mean_squared_error(integrated5, pDF_Srxn1_Rmean_ES1[:,5])), 2)

            plt.plot(time, integrated0, label='ES1 dose 1 Srxn1 prediction', color=lighten_color('green', (1/4)*1), linestyle='dashed')
            plt.plot(time, integrated1, label='ES1 dose 2 Srxn1 prediction', color=lighten_color('green', (1/4)*2), linestyle='dashed')
            plt.plot(time, integrated2, label='ES1 dose 3 Srxn1 prediction', color=lighten_color('green', (1/4)*3), linestyle='dashed')
            plt.plot(time, integrated3, label='ES1 dose 4 Srxn1 prediction', color=lighten_color('green', (1/4)*4), linestyle='dashed')
            plt.plot(time, integrated4, label='ES1 dose 5 Srxn1 prediction', color=lighten_color('yellow', (1/2)*1), linestyle='dashed')
            plt.plot(time, integrated5, label='ES1 dose 6 Srxn1 prediction', color=lighten_color('yellow', (1/2)*1), linestyle='dashed')

            plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 0], label='ES1 dose 1 Srxn1 data', color=lighten_color('green', (1/4)*1))
            plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 1], label='ES1 dose 2 Srxn1 data', color=lighten_color('green', (1/4)*2))
            plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 2], label='ES1 dose 3 Srxn1 data', color=lighten_color('green', (1/4)*3))
            plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 3], label='ES1 dose 4 Srxn1 data', color=lighten_color('green', (1/4)*4))
            plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 4], label='ES1 dose 5 Srxn1 data', color=lighten_color('yellow', (1/2)*1))
            plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 5], label='ES1 dose 6 Srxn1 data', color=lighten_color('yellow', (1/2)*1))

            plt.xlabel('Time (h)')
            plt.ylabel('Value (AU)')
            plt.legend()
            # plt.savefig('E:/BFW/Master BPS/RP1/Technical/ODE learning tests/run' + str(
            #     run) + ' seed=' + str(seed) + ' SUL dydt smoothing test.png')
            # plt.clf()

            if collectMSE == True:
                MSElist = np.zeros((6, 2))
                MSElist[0, 0] = '1'
                MSElist[0, 1] = np.log10(mean_squared_error(integrated0, pDF_Srxn1_Rmean_SUL32[0:1999, 0]))
                MSElist[1, 0] = '2'
                MSElist[1, 1] = np.log10(mean_squared_error(integrated1, pDF_Srxn1_Rmean_SUL32[0:1999, 1]))
                MSElist[2, 0] = '3'
                MSElist[2, 1] = np.log10(mean_squared_error(integrated2, pDF_Srxn1_Rmean_SUL32[0:1999, 2]))
                MSElist[3, 0] = '4'
                MSElist[3, 1] = np.log10(mean_squared_error(integrated3, pDF_Srxn1_Rmean_SUL32[0:1999, 3]))
                MSElist[4, 0] = '6'
                MSElist[4, 1] = np.log10(mean_squared_error(integrated4, pDF_Srxn1_Rmean_SUL32[0:1999, 4]))
                MSElist[5, 0] = '7'
                MSElist[5, 1] = np.log10(mean_squared_error(integrated5, pDF_Srxn1_Rmean_SUL32[0:1999, 5]))

                # MSE0 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 0]))
                # MSE1 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 1]))
                # MSE2 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 2]))
                # MSE3 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 3]))
                # MSE4 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 4]))
                # MSE5 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 5]))
            return MSElist
        def ES1_integration_plotting2(collectMSE, plotlist):
            for i in range(len(plotlist)):
                if plotlist[i] == 1:
                    MSE1 = np.around(np.log10(mean_squared_error(integrated0, pDF_Srxn1_Rmean_ES1[0:1999, 0])), 2)
                    plt.plot(time, integrated0, label='ES1 dose 1 Srxn1 prediction, log MSE=' + str(MSE1), color=lighten_color('green', (1 / 4) * 1), linestyle='dashed')
                    plt.plot(time, pDF_Srxn1_Rmean_ES1[0:1999, 0], label='ES1 dose 1 Srxn1 data', color=lighten_color('green', (1 / 4) * 1))
                if plotlist[i] == 2:
                    MSE2 = np.around(np.log10(mean_squared_error(integrated1, pDF_Srxn1_Rmean_ES1[0:1999:, 1])), 2)
                    plt.plot(time, integrated1, label='ES1 dose 2 Srxn1 prediction, log MSE=' + str(MSE2), color=lighten_color('green', (1 / 4) * 2), linestyle='dashed')
                    plt.plot(time, pDF_Srxn1_Rmean_ES1[0:1999:, 1], label='ES1 dose 2 Srxn1 data', color=lighten_color('green', (1 / 4) * 2))
                if plotlist[i] == 3:
                    MSE3 = np.around(np.log10(mean_squared_error(integrated2, pDF_Srxn1_Rmean_ES1[0:1999:, 2])), 2)
                    plt.plot(time, integrated2, label='ES1 dose 3 Srxn1 prediction, log MSE=' + str(MSE3), color=lighten_color('green', (1 / 4) * 3), linestyle='dashed')
                    plt.plot(time, pDF_Srxn1_Rmean_ES1[0:1999:, 2], label='ES1 dose 3 Srxn1 data', color=lighten_color('green', (1 / 4) * 3))
                if plotlist[i] == 4:
                    MSE4 = np.around(np.log10(mean_squared_error(integrated3, pDF_Srxn1_Rmean_ES1[0:1999:, 3])), 2)
                    plt.plot(time, integrated3, label='ES1 dose 4 Srxn1 prediction, log MSE=' + str(MSE4), color=lighten_color('green', (1 / 4) * 4), linestyle='dashed')
                    plt.plot(time, pDF_Srxn1_Rmean_ES1[0:1999:, 3], label='ES1 dose 4 Srxn1 data', color=lighten_color('green', (1 / 4) * 4))
                if plotlist[i] == 5:
                    MSE5 = np.around(np.log10(mean_squared_error(integrated4, pDF_Srxn1_Rmean_ES1[0:1999:, 4])), 2)
                    plt.plot(time, integrated4, label='ES1 dose 5 Srxn1 prediction, log MSE=' + str(MSE5), color=lighten_color('orange', (1 / 2) * 1), linestyle='dashed')
                    plt.plot(time, pDF_Srxn1_Rmean_ES1[0:1999:, 4], label='ES1 dose 5 Srxn1 data', color=lighten_color('orange', (1 / 2) * 1))
                if plotlist[i] == 6:
                    MSE6 = np.around(np.log10(mean_squared_error(integrated5, pDF_Srxn1_Rmean_ES1[0:1999:, 5])), 2)
                    plt.plot(time, integrated5, label='ES1 dose 6 Srxn1 prediction, log MSE=' + str(MSE6), color=lighten_color('orange', (1 / 2) * 2), linestyle='dashed')
                    plt.plot(time, pDF_Srxn1_Rmean_ES1[0:1999:, 5], label='ES1 dose 6 Srxn1 data', color=lighten_color('orange', (1 / 2) * 2))
                #
                #
                # MSE2 = np.around(np.log10(mean_squared_error(integrated1, pDF_Srxn1_Rmean_ES1[:, 1])), 2)
                # MSE3 = np.around(np.log10(mean_squared_error(integrated2, pDF_Srxn1_Rmean_ES1[:, 2])), 2)
                # MSE4 = np.around(np.log10(mean_squared_error(integrated3, pDF_Srxn1_Rmean_ES1[:, 3])), 2)
                # MSE5 = np.around(np.log10(mean_squared_error(integrated4, pDF_Srxn1_Rmean_ES1[:, 4])), 2)
                # MSE6 = np.around(np.log10(mean_squared_error(integrated5, pDF_Srxn1_Rmean_ES1[:, 5])), 2)
                #
                # plt.plot(time, integrated0, label='ES1 dose 1 Srxn1 prediction', color=lighten_color('green', (1 / 4) * 1),
                #          linestyle='dashed')
                # plt.plot(time, integrated1, label='ES1 dose 2 Srxn1 prediction', color=lighten_color('green', (1 / 4) * 2),
                #          linestyle='dashed')
                # plt.plot(time, integrated2, label='ES1 dose 3 Srxn1 prediction', color=lighten_color('green', (1 / 4) * 3),
                #          linestyle='dashed')
                # plt.plot(time, integrated3, label='ES1 dose 4 Srxn1 prediction', color=lighten_color('green', (1 / 4) * 4),
                #          linestyle='dashed')
                # plt.plot(time, integrated4, label='ES1 dose 5 Srxn1 prediction', color=lighten_color('yellow', (1 / 2) * 1),
                #          linestyle='dashed')
                # plt.plot(time, integrated5, label='ES1 dose 6 Srxn1 prediction', color=lighten_color('yellow', (1 / 2) * 1),
                #          linestyle='dashed')
                #
                # plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 0], label='ES1 dose 1 Srxn1 data',
                #          color=lighten_color('green', (1 / 4) * 1))
                # plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 1], label='ES1 dose 2 Srxn1 data',
                #          color=lighten_color('green', (1 / 4) * 2))
                # plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 2], label='ES1 dose 3 Srxn1 data',
                #          color=lighten_color('green', (1 / 4) * 3))
                # plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 3], label='ES1 dose 4 Srxn1 data',
                #          color=lighten_color('green', (1 / 4) * 4))
                # plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 4], label='ES1 dose 5 Srxn1 data',
                #          color=lighten_color('yellow', (1 / 2) * 1))
                # plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 5], label='ES1 dose 6 Srxn1 data',
                #          color=lighten_color('yellow', (1 / 2) * 1))
                #
                plt.legend()
                plt.xlabel('Time (h)')
                plt.ylabel('Log10 value (AU)')
                # plt.savefig('E:/BFW/Master BPS/RP1/Technical/ODE learning tests/run' + str(
                #     run) + ' seed=' + str(seed) + ' SUL dydt smoothing test.png')
                # plt.clf()

                if collectMSE == True:
                    MSElist = np.zeros((6, 2))
                    MSElist[0, 0] = '1'
                    MSElist[0, 1] = np.log10(mean_squared_error(integrated0, pDF_Srxn1_Rmean_SUL32[0:1999, 0]))
                    MSElist[1, 0] = '2'
                    MSElist[1, 1] = np.log10(mean_squared_error(integrated1, pDF_Srxn1_Rmean_SUL32[0:1999, 1]))
                    MSElist[2, 0] = '3'
                    MSElist[2, 1] = np.log10(mean_squared_error(integrated2, pDF_Srxn1_Rmean_SUL32[0:1999, 2]))
                    MSElist[3, 0] = '4'
                    MSElist[3, 1] = np.log10(mean_squared_error(integrated3, pDF_Srxn1_Rmean_SUL32[0:1999, 3]))
                    MSElist[4, 0] = '6'
                    MSElist[4, 1] = np.log10(mean_squared_error(integrated4, pDF_Srxn1_Rmean_SUL32[0:1999, 4]))
                    MSElist[5, 0] = '7'
                    MSElist[5, 1] = np.log10(mean_squared_error(integrated5, pDF_Srxn1_Rmean_SUL32[0:1999, 5]))

                    # MSE0 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 0]))
                    # MSE1 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 1]))
                    # MSE2 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 2]))
                    # MSE3 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 3]))
                    # MSE4 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 4]))
                    # MSE5 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 5]))
            return
        def ES1_integration_plotting3(collectMSE, plotlist):
            for i in range(len(plotlist)):
                if plotlist[i] == 1:
                    MSE1 = np.around(np.log10(mean_squared_error(integrated0, pDF_Srxn1_Rmean_ES1[0:1999, 0])), 2)
                    plt.plot(time, integrated0, label='Dose 1 Srxn1 prediction, log MSE=' + str(MSE1), color=lighten_color('green', (1 / 4) * 1), linestyle='dashed')
                    plt.plot(time, pDF_Srxn1_Rmean_ES1[0:1999, 0], label='Dose 1 Srxn1 data', color=lighten_color('green', (1 / 4) * 1))
                if plotlist[i] == 2:
                    MSE2 = np.around(np.log10(mean_squared_error(integrated1, pDF_Srxn1_Rmean_ES1[0:1999:, 1])), 2)
                    plt.plot(time, integrated1, label='Dose 2 Srxn1 prediction, log MSE=' + str(MSE2), color=lighten_color('green', (1 / 4) * 2), linestyle='dashed')
                    plt.plot(time, pDF_Srxn1_Rmean_ES1[0:1999:, 1], label='Dose 2 Srxn1 data', color=lighten_color('green', (1 / 4) * 2))
                if plotlist[i] == 3:
                    MSE3 = np.around(np.log10(mean_squared_error(integrated2, pDF_Srxn1_Rmean_ES1[0:1999:, 2])), 2)
                    plt.plot(time, integrated2, label='Dose 3 Srxn1 prediction, log MSE=' + str(MSE3), color=lighten_color('green', (1 / 4) * 3), linestyle='dashed')
                    plt.plot(time, pDF_Srxn1_Rmean_ES1[0:1999:, 2], label='Dose 3 Srxn1 data', color=lighten_color('green', (1 / 4) * 3))
                if plotlist[i] == 4:
                    MSE4 = np.around(np.log10(mean_squared_error(integrated3, pDF_Srxn1_Rmean_ES1[0:1999:, 3])), 2)
                    plt.plot(time, integrated3, label='Dose 4 Srxn1 prediction, log MSE=' + str(MSE4), color=lighten_color('green', (1 / 4) * 4), linestyle='dashed')
                    plt.plot(time, pDF_Srxn1_Rmean_ES1[0:1999:, 3], label='Dose 4 Srxn1 data', color=lighten_color('green', (1 / 4) * 4))
                if plotlist[i] == 5:
                    MSE5 = np.around(np.log10(mean_squared_error(integrated4, pDF_Srxn1_Rmean_ES1[0:1999:, 4])), 2)
                    plt.plot(time, integrated4, label='Dose 5 Srxn1 prediction, log MSE=' + str(MSE5), color=lighten_color('orange', (1 / 2) * 1), linestyle='dashed')
                    plt.plot(time, pDF_Srxn1_Rmean_ES1[0:1999:, 4], label='Dose 5 Srxn1 data', color=lighten_color('orange', (1 / 2) * 1))
                if plotlist[i] == 6:
                    MSE6 = np.around(np.log10(mean_squared_error(integrated5, pDF_Srxn1_Rmean_ES1[0:1999:, 5])), 2)
                    plt.plot(time, integrated5, label='Dose 6 Srxn1 prediction, log MSE=' + str(MSE6), color=lighten_color('orange', (1 / 2) * 2), linestyle='dashed')
                    plt.plot(time, pDF_Srxn1_Rmean_ES1[0:1999:, 5], label='Dose 6 Srxn1 data', color=lighten_color('orange', (1 / 2) * 2))
                #
                #
                # MSE2 = np.around(np.log10(mean_squared_error(integrated1, pDF_Srxn1_Rmean_ES1[:, 1])), 2)
                # MSE3 = np.around(np.log10(mean_squared_error(integrated2, pDF_Srxn1_Rmean_ES1[:, 2])), 2)
                # MSE4 = np.around(np.log10(mean_squared_error(integrated3, pDF_Srxn1_Rmean_ES1[:, 3])), 2)
                # MSE5 = np.around(np.log10(mean_squared_error(integrated4, pDF_Srxn1_Rmean_ES1[:, 4])), 2)
                # MSE6 = np.around(np.log10(mean_squared_error(integrated5, pDF_Srxn1_Rmean_ES1[:, 5])), 2)
                #
                # plt.plot(time, integrated0, label='Dose 1 Srxn1 prediction', color=lighten_color('green', (1 / 4) * 1),
                #          linestyle='dashed')
                # plt.plot(time, integrated1, label='Dose 2 Srxn1 prediction', color=lighten_color('green', (1 / 4) * 2),
                #          linestyle='dashed')
                # plt.plot(time, integrated2, label='Dose 3 Srxn1 prediction', color=lighten_color('green', (1 / 4) * 3),
                #          linestyle='dashed')
                # plt.plot(time, integrated3, label='Dose 4 Srxn1 prediction', color=lighten_color('green', (1 / 4) * 4),
                #          linestyle='dashed')
                # plt.plot(time, integrated4, label='Dose 5 Srxn1 prediction', color=lighten_color('yellow', (1 / 2) * 1),
                #          linestyle='dashed')
                # plt.plot(time, integrated5, label='Dose 6 Srxn1 prediction', color=lighten_color('yellow', (1 / 2) * 1),
                #          linestyle='dashed')
                #
                # plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 0], label='Dose 1 Srxn1 data',
                #          color=lighten_color('green', (1 / 4) * 1))
                # plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 1], label='Dose 2 Srxn1 data',
                #          color=lighten_color('green', (1 / 4) * 2))
                # plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 2], label='Dose 3 Srxn1 data',
                #          color=lighten_color('green', (1 / 4) * 3))
                # plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 3], label='Dose 4 Srxn1 data',
                #          color=lighten_color('green', (1 / 4) * 4))
                # plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 4], label='Dose 5 Srxn1 data',
                #          color=lighten_color('yellow', (1 / 2) * 1))
                # plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 5], label='Dose 6 Srxn1 data',
                #          color=lighten_color('yellow', (1 / 2) * 1))
                #
                plt.legend()
                plt.xlabel('Time (h)')
                plt.ylabel('Log10 value (AU)')
                # plt.savefig('E:/BFW/Master BPS/RP1/Technical/ODE learning tests/run' + str(
                #     run) + ' seed=' + str(seed) + ' SUL dydt smoothing test.png')
                # plt.clf()

                if collectMSE == True:
                    MSElist = np.zeros((6, 2))
                    MSElist[0, 0] = '1'
                    MSElist[0, 1] = np.log10(mean_squared_error(integrated0, pDF_Srxn1_Rmean_SUL32[0:1999, 0]))
                    MSElist[1, 0] = '2'
                    MSElist[1, 1] = np.log10(mean_squared_error(integrated1, pDF_Srxn1_Rmean_SUL32[0:1999, 1]))
                    MSElist[2, 0] = '3'
                    MSElist[2, 1] = np.log10(mean_squared_error(integrated2, pDF_Srxn1_Rmean_SUL32[0:1999, 2]))
                    MSElist[3, 0] = '4'
                    MSElist[3, 1] = np.log10(mean_squared_error(integrated3, pDF_Srxn1_Rmean_SUL32[0:1999, 3]))
                    MSElist[4, 0] = '6'
                    MSElist[4, 1] = np.log10(mean_squared_error(integrated4, pDF_Srxn1_Rmean_SUL32[0:1999, 4]))
                    MSElist[5, 0] = '7'
                    MSElist[5, 1] = np.log10(mean_squared_error(integrated5, pDF_Srxn1_Rmean_SUL32[0:1999, 5]))

                    # MSE0 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 0]))
                    # MSE1 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 1]))
                    # MSE2 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 2]))
                    # MSE3 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 3]))
                    # MSE4 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 4]))
                    # MSE5 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 5]))
            return


        # figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
        figure(num=None, figsize=(7, 6), dpi=80, facecolor='w', edgecolor='k')
        ES1_integration_plotting3(False, [1,2,3,4])
        plt.tight_layout()
        plt.legend(loc=4)
        plt.ylim(-1.5,4)

        # figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
        figure(num=None, figsize=(7, 6), dpi=80, facecolor='w', edgecolor='k')
        ES1_integration_plotting3(False, [5, 6])
        plt.tight_layout()
        plt.legend(loc=4)
        plt.ylim(-1.5,4)

    # time point time tests (fig 13)
    for h in range(1):
        traindoseindexlist = trainindex([1,2,3,4], 'ES1')
        valdoseindexlist = valindex([5,6], 'ES1')
        datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
        trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1, traindoseindexlist, valdoseindexlist)
        trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)

        model = load_model(str(directory) + 'Versions/NN2/run207 seed=1.h5')

        scaler.fit(scalerlist)
        time = scaler.inverse_transform(listt[0:2000, 0].reshape(2000, 1))

        def initial_time_index_generator(starttime, datalen, datahours):
            init = int((datalen/datahours) * starttime)
            return init

        def ODE_integrator_wtime(Nrf2_data, y0Srxn1, starttime, endtime):
            timelist = [listt[int((2000/32) * starttime) , 0]]
            Nrf2list = [Nrf2_data[int((2000/32) * starttime)]]
            Srxn1list = [y0Srxn1]
            dSrxn1list = []
            for i in range(int((2000/32) * (endtime-starttime))-1):
                print('Integrating ' + str(i) + '/' + str((2000/32) * (endtime-starttime)))
                Nrf2 = Nrf2list[-1]
                Nrf2 = Nrf2.reshape(1, 1)
                scaler.fit(datascalerlistNrf2)
                Nrf2 = scaler.transform(Nrf2)
                Srxn1 = Srxn1list[-1]
                Srxn1 = Srxn1.reshape(1, 1)
                scaler.fit(datascalerlistSrxn1)
                Srxn1 = scaler.transform(Srxn1)
                tensor = np.zeros((1, 1, 3))
                tensor[0, 0, 0] = timelist[-1]
                tensor[0, 0, 1] = Nrf2[0, 0]
                tensor[0, 0, 2] = Srxn1[0, 0]
                print(tensor)
                predic = model.predict(tensor)
                dSrxn1list.append(predic[0, 0])
                scaler.fit(diffscalerlist)
                predic = scaler.inverse_transform(predic)
                # scaler.fit(datascalerlistSrxn1)
                # predic = scaler.inverse_transform(predic)
                # predic = 10**predic
                nextSrxn1 = Srxn1list[-1]
                scaler.fit(datascalerlistSrxn1)
                nextSrxn1 = nextSrxn1.reshape(1, 1)
                nextSrxn1 = scaler.transform(nextSrxn1)
                nextSrxn1 = nextSrxn1[0, 0] + predic[0]
                nextSrxn1 = nextSrxn1.reshape(1, 1)
                nextSrxn1 = scaler.inverse_transform(nextSrxn1)
                Srxn1list.append(nextSrxn1[0])
                print(predic)
                timelist.append(listt[(i + 1 + int((2000/32) * starttime)), 0])
                Nrf2list.append(Nrf2_data[(i + 1 + int((2000/32) * starttime))])
            # def integration_loss():
            #     integratedSrxn1_diff = np.diff()
            return Srxn1list, Nrf2list


        linetype = 3
        def tp_timetesting5(starttime, endtime, timelist, datalist,  prediccolor='purple', datacolvalue=0.5, linewidth=[3,3,3,3], linestyle=[(0,(linetype,1)),(0,(linetype,1.5)),(0,(linetype,2)),(0,(linetype,2.5))]):
            plt.plot(time[:, 0], pDF_Srxn1_Rmean_ES1[:,1],
                     label='Data ' + str(0) + '-' + str(32) + 'h',
                     color=lighten_color('blue', 0.4), linewidth=8)

            for i in range(len(starttime)):
                print(i)
                plt.plot(timelist[i], datalist[i],
                         label = 'Prediction ' + str(starttime[i]) + '-' + str(endtime[i]) + 'h',
                         color = lighten_color(prediccolor, (1 / len(starttime)) * (i + 1)),
                         linestyle = linestyle[i], linewidth =linewidth[0])
            return

        start1 = 0
        end1 = 32
        time1 = time[int(initial_time_index_generator(start1, 2000, 32)):, 0]
        Srxn1test1, Nrf2test1 = ODE_integrator_wtime(pDF_Nrf2_Rmean_ES1[:,1], pDF_Srxn1_Rmean_ES1[int(initial_time_index_generator(start1, 2000, 32)),1],start1, end1)
        start2 = 4
        end2 = 32
        time2 = time[int(initial_time_index_generator(start2, 2000, 32)):, 0]
        Srxn1test2, Nrf2test2 = ODE_integrator_wtime(pDF_Nrf2_Rmean_ES1[:,1], pDF_Srxn1_Rmean_ES1[initial_time_index_generator(start2, 2000, 32),1],start2, end2)
        start3 = 8
        end3 = 32
        time3 = time[int(initial_time_index_generator(start3, 2000, 32)):, 0]
        Srxn1test3, Nrf2test3 = ODE_integrator_wtime(pDF_Nrf2_Rmean_ES1[:,1], pDF_Srxn1_Rmean_ES1[initial_time_index_generator(start3, 2000, 32),1],start3, end3)
        start4 = 16
        end4 = 32
        time4 = time[int(initial_time_index_generator(start4, 2000, 32)):, 0]
        Srxn1test4, Nrf2test4 = ODE_integrator_wtime(pDF_Nrf2_Rmean_ES1[:,1], pDF_Srxn1_Rmean_ES1[initial_time_index_generator(start4, 2000, 32),1],start4, end4)

        figure(num=None, figsize=(7, 6), dpi=80, facecolor='w', edgecolor='k')
        tp_timetesting5([0,4,8,16],[32,32,32,32],[time1,time2,time3,time4], [Srxn1test1,Srxn1test2,Srxn1test3,Srxn1test4])
        plt.tight_layout()
        plt.legend(loc=4)
        plt.ylim(-1,3)
        plt.xlabel('Time (h)')
        plt.ylabel('Log10 value (AU)')
        plt.legend(loc=4)
        plt.tight_layout()

        start1 = 0
        end1 = 32
        time1 = time[:int(initial_time_index_generator(end1, 2000, 32)), 0]
        Srxn1test1, Nrf2test1 = ODE_integrator_wtime(pDF_Nrf2_Rmean_ES1[:,1], pDF_Srxn1_Rmean_ES1[int(initial_time_index_generator(start1, 2000, 32)),1],start1, end1)
        start2 = 0
        end2 = 16
        time2 = time[:int(initial_time_index_generator(end2, 2000, 32)), 0]
        Srxn1test2, Nrf2test2 = ODE_integrator_wtime(pDF_Nrf2_Rmean_ES1[:,1], pDF_Srxn1_Rmean_ES1[initial_time_index_generator(start2, 2000, 32),1],start2, end2)
        start3 = 0
        end3 = 8
        time3 = time[:int(initial_time_index_generator(end3, 2000, 32)), 0]
        Srxn1test3, Nrf2test3 = ODE_integrator_wtime(pDF_Nrf2_Rmean_ES1[:,1], pDF_Srxn1_Rmean_ES1[initial_time_index_generator(start3, 2000, 32),1],start3, end3)

        figure(num=None, figsize=(7, 6), dpi=80, facecolor='w', edgecolor='k')
        tp_timetesting5([0,0,0],[32,16,8],[time1,time2,time3], [Srxn1test1,Srxn1test2,Srxn1test3])
        plt.tight_layout()
        plt.legend(loc=4)
        plt.ylim(-1,3)
        plt.xlabel('Time (h)')
        plt.ylabel('Log10 value (AU)')
        plt.legend(loc=4)
        plt.tight_layout()


        def tp_timetesting4(starttime, endtime, dose, plottype,  prediccolor='purple', datacolvalue=0.5, linewidth=[0.5], linestyle=['dashed']):
            for i in range(len(starttime)):
                print(starttime)
                firstindex = np.where(time >= starttime[i])[0][0]
                lastindex = np.where(time <= endtime[i])[0][-1]
                ttvector = listt[firstindex:(lastindex+1), 0]
                ttdatalen = len(listt[firstindex:(lastindex+1), 0])
                datavector = trainX[(dose-1), firstindex:(lastindex+1),1]
                tensor = np.zeros((1, ttdatalen, 2))
                tensor[:,:,0] = ttvector
                tensor[:,:,1] = datavector
                # plt.plot(datavector)
                predic = model.predict(tensor)
                predic = predic.reshape(ttdatalen, 1)
                scaler.fit(datascalerlistSrxn1)
                predic = scaler.inverse_transform(predic)
                data = pDF_Srxn1_Rmean_ES1[firstindex:(lastindex+1), (dose-1)]
                if len(linestyle) >1:
                    if plottype == 'data':
                        plt.plot(time[firstindex:(lastindex + 1), 0], data,
                                 label='Data ' + str(starttime[i]) + '-' + str(endtime[i]) + 'h',
                                 color=lighten_color('blue', datacolvalue), linewidth=linewidth)
                    if plottype == 'predic':
                        plt.plot(time[firstindex:(lastindex + 1), 0], predic,
                                 label='Prediction ' + str(starttime[i]) + '-' + str(endtime[i]) + 'h',
                                 color=lighten_color(prediccolor, (1 / len(starttime)) * (i + 1)),linestyle=linestyle[i], linewidth=linewidth[i])
                else:
                    if plottype == 'data':
                        plt.plot(time[firstindex:(lastindex + 1), 0], data,
                                 label='Data ' + str(starttime[i]) + '-' + str(endtime[i]) + 'h',
                                 color=lighten_color('blue', datacolvalue), linewidth=linewidth)
                    if plottype == 'predic':
                        plt.plot(time[firstindex:(lastindex + 1), 0], predic,
                                 label='Prediction ' + str(starttime[i]) + '-' + str(endtime[i]) + 'h',
                                 color=lighten_color(prediccolor, (1 / len(starttime)) * (i + 1)),linestyle=linestyle[0], linewidth=linewidth[i])
            return


        def ES1_integration_plotting4(collectMSE, plotlist):
            for i in range(len(plotlist)):
                if plotlist[i] == 1:
                    MSE1 = np.around(np.log10(mean_squared_error(integrated0, pDF_Srxn1_Rmean_ES1[0:1999, 0])), 2)
                    plt.plot(time, integrated0, label='Dose 1 Srxn1 prediction, log MSE=' + str(MSE1), color=lighten_color('green', (1 / 4) * 1), linestyle='dashed')
                    plt.plot(time, pDF_Srxn1_Rmean_ES1[0:1999, 0], label='Dose 1 Srxn1 data', color=lighten_color('green', (1 / 4) * 1))
                if plotlist[i] == 2:
                    MSE2 = np.around(np.log10(mean_squared_error(integrated1, pDF_Srxn1_Rmean_ES1[0:1999:, 1])), 2)
                    plt.plot(time, integrated1, label='Dose 2 Srxn1 prediction, log MSE=' + str(MSE2), color=lighten_color('green', (1 / 4) * 2), linestyle='dashed')
                    plt.plot(time, pDF_Srxn1_Rmean_ES1[0:1999:, 1], label='Dose 2 Srxn1 data', color=lighten_color('green', (1 / 4) * 2))
                if plotlist[i] == 3:
                    MSE3 = np.around(np.log10(mean_squared_error(integrated2, pDF_Srxn1_Rmean_ES1[0:1999:, 2])), 2)
                    plt.plot(time, integrated2, label='Dose 3 Srxn1 prediction, log MSE=' + str(MSE3), color=lighten_color('green', (1 / 4) * 3), linestyle='dashed')
                    plt.plot(time, pDF_Srxn1_Rmean_ES1[0:1999:, 2], label='Dose 3 Srxn1 data', color=lighten_color('green', (1 / 4) * 3))
                if plotlist[i] == 4:
                    MSE4 = np.around(np.log10(mean_squared_error(integrated3, pDF_Srxn1_Rmean_ES1[0:1999:, 3])), 2)
                    plt.plot(time, integrated3, label='Dose 4 Srxn1 prediction, log MSE=' + str(MSE4), color=lighten_color('green', (1 / 4) * 4), linestyle='dashed')
                    plt.plot(time, pDF_Srxn1_Rmean_ES1[0:1999:, 3], label='Dose 4 Srxn1 data', color=lighten_color('green', (1 / 4) * 4))
                if plotlist[i] == 5:
                    MSE5 = np.around(np.log10(mean_squared_error(integrated4, pDF_Srxn1_Rmean_ES1[0:1999:, 4])), 2)
                    plt.plot(time, integrated4, label='Dose 5 Srxn1 prediction, log MSE=' + str(MSE5), color=lighten_color('orange', (1 / 2) * 1), linestyle='dashed')
                    plt.plot(time, pDF_Srxn1_Rmean_ES1[0:1999:, 4], label='Dose 5 Srxn1 data', color=lighten_color('orange', (1 / 2) * 1))
                if plotlist[i] == 6:
                    MSE6 = np.around(np.log10(mean_squared_error(integrated5, pDF_Srxn1_Rmean_ES1[0:1999:, 5])), 2)
                    plt.plot(time, integrated5, label='Dose 6 Srxn1 prediction, log MSE=' + str(MSE6), color=lighten_color('orange', (1 / 2) * 2), linestyle='dashed')
                    plt.plot(time, pDF_Srxn1_Rmean_ES1[0:1999:, 5], label='Dose 6 Srxn1 data', color=lighten_color('orange', (1 / 2) * 2))
                #
                #
                # MSE2 = np.around(np.log10(mean_squared_error(integrated1, pDF_Srxn1_Rmean_ES1[:, 1])), 2)
                # MSE3 = np.around(np.log10(mean_squared_error(integrated2, pDF_Srxn1_Rmean_ES1[:, 2])), 2)
                # MSE4 = np.around(np.log10(mean_squared_error(integrated3, pDF_Srxn1_Rmean_ES1[:, 3])), 2)
                # MSE5 = np.around(np.log10(mean_squared_error(integrated4, pDF_Srxn1_Rmean_ES1[:, 4])), 2)
                # MSE6 = np.around(np.log10(mean_squared_error(integrated5, pDF_Srxn1_Rmean_ES1[:, 5])), 2)
                #
                # plt.plot(time, integrated0, label='Dose 1 Srxn1 prediction', color=lighten_color('green', (1 / 4) * 1),
                #          linestyle='dashed')
                # plt.plot(time, integrated1, label='Dose 2 Srxn1 prediction', color=lighten_color('green', (1 / 4) * 2),
                #          linestyle='dashed')
                # plt.plot(time, integrated2, label='Dose 3 Srxn1 prediction', color=lighten_color('green', (1 / 4) * 3),
                #          linestyle='dashed')
                # plt.plot(time, integrated3, label='Dose 4 Srxn1 prediction', color=lighten_color('green', (1 / 4) * 4),
                #          linestyle='dashed')
                # plt.plot(time, integrated4, label='Dose 5 Srxn1 prediction', color=lighten_color('yellow', (1 / 2) * 1),
                #          linestyle='dashed')
                # plt.plot(time, integrated5, label='Dose 6 Srxn1 prediction', color=lighten_color('yellow', (1 / 2) * 1),
                #          linestyle='dashed')
                #
                # plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 0], label='Dose 1 Srxn1 data',
                #          color=lighten_color('green', (1 / 4) * 1))
                # plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 1], label='Dose 2 Srxn1 data',
                #          color=lighten_color('green', (1 / 4) * 2))
                # plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 2], label='Dose 3 Srxn1 data',
                #          color=lighten_color('green', (1 / 4) * 3))
                # plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 3], label='Dose 4 Srxn1 data',
                #          color=lighten_color('green', (1 / 4) * 4))
                # plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 4], label='Dose 5 Srxn1 data',
                #          color=lighten_color('yellow', (1 / 2) * 1))
                # plt.plot(time, pDF_Srxn1_Rmean_ES1[:, 5], label='Dose 6 Srxn1 data',
                #          color=lighten_color('yellow', (1 / 2) * 1))
                #
                plt.legend()
                plt.xlabel('Time (h)')
                plt.ylabel('Log10 value (AU)')
                # plt.savefig('E:/BFW/Master BPS/RP1/Technical/ODE learning tests/run' + str(
                #     run) + ' seed=' + str(seed) + ' SUL dydt smoothing test.png')
                # plt.clf()

                if collectMSE == True:
                    MSElist = np.zeros((6, 2))
                    MSElist[0, 0] = '1'
                    MSElist[0, 1] = np.log10(mean_squared_error(integrated0, pDF_Srxn1_Rmean_SUL32[0:1999, 0]))
                    MSElist[1, 0] = '2'
                    MSElist[1, 1] = np.log10(mean_squared_error(integrated1, pDF_Srxn1_Rmean_SUL32[0:1999, 1]))
                    MSElist[2, 0] = '3'
                    MSElist[2, 1] = np.log10(mean_squared_error(integrated2, pDF_Srxn1_Rmean_SUL32[0:1999, 2]))
                    MSElist[3, 0] = '4'
                    MSElist[3, 1] = np.log10(mean_squared_error(integrated3, pDF_Srxn1_Rmean_SUL32[0:1999, 3]))
                    MSElist[4, 0] = '6'
                    MSElist[4, 1] = np.log10(mean_squared_error(integrated4, pDF_Srxn1_Rmean_SUL32[0:1999, 4]))
                    MSElist[5, 0] = '7'
                    MSElist[5, 1] = np.log10(mean_squared_error(integrated5, pDF_Srxn1_Rmean_SUL32[0:1999, 5]))

                    # MSE0 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 0]))
                    # MSE1 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 1]))
                    # MSE2 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 2]))
                    # MSE3 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 3]))
                    # MSE4 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 4]))
                    # MSE5 = np.log10(mean_squared_error(SULintegrated0, pDF_Srxn1_Rmean_SUL32[:, 5]))
            return

    # Simple artificial data tests (figure 16)
    for h in range(1):
        model = load_model(str(directory) + 'Versions/NN2/run210 seed=1.h5')
        # model = load_model(str(directory) + 'Versions/NN2/run209 seed=6.h5')

        traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
        valdoseindexlist = valindex([3.5], 'SUL')
        datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
        trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
        trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32,1501, 5)

        scaler.fit(scalerlist)
        time = scaler.inverse_transform(listt[:,0].reshape(2000, 1))

        def ODE_integrator(Nrf2_data, y0Srxn1, timesteps):
            timelist = [listt[0, 0]]
            Nrf2list = [Nrf2_data[0]]
            Srxn1list = [np.float64(y0Srxn1)]
            dSrxn1list = []
            for i in range(timesteps):
                print('Integrating ' + str(i) + '/' + str(timesteps))
                Nrf2 = Nrf2list[-1]
                Nrf2 = Nrf2.reshape(1, 1)
                scaler.fit(datascalerlistNrf2)
                Nrf2 = scaler.transform(Nrf2)
                Srxn1 = Srxn1list[-1]
                Srxn1 = Srxn1.reshape(1, 1)
                scaler.fit(datascalerlistSrxn1)
                Srxn1 = scaler.transform(Srxn1)
                tensor = np.zeros((1, 1, 3))
                tensor[0, 0, 0] = timelist[-1]
                tensor[0, 0, 1] = Nrf2[0, 0]
                tensor[0, 0, 2] = Srxn1[0, 0]
                print(tensor)
                predic = model.predict(tensor)
                dSrxn1list.append(predic[0, 0])
                scaler.fit(diffscalerlist)
                predic = scaler.inverse_transform(predic)
                # scaler.fit(datascalerlistSrxn1)
                # predic = scaler.inverse_transform(predic)
                # predic = 10**predic
                nextSrxn1 = Srxn1list[-1]
                scaler.fit(datascalerlistSrxn1)
                nextSrxn1 = nextSrxn1.reshape(1, 1)
                nextSrxn1 = scaler.transform(nextSrxn1)
                nextSrxn1 = nextSrxn1[0, 0] + predic[0]
                nextSrxn1 = nextSrxn1.reshape(1, 1)
                nextSrxn1 = scaler.inverse_transform(nextSrxn1)
                Srxn1list.append(nextSrxn1[0])
                print(predic)
                timelist.append(listt[(i + 1), 0])
                Nrf2list.append(Nrf2_data[(i + 1)])
            # def integration_loss():
            #     integratedSrxn1_diff = np.diff()
            return Srxn1list, dSrxn1list, Nrf2list, timelist

        # figure 16a
        def emptyvectortests(valuelist):
            collected = np.zeros((2000,len(valuelist)))
            for valueindex in range(len(valuelist)):
                valuevector = np.zeros((2000))
                for i in range(len(valuevector)):
                    valuevector[i] = valuelist[valueindex]
                outputx, dSrxn1control, nrf2, time = ODE_integrator(valuevector, 0, 1999)
                # outputx, dSrxn1control, nrf2, time = ODE_integrator(valuevector, pDF_Srxn1_Rmean_SUL32[0, 1], 1999)

                collected[:,valueindex] = outputx[:]
            return collected

        # figure 16c
        def expvectortests(valuelist):
            output = np.zeros((2000,len(valuelist)))
            for valueindex in range(len(valuelist)):
                valuevector = np.zeros((2000))
                for i in range(len(valuevector)):
                    x = (1 / 2000000) * ((i + 1) ** 2)
                    valuevector[i] = x
                outputx, dSrxn1control, nrf2, time = ODE_integrator(valuevector, pDF_Srxn1_Rmean_SUL32[0, 1], 1999)
                output[:,valueindex] = outputx[:]
            return output, valuevector

        # Figure 16b
        def negexpvectortests(valuelist):
            output = np.zeros((2000,len(valuelist)))
            for valueindex in range(len(valuelist)):
                valuevector = np.zeros((2000))
                for i in range(len(valuevector)):
                    x = (-1 / 2000000) * ((i + 1) ** 2) + 2
                    valuevector[i] = x
                outputx, dSrxn1control, nrf2, time = ODE_integrator(valuevector, pDF_Srxn1_Rmean_SUL32[0, 1], 1999)
                output[:,valueindex] = outputx[:]
            return output, valuevector

        # Experimental, attempt at finding a steady state Nrf2 value that resulted in a steady state Srxn1 value
        def steadystate_algorithm(scalereps, initscale):
            output = np.zeros((2000,1))
            derivslist = np.zeros((6, 2))
            scalelist=[initscale]
            for k in range(scalereps):
                Nrf2ssconc = []
                scale = scalelist[k]
                print('scale=' +str(scale))
                for i in range(6):
                    Nrf2ssconc.append(scale * (i + 1) - 0.5*scale)
                for j in range(6):
                    valuevector = np.zeros((2000))
                    for i in range(len(valuevector)):
                        if i <= 999:
                            valuevector[i] = Nrf2ssconc[j]
                        if i > 999:
                            valuevector[i] = 1
                    outputx, dSrxn1control, nrf2, time = ODE_integrator(valuevector, pDF_Srxn1_Rmean_SUL32[0, 1], 1999)
                    output[:,0] = outputx[:]
                    deriv = (output[600] - output[0])**2
                    plt.plot(output, label='deriv=' + str(deriv) + 'scale=' + str(Nrf2ssconc[j]),
                             color=lighten_color('blue', (1/(8*scalereps)) *((j+1)*(k+1))))
                    derivslist[j,0] = deriv
                    derivslist[j,1] = Nrf2ssconc[j]
                min = np.where(derivslist == np.amin(derivslist))[0][0]
                minscale = derivslist[np.where(derivslist == np.amin(derivslist))[0][0],1]
                print('minscale='+str(minscale))
                scalelist.append(minscale*0.5)
            return output, valuevector

        # Figure 16d
        def multexposuretests(dose1, dose2):
            valuevector = np.zeros((2000))
            for i in range(len(valuevector)):
                if i <= 999:
                    valuevector[i] = dose1
                if i > 999:
                    valuevector[i] = dose2
            print(valuevector)
            valuevector = np.array(valuevector)
            outputx, dSrxn1control, nrf2, time = ODE_integrator(valuevector, 0.05, 1999)
            output = outputx[:]
            return output, valuevector

    # exp tests
    for h in range(1):
        posexpSrxn1, posexpNrf2 = expvectortests([1])
        negexpSrxn1, negexpNrf2 = negexpvectortests([1])
        figure(num=None, figsize=(7, 6), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(time[:-1], posexpSrxn1[:-1],
                 label='integrated Srxn1 prediction',
                 color='blue',
                 linestyle='dashed')
        plt.plot(time[:-1], posexpNrf2[:-1],
                 label='artificial Nrf2 data',
                 color='black',
                 linestyle='dotted')
        # plt.ylim(0, 4)
        plt.xlabel('Time (h)')
        plt.ylabel('Log10 value (AU)')
        plt.legend(loc=2)
        plt.tight_layout()


        figure(num=None, figsize=(7, 6), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(time, negexpSrxn1,
                 label='integrated Srxn1 prediction',
                 color='blue',
                 linestyle='dashed')
        plt.plot(time, negexpNrf2,
                 label='artificial Nrf2 data',
                 color='black',
                 linestyle='dotted')
        # plt.ylim(0, 4)
        plt.xlabel('Time (h)')
        plt.ylabel('Log10 value (AU)')
        plt.legend(loc=2)
        plt.tight_layout()

    # multiple exposure
    for h in range(1):
        # output, valuevector = multexposuretests(0.3, 0.8)
        output, valuevector = multexposuretests(0.025, 0.8)

        figure(num=None, figsize=(7, 6), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(time, output,
                 color='blue',
                 linestyle='dashed',
                 label='integrated Srxn1 prediction')
        plt.plot(time, valuevector,
                 color='black',
                 linestyle='dotted',
                 label='artificial Nrf2 data')
        # plt.ylim(0, 4)
        plt.xlabel('Time (h)')
        plt.ylabel('Log10 value (AU)')
        plt.legend(loc=2)
        plt.tight_layout()







        valuelist = [0,0.025,0.05,0.1,0.2,0.4,0.8]
        collected = emptyvectortests(valuelist)

        plt.plot(collected)
        valuelistplot = [0.025,0.1,0.2,0.4,0.8]


        def emptyvectorplotting(valuelist, plottype, prediccolor='purple', datacolvalue=0.5, linewidth=[1],
                                  linestyle=['solid']):
            for i in range(len(valuelist)):
                    for j in range(len(linewidth)):
                        if plottype == 'data':
                            plt.plot(time, data,
                                     # label='Data ' + str(starttime[i]) + '-' + str(endtime[i]) + 'h',
                                     color=lighten_color('blue', datacolvalue), linewidth=linewidth)
                        if plottype == 'predic':
                            if i == 0:
                                plt.plot(time[:,0], collected[:,i],
                                         label='Nrf2 steady state value ' + str(valuelist[i]),
                                         color=lighten_color(prediccolor, (1 / len(valuelist)) * (i + 1)), linestyle=linestyle[0],
                                         linewidth=linewidth[j])
                                print('check1')

                            if i >= 1:
                                plt.plot(time[:,0], collected[:,i+1],
                                         label='Nrf2 steady state value ' + str(valuelist[i]),
                                         color=lighten_color(prediccolor, (1 / len(valuelist)) * (i + 1)), linestyle=linestyle[0],
                                         linewidth=linewidth[j])
            return
        np.shape(collected)
        # figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
        figure(num=None, figsize=(7, 6), dpi=80, facecolor='w', edgecolor='k')
        # convlstm_timetesting4([0,0,0],[32,16,8],3, 'predic', linewidth=[3,3,3], linestyle=['dashdot','dashed','dotted'])
        linetype = 3
        emptyvectorplotting(valuelistplot, 'predic', linestyle=['dashed'], prediccolor='blue', linewidth=[2])
        plt.ylim(0, 2)
        plt.xlabel('Time (h)')
        plt.ylabel('Log10 value (AU)')
        plt.legend(loc=2)
        plt.tight_layout()

    # SUL data tests (fig 14, 15, 17)
    for h in range(1):
        # LSTMdtp SUL test figure 14
        for h in range(1):
            traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
            valdoseindexlist = valindex([3.5], 'SUL')
            datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
            trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
            trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32,1501, 5)

            model = load_model(str(directory) + 'Versions/NN2/run209 seed=6.h5')
            scaler.fit(scalerlist)
            time = scaler.inverse_transform(listt[:,0].reshape(2000, 1))

            def ODE_integrator(Nrf2_data, y0Srxn1, timesteps):
                timelist = [listt[0, 0]]
                Nrf2list = [Nrf2_data[0]]
                Srxn1list = [y0Srxn1]
                dSrxn1list = []
                for i in range(timesteps):
                    print('Integrating ' + str(i) + '/' + str(timesteps))
                    Nrf2 = Nrf2list[-1]
                    Nrf2 = Nrf2.reshape(1, 1)
                    scaler.fit(datascalerlistNrf2)
                    Nrf2 = scaler.transform(Nrf2)
                    Srxn1 = Srxn1list[-1]
                    Srxn1 = Srxn1.reshape(1, 1)
                    scaler.fit(datascalerlistSrxn1)
                    Srxn1 = scaler.transform(Srxn1)
                    tensor = np.zeros((1, 1, 3))
                    tensor[0, 0, 0] = timelist[-1]
                    tensor[0, 0, 1] = Nrf2[0, 0]
                    tensor[0, 0, 2] = Srxn1[0, 0]
                    print(tensor)
                    predic = model.predict(tensor)
                    dSrxn1list.append(predic[0, 0])
                    scaler.fit(diffscalerlist)
                    predic = scaler.inverse_transform(predic)
                    # scaler.fit(datascalerlistSrxn1)
                    # predic = scaler.inverse_transform(predic)
                    # predic = 10**predic
                    nextSrxn1 = Srxn1list[-1]
                    scaler.fit(datascalerlistSrxn1)
                    nextSrxn1 = nextSrxn1.reshape(1, 1)
                    nextSrxn1 = scaler.transform(nextSrxn1)
                    nextSrxn1 = nextSrxn1[0, 0] + predic[0]
                    nextSrxn1 = nextSrxn1.reshape(1, 1)
                    nextSrxn1 = scaler.inverse_transform(nextSrxn1)
                    Srxn1list.append(nextSrxn1[0])
                    print(predic)
                    timelist.append(listt[(i + 1), 0])
                    Nrf2list.append(Nrf2_data[(i + 1)])
                # def integration_loss():
                #     integratedSrxn1_diff = np.diff()
                return Srxn1list
            collected = np.zeros((2000,6))
            for i in range(6):
                Srxn1output = ODE_integrator(pDF_Nrf2_Rmean_SUL32[:,i], pDF_Srxn1_Rmean_SUL32[0,i], 1999)
                collected[:,i] = Srxn1output
                np.shape(Srxn1output)
            # dose1625_alt, control2, Nrf2_2, timelist2 = ODE_integrator(pDF_Nrf2_Rmean_SUL32[:,5], pDF_Srxn1_Rmean_SUL32[0,4], 1999)


            def tp_SULplotting(doseindexlist, doselist, prediccolor='purple', datacolvalue=0.5, linewidth=[1],
                                      linestyle=['solid', 'dashed']):
                for i in range(len(doselist)):
                    MSE = np.around(np.log10(mean_squared_error(collected[:,i], pDF_Srxn1_Rmean_SUL32[:,i])),2)
                    if doselist[i] != 3.5:
                        # if doselist[i] == 16.25:
                        #     plt.plot(time[:,0], pDF_Srxn1_Rmean_SUL32[:,doseindexlist[i]],
                        #              label='Srxn1 data ' + str(doselist[i]) + 'uM SUL, MSE = ' + str(MSE),
                        #              linestyle=linestyle[0],
                        #              linewidth=linewidth[0],
                        #              color=lighten_color('green', (1 / len(doselist)) * (i + 1)))
                        #     plt.plot(time[:,0], dose1625_alt,
                        #              label='Srxn1 prediction ' + str(doselist[i]) + 'uM SUL, MSE = ' + str(MSE),
                        #              color=lighten_color('green', (1 / len(doselist)) * (i + 1)),
                        #              linestyle=linestyle[1],
                        #              linewidth=linewidth[0])
                        # else:
                        plt.plot(time[:, 0], pDF_Srxn1_Rmean_SUL32[:, doseindexlist[i]],
                                 label='Srxn1 data ' + str(doselist[i]) + 'uM SUL, MSE = ' + str(MSE),
                                 linestyle=linestyle[0],
                                 linewidth=linewidth[0],
                                 color=lighten_color('green', (1 / 5) * (i + 1)))
                        plt.plot(time[:, 0], collected[:,doseindexlist[i]],
                                 label='Srxn1 prediction ' + str(doselist[i]) + 'uM SUL, MSE = ' + str(MSE),
                                 color=lighten_color('green', (1 / 5) * (i +1 )),
                                 linestyle=linestyle[1],
                                 linewidth=linewidth[0])
                    else:
                        plt.plot(time[:,0], pDF_Srxn1_Rmean_SUL32[:,doseindexlist[i]],
                                 label='Srxn1 data ' + str(doselist[i]) + 'uM SUL, MSE = ' + str(MSE),
                                 linestyle=linestyle[0],
                                 linewidth=linewidth[0],
                                 color=lighten_color('orange', (1 / len(doselist)) * (i + 1)))
                        plt.plot(time[:,0], collected[:,doseindexlist[i]],
                                 label='Srxn1 prediction ' + str(doselist[i]) + 'uM SUL, MSE = ' + str(MSE),
                                 color=lighten_color('orange', (1 / len(doselist)) * (i + 1)),
                                 linestyle=linestyle[1],
                                 linewidth=linewidth[0])
                return

            doseindex = [0,1,2,3,4,5]
            doselist = [0.35, 0.75, 1.62, 3.5, 7.54]

            figure(num=None, figsize=(10, 9), dpi=80, facecolor='w', edgecolor='k')
            # convlstm_timetesting4([0,0,0],[32,16,8],3, 'predic', linewidth=[3,3,3], linestyle=['dashdot','dashed','dotted'])
            linetype = 3
            tp_SULplotting(doseindex, doselist, prediccolor='blue', linewidth=[2])
            plt.ylim(-0.1, 2.5)
            plt.xlabel('Time (h)')
            plt.ylabel('Log10 value (AU)')
            plt.legend(loc=2)
            plt.tight_layout()

        # ANDR CDDO comparable data tests (fig 15)
        for h in range(1):
            CDDOSrxn1, adfsljk, alsjdfk, aslfj = ODE_integrator(pDF_Nrf2_Rmean_CDDO32[:,4], pDF_Srxn1_Rmean_CDDO32[0,4], 1999)
            AndrSrxn1, askldfj, asldfj, asjdklf = ODE_integrator(pDF_Nrf2_Rmean_Andr32[:,4], pDF_Srxn1_Rmean_CDDO32[0,4], 1999)

            figure(num=None, figsize=(10, 9), dpi=80, facecolor='w', edgecolor='k')
            plt.plot(time, CDDOSrxn1, color='blue', label='0.22uM CDDO Srxn1 prediction', linestyle='dashed')
            plt.plot(time, pDF_Nrf2_Rmean_CDDO32[:,4], color='blue', label='0.22uM CDDO Nrf2 data', linestyle='dotted')
            plt.plot(time, AndrSrxn1, color='purple', label='10.0uM ANDR Srxn1 prediction', linestyle='dashed')
            plt.plot(time, pDF_Nrf2_Rmean_Andr32[:,4], color='purple', label='10.0uM ANDR Nrf2 data', linestyle='dotted')
            plt.ylim(-0.1, 1.2)
            plt.xlabel('Time (h)')
            plt.ylabel('Log10 value (AU)')
            plt.legend(loc=2)
            plt.tight_layout()

        # Compound comparison (fig 17)
        for h in range(1):
            collectedSUL = np.zeros((2000,6))
            collectedAndr = np.zeros((2000,6))
            collectedCDDO = np.zeros((2000,6))
            for i in range(6):
                collectedSUL[:,i] = ODE_integrator(pDF_Nrf2_Rmean_SUL32[:,i], pDF_Srxn1_Rmean_SUL32[0,i], 1999)
                # collectedAndr[:,i] = ODE_integrator(pDF_Nrf2_Rmean_Andr32[:,i], pDF_Srxn1_Rmean_Andr32[0,i], 1999)
                collectedCDDO[:,i] = ODE_integrator(pDF_Nrf2_Rmean_CDDO32[:,i], pDF_Srxn1_Rmean_CDDO32[0,i], 1999)

            def qualcomp_plotting(indexlist, proteinlist):
                SULdoses = [0.35, 0.75, 1.62, 3.5, 7.54, 16.25]
                Andrdoses = [0.1, 0.32, 1.0, 3.16, 10, 31.62, 100]
                CDDOdoses = [0.01, 0.02, 0.05, 0.1, 0.22, 0.46, 1.00]
                for i in range(len(indexlist)):
                    for j in range(len(proteinlist)):
                        if proteinlist[j] == 'SUL':
                            if indexlist[i] == 3:
                                # plt.plot(time,pDF_Nrf2_Rmean_SUL32[:,indexlist[i]], color=lighten_color('black', (1 / len(indexlist)) * (i + 1)), label=str(SULdoses[indexlist[i]]) + 'uM SUL Nrf2 data', linestyle='dotted')
                                plt.plot(time,pDF_Srxn1_Rmean_SUL32[:,indexlist[i]], color=lighten_color('orange', (1 / len(indexlist)) * (i + 1)), label=str(SULdoses[indexlist[i]]) + 'uM SUL Srxn1 data')
                                plt.plot(time,collectedSUL[:,indexlist[i]], color=lighten_color('orange', (1 / len(indexlist)) * (i + 1)), linestyle='dashed', label=str(SULdoses[indexlist[i]]) + 'uM SUL Srxn1 prediction')
                            else:
                                # plt.plot(time,pDF_Nrf2_Rmean_SUL32[:,indexlist[i]], color=lighten_color('black', (1 / len(indexlist)) * (i + 1)), label=str(SULdoses[indexlist[i]]) + 'uM SUL Nrf2 data', linestyle='dotted')
                                plt.plot(time,pDF_Srxn1_Rmean_SUL32[:,indexlist[i]], color=lighten_color('green', (1 / len(indexlist)) * (i + 1)), label=str(SULdoses[indexlist[i]]) + 'uM SUL Srxn1 data')
                                plt.plot(time,collectedSUL[:,indexlist[i]], color=lighten_color('green', (1 / len(indexlist)) * (i + 1)), linestyle='dashed', label=str(SULdoses[indexlist[i]]) + 'uM SUL Srxn1 prediction')
                        if proteinlist[j] == 'Andr':
                            plt.plot(time, pDF_Nrf2_Rmean_Andr32[:, indexlist[i]],
                                     color=lighten_color('black', (1 / len(indexlist)) * (i + 1)),
                                     label=str(Andrdoses[indexlist[i]]) + 'uM Andr Nrf2 data', linestyle='dotted')
                            plt.plot(time,collectedAndr[:,indexlist[i]],
                                     color=lighten_color('purple', (1 / len(indexlist)) * (i + 1)),
                                     linestyle='dashed',
                                     label=str(Andrdoses[indexlist[i]]) + 'uM Andr prediction')
                            plt.plot(time,pDF_Srxn1_Rmean_Andr32[:,indexlist[i]],
                                     color=lighten_color('purple', (1 / len(indexlist)) * (i + 1)),
                                     label=str(Andrdoses[indexlist[i]]) + 'uM Andr data')

                        if proteinlist[j] == 'CDDO':
                            plt.plot(time, pDF_Nrf2_Rmean_CDDO32[:, indexlist[i]],
                                     color=lighten_color('black', (1 / len(indexlist)) * (i + 1)),
                                     label=str(CDDOdoses[indexlist[i]]) + 'uM CDDO Nrf2 data', linestyle='dotted')
                            plt.plot(time,collectedCDDO[:,indexlist[i]],
                                     color=lighten_color('blue', (1 / len(indexlist)) * (i + 1)),
                                     linestyle='dashed',
                                     label=str(CDDOdoses[indexlist[i]]) + 'uM CDDO prediction')
                            plt.plot(time,pDF_Srxn1_Rmean_CDDO32[:,indexlist[i]],
                                     color=lighten_color('blue', (1 / len(indexlist)) * (i + 1)),
                                     label=str(CDDOdoses[indexlist[i]]) + 'uM CDDO data')

            def qualcomp_plotting_Nrf2(indexlist, proteinlist):
                SULdoses = [0.35, 0.75, 1.62, 3.5, 7.54, 16.25]
                Andrdoses = [0.1, 0.32, 1.0, 3.16, 10, 31.62, 100]
                CDDOdoses = [0.01, 0.02, 0.05, 0.1, 0.22, 0.46, 1.00]
                for i in range(len(indexlist)):
                    for j in range(len(proteinlist)):
                        if proteinlist[j] == 'SUL':
                            if indexlist[i] == 3:
                                plt.plot(time,pDF_Nrf2_Rmean_SUL32[:,indexlist[i]], color=lighten_color('orange', (1 / len(indexlist)) * (i + 1)), label=str(SULdoses[indexlist[i]]) + 'uM SUL Nrf2 data')
                                # plt.plot(time,pDF_Srxn1_Rmean_SUL32[:,indexlist[i]], color=lighten_color('orange', (1 / len(indexlist)) * (i + 1)), label=str(SULdoses[indexlist[i]]) + 'uM SUL Srxn1 data')
                                # plt.plot(time,collectedSUL[:,indexlist[i]], color=lighten_color('orange', (1 / len(indexlist)) * (i + 1)), linestyle='dashed', label=str(SULdoses[indexlist[i]]) + 'uM SUL Srxn1 prediction')
                            else:
                                plt.plot(time,pDF_Nrf2_Rmean_SUL32[:,indexlist[i]], color=lighten_color('green', (1 / len(indexlist)) * (i + 1)), label=str(SULdoses[indexlist[i]]) + 'uM SUL Nrf2 data')
                                # plt.plot(time,pDF_Srxn1_Rmean_SUL32[:,indexlist[i]], color=lighten_color('green', (1 / len(indexlist)) * (i + 1)), label=str(SULdoses[indexlist[i]]) + 'uM SUL Srxn1 data')
                                # plt.plot(time,collectedSUL[:,indexlist[i]], color=lighten_color('green', (1 / len(indexlist)) * (i + 1)), linestyle='dashed', label=str(SULdoses[indexlist[i]]) + 'uM SUL Srxn1 prediction')
                        if proteinlist[j] == 'Andr':
                            plt.plot(time, pDF_Nrf2_Rmean_Andr32[:, indexlist[i]],
                                     color=lighten_color('black', (1 / len(indexlist)) * (i + 1)),
                                     label=str(Andrdoses[indexlist[i]]) + 'uM Andr Nrf2 data', linestyle='dotted')
                            plt.plot(time,collectedAndr[:,indexlist[i]],
                                     color=lighten_color('purple', (1 / len(indexlist)) * (i + 1)),
                                     linestyle='dashed',
                                     label=str(Andrdoses[indexlist[i]]) + 'uM Andr prediction')
                            plt.plot(time,pDF_Srxn1_Rmean_Andr32[:,indexlist[i]],
                                     color=lighten_color('purple', (1 / len(indexlist)) * (i + 1)),
                                     label=str(Andrdoses[indexlist[i]]) + 'uM Andr data')

                        if proteinlist[j] == 'CDDO':
                            plt.plot(time, pDF_Nrf2_Rmean_CDDO32[:, indexlist[i]],
                                     color=lighten_color('black', (1 / len(indexlist)) * (i + 1)),
                                     label=str(CDDOdoses[indexlist[i]]) + 'uM CDDO Nrf2 data', linestyle='dotted')
                            plt.plot(time,collectedCDDO[:,indexlist[i]],
                                     color=lighten_color('blue', (1 / len(indexlist)) * (i + 1)),
                                     linestyle='dashed',
                                     label=str(CDDOdoses[indexlist[i]]) + 'uM CDDO prediction')
                            plt.plot(time,pDF_Srxn1_Rmean_CDDO32[:,indexlist[i]],
                                     color=lighten_color('blue', (1 / len(indexlist)) * (i + 1)),
                                     label=str(CDDOdoses[indexlist[i]]) + 'uM CDDO data')


            figure(num=None, figsize=(7, 6), dpi=80, facecolor='w', edgecolor='k')
            qualcomp_plotting([0, 1, 2, 3, 4, 5], ['SUL'])
            # plt.ylim(-0.1, 1.2)
            plt.xlabel('Time (h)')
            plt.ylabel('Log10 value (AU)')
            plt.legend(loc=2)
            plt.tight_layout()

            figure(num=None, figsize=(7, 6), dpi=80, facecolor='w', edgecolor='k')
            qualcomp_plotting_Nrf2([0, 1, 2, 3, 4, 5], ['SUL'])
            # plt.ylim(-0.1, 1.2)
            plt.xlabel('Time (h)')
            plt.ylabel('Log10 value (AU)')
            plt.legend(loc=1)
            plt.tight_layout()

# 15. Run history (128-200)
for i in range(1):
    def plotting(dose, DF, protein, predictime):
        # Data collection
        for i in range(1):
            fulldoselist = np.array([0.35, 0.75, 1.62, 3.5, 7.54, 16.25, 35])
            fullprediclistNrf2 = [predic1, predic2, predic3, predic4, predic5, predic6, predic7]
            # fullprediclistSrxn1 = [Srxn1_predic1, Srxn1_predic2, Srxn1_predic3, Srxn1_predic4,
            #                        Srxn1_predic5, Srxn1_predic6, Srxn1_predic7]
            dosecodelist = [str('035'), str('075'), str('162'), str('350'), str('754'), str('1625'),
                            str('3500')]
            plottingdoseindex = np.where(fulldoselist[:] == dose)
            if protein == 'Nrf2':
                predic = fullprediclistNrf2[int(plottingdoseindex[0])]
            # if protein == 'Srxn1':
            #     predic = fullprediclistSrxn1[int(plottingdoseindex[0])]
            dosecode = dosecodelist[int(plottingdoseindex[0])]

            if protein == str('Nrf2'):
                # if predictime == 36:
                #     Urangelist = [Urange_Nrf2_36_035, Urange_Nrf2_36_075, Urange_Nrf2_36_162,
                #                   Urange_Nrf2_36_350, Urange_Nrf2_36_754, Urange_Nrf2_36_1625,
                #                   Urange_Nrf2_36_3500]
                #     Lrangelist = [Lrange_Nrf2_36_035, Lrange_Nrf2_36_075, Lrange_Nrf2_36_162,
                #                   Lrange_Nrf2_36_350, Lrange_Nrf2_36_754, Lrange_Nrf2_36_1625,
                #                   Lrange_Nrf2_36_3500]

                if predictime == 36:
                    Urangelist = [Urange_Srxn1_36_035, Urange_Srxn1_36_075, Urange_Srxn1_36_162,
                                  Urange_Srxn1_36_350,
                                  Urange_Srxn1_36_754, Urange_Srxn1_36_1625, Urange_Srxn1_36_3500]
                    Lrangelist = [Lrange_Srxn1_36_035, Lrange_Srxn1_36_075, Lrange_Srxn1_36_162,
                                  Lrange_Srxn1_36_350,
                                  Lrange_Srxn1_36_754, Lrange_Srxn1_36_1625, Lrange_Srxn1_36_3500]

                # if predictime == 24:
                #     Urangelist = [Urange_Nrf2_24_035, Urange_Nrf2_24_075, Urange_Nrf2_24_162, Urange_Nrf2_24_350, Urange_Nrf2_24_754, Urange_Nrf2_24_1625, Urange_Nrf2_24_3500]
                #     Lrangelist = [Urange_Nrf2_24_035, Urange_Nrf2_24_075, Urange_Nrf2_24_162, Urange_Nrf2_24_350, Urange_Nrf2_24_754, Urange_Nrf2_24_1625, Urange_Nrf2_24_3500]
                #

            if protein == str('Srxn1'):
                if predictime == 36:
                    Urangelist = [Urange_Srxn1_36_035, Urange_Srxn1_36_075, Urange_Srxn1_36_162,
                                  Urange_Srxn1_36_350,
                                  Urange_Srxn1_36_754, Urange_Srxn1_36_1625, Urange_Srxn1_36_3500]
                    Lrangelist = [Lrange_Srxn1_36_035, Lrange_Srxn1_36_075, Lrange_Srxn1_36_162,
                                  Lrange_Srxn1_36_350,
                                  Lrange_Srxn1_36_754, Lrange_Srxn1_36_1625, Lrange_Srxn1_36_3500]

                # if predictime == 24:
                #     Urangelist = [Urange_Srxn1_24_035, Urange_Srxn1_24_075, Urange_Srxn1_24_162, Urange_Srxn1_24_350, Urange_Srxn1_24_754, Urange_Srxn1_24_1625, Urange_Srxn1_24_3500]
                #     Lrangelist = [Urange_Srxn1_24_035, Urange_Srxn1_24_075, Urange_Srxn1_24_162, Urange_Srxn1_24_350, Urange_Srxn1_24_754, Urange_Srxn1_24_1625, Urange_Srxn1_24_3500]
                #

            Urange = Urangelist[int(plottingdoseindex[0])]
            Lrange = Lrangelist[int(plottingdoseindex[0])]

            if derivative_training == True:
                Urange = np.diff(Urange)
                Lrange = np.diff(Lrange)

            # predics have two columns: 0 = Nrf2, 1 = Srxn1, difference between predic (Nrf2) and Srxn1_predic is in scaling
            if protein == str('Nrf2'):
                prediccolumn = 0
            if protein == str('Srxn1'):
                prediccolumn = 1
                prediccolumn

        if protein == 'Nrf2':
            MSElist = [p1nrf2mse, p2nrf2mse, p3nrf2mse, p4nrf2mse, p5nrf2mse, p6nrf2mse, p7nrf2mse]
        MSE = MSElist[int(plottingdoseindex[0])]

        # Calculating Tmax accuracy
        tmaxlist = []
        for i in range(np.shape(DF)[1]):
            tmax = np.where(DF[:, i] == np.amax(DF[:, i]))[0]
            tmaxlist.append(tmax)
        tmaxlist = np.array(tmaxlist)
        utmax = np.amax(tmaxlist)
        ltmax = np.amin(tmaxlist)

        predictmax = np.where(predic[:, 0] == np.amax(predic[:, 0]))[0]
        if predictmax[0] >= ltmax and predictmax[0] <= utmax:
            tmaxpredic = 'True'
        else:
            tmaxpredic = 'False'

        CIfitlist = []
        for i in range(2000):
            if predic1[i, 0] >= Lrange[i, 0] and predic1[i, 0] <= Urange[i, 0]:
                fit = 1
            else:
                fit = 0
            CIfitlist.append(fit)
        Accuracy = np.count_nonzero(CIfitlist) / len(CIfitlist)
        plt.plot(time, predic[:, int(prediccolumn)],
                 label=str(dose) + 'uM prediction, mse=' + str(np.around(MSE, 4)) + ' accuracy=' + str(
                     Accuracy) + ' Tmax accuracy=' + str(tmaxpredic), color='green', linestyle='dashed')
        plt.plot(time, Lrange[:, 0], label=str(dose) + 'uM lower replicate range ' + str(protein),
                 color='red',
                 linestyle='dashed')
        plt.plot(time, Urange[:, 0], label=str(dose) + 'uM upper replicate range ' + str(protein),
                 color='red',
                 linestyle='dashed')
        # plt.plot(Nrf2_035_mean, label='0.35uM replicate mean', color='blue')
        plt.legend()
        plt.xlabel('time (hours)')
        plt.ylabel('Nuclei Integrated Fluorescence Intensity')
        plt.title(
            str(dose) + 'uM replicate accuracy run' + str(run) + ' seed=' + str(seed) + ' loss=10^' + str(
                loss))
        plt.savefig('E:/BFW/Master BPS/RP1/Technical/Causal model tests/Training predictions/NN2/run' + str(
            run) + ' s' + str(seed) + ' loss=10^' + str(loss) + ' ' + str(protein) + ' ' + str(
            dosecode) + '.png')
        plt.clf()
        # Predictions to txt
        predic = pd.DataFrame(predic)
        MSE = np.array(MSE).reshape(1, 1)
        predic.to_csv(
            str(directory) + 'Predic DataFrames/NN2/run' + str(
                run) + ' s' + str(
                seed) + ' loss=10^' + str(loss) + ' MSE=' + str(MSE) + ' ' + str(protein) + ' ' + str(
                dosecode) + '.txt')
        return

    run = 128.1
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(9):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 649

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2)

    run = 129
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(9):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 800

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(100, return_sequences=True, activation='tanh'))
        model.add(LSTM(100, return_sequences=True, activation='tanh'))
        model.add(LSTM(100, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2)

    run = 130.1
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(9):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 372

        model = Sequential()
        model.add(
            LSTM(150, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(150, return_sequences=True, activation='tanh'))
        model.add(LSTM(150, return_sequences=True, activation='tanh'))
        model.add(LSTM(150, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2)

    # Size test 1/4
    run = 131.2
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(9):
        seed = i
        tf.random.set_seed(seed=i)

        # 1000, 950, 1200
        epch = 1200

        model = Sequential()
        model.add(
            LSTM(30, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(30, return_sequences=True, activation='tanh'))
        model.add(LSTM(30, return_sequences=True, activation='tanh'))
        model.add(LSTM(30, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2)

    # Size test 2/4
    run = 132
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(9):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2)

    # Size test 3/4
    run = 133
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(9):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(70, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(70, return_sequences=True, activation='tanh'))
        model.add(LSTM(70, return_sequences=True, activation='tanh'))
        model.add(LSTM(70, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2)

    # Size test 4/4, shape test 1/3
    run = 134.1
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(9):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 775

        model = Sequential()
        model.add(
            LSTM(40, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(40, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2)

    # shape test 2/3
    run = 135
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(9):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(60, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(20, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2)

    # shape test 3/3
    run = 136
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(9):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(20, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(60, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2)

    # shape test initial change 1/3
    run = 137
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(9):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(10, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(60, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2)

    # shape test initial change 2/3
    run = 138
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(9):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(20, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(60, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2)

    # shape test initial change 3/3
    run = 139
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(9):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(30, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(60, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2)

    # shape test last layer change 1/3
    run = 140
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(9):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(20, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2)

    # shape test last layer change 2/3
    run = 141
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(9):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(20, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(60, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2)

    # shape test last layer change 3/3
    run = 142
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(9):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(20, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(70, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2)

    # ES3 test
    run = 143
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(3)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES3, pDF_Srxn1_Rmean_ES3)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES3)
    NNconfig = 2
    for i in range(9):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(20, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(70, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2)

    run = 144
    traindoseindexlist = trainindex([1, 2, 3, 3])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(3)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES3, pDF_Srxn1_Rmean_ES3)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES3)
    NNconfig = 2
    for i in range(3):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(20, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(70, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2)

    run = 145
    traindoseindexlist = trainindex([1, 2, 3, 3])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(3)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES3, pDF_Srxn1_Rmean_ES3)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES3)
    NNconfig = 2
    for i in range(3):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(20, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(70, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2)

    run = 146.32
    traindoseindexlist = trainindex([2, 2, 3, 3])
    valdoseindexlist = valindex([5, 6, 7, 8])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(3)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES3, pDF_Srxn1_Rmean_ES3)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES3)
    plt.plot(trainX[:, 0, :])
    plt.plot(pDF_Srxn1_Rmean_ES3[:, 1])
    plt.clf()
    NNconfig = 2
    for i in range(3):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(40, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(40, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 4)

    run = 147
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc9_Nrf2_to_Srxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc9_Nrf2_to_Srxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(3):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(20, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(70, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2)

    run = 148
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc9_Nrf2_to_Srxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc9_Nrf2_to_Srxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(3):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 5000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(100, return_sequences=True, activation='tanh'))
        model.add(LSTM(100, return_sequences=True, activation='tanh'))
        model.add(LSTM(100, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2, 2)

    run = 149
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc9_Nrf2_to_Srxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc9_Nrf2_to_Srxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(3):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 5000

        model = Sequential()
        model.add(
            LSTM(10, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(10, return_sequences=True, activation='tanh'))
        model.add(LSTM(10, return_sequences=True, activation='tanh'))
        model.add(LSTM(10, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2, 2)

    run = 150
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc9_Nrf2_to_Srxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc9_Nrf2_to_Srxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(3):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 5000

        model = Sequential()
        model.add(
            LSTM(30, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(30, return_sequences=True, activation='tanh'))
        model.add(LSTM(30, return_sequences=True, activation='tanh'))
        model.add(LSTM(30, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2, 2)

    run = 151
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc9_Nrf2_to_Srxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc9_Nrf2_to_Srxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(3):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 5000

        model = Sequential()
        model.add(
            LSTM(30, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='relu'))
        model.add(LSTM(30, return_sequences=True, activation='relu'))
        model.add(LSTM(30, return_sequences=True, activation='relu'))
        model.add(LSTM(30, return_sequences=False, activation='relu'))
        model.add(Dense(1, activation='relu'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2, 2)

    run = 152
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc9_Nrf2_to_Srxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc9_Nrf2_to_Srxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(3):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 5000

        model = Sequential()
        model.add(
            LSTM(5, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(5, return_sequences=True, activation='tanh'))
        model.add(LSTM(5, return_sequences=True, activation='tanh'))
        model.add(LSTM(5, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2, 2)

    # Tests ES5 Nrf2 - Srxn1
    run = 153
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(5)
    trainX, testX = trainXfunc9_Nrf2_to_Srxn1(pDF_Nrf2_Rmean_ES5, pDF_Srxn1_Rmean_ES5)
    trainY, testY = trainYfunc9_Nrf2_to_Srxn1(pDF_Srxn1_Rmean_ES5)
    NNconfig = 2
    for i in range(3):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 5000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES5, pDF_Srxn1_Rmean_ES5)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2, 2)

    # Tests ES1 Nrf2 - Srxn1
    run = 154
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc9_Nrf2_to_Srxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc9_Nrf2_to_Srxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(3):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1500

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES5, pDF_Srxn1_Rmean_ES5)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2, 2)

    # Tests ES1 Nrf2 - Srxn1 (2)
    run = 155
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(5)
    trainX, testX = trainXfunc9_Nrf2_to_Srxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc9_Nrf2_to_Srxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(3):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1500

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES5, pDF_Srxn1_Rmean_ES5)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2, 2)

    # Tests ES5
    run = 156
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(5)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES5, pDF_Srxn1_Rmean_ES5)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES5)
    NNconfig = 2
    for i in range(2):
        seed = i + 1
        tf.random.set_seed(seed=i + 1)

        epch = 1500

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES5, pDF_Srxn1_Rmean_ES5)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2, 3)

    # Tests ES5
    run = 157.1
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6, 7, 8])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(5)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES5, pDF_Srxn1_Rmean_ES5)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES5)
    NNconfig = 2
    for i in range(3):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 448

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES5, pDF_Srxn1_Rmean_ES5)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 4, 3)

    # ES5 tests
    run = 158
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([6, 7])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(5.2)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES5, pDF_Srxn1_Rmean_ES5)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES5)
    NNconfig = 2
    for i in range(9):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(20, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(40, return_sequences=True, activation='tanh'))
        model.add(LSTM(60, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2, 3)

    # ES5 tests
    run = 159
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([6, 7])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(5.2)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES5, pDF_Srxn1_Rmean_ES5)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES5)
    NNconfig = 2
    for i in range(9):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(60, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(60, return_sequences=True, activation='tanh'))
        model.add(LSTM(60, return_sequences=True, activation='tanh'))
        model.add(LSTM(60, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2, 3)

    # ES5 tests
    run = 160
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([6, 7])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(5.2)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES5, pDF_Srxn1_Rmean_ES5)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES5)
    NNconfig = 2
    for i in range(9):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 2000

        model = Sequential()
        model.add(
            LSTM(10, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(10, return_sequences=True, activation='tanh'))
        model.add(LSTM(10, return_sequences=True, activation='tanh'))
        model.add(LSTM(10, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2, 3)

    # ES5 tests
    run = 161
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([6, 7])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(5.2)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES5, pDF_Srxn1_Rmean_ES5)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES5)
    NNconfig = 2
    for i in range(9):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 2000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(100, return_sequences=True, activation='tanh'))
        model.add(LSTM(100, return_sequences=True, activation='tanh'))
        model.add(LSTM(100, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2, 3)

    # ES5 tests
    run = 162
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([6, 7])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(5.2)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES5, pDF_Srxn1_Rmean_ES5)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES5)
    NNconfig = 2
    for i in range(9):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(100, return_sequences=True, activation='tanh'))
        model.add(LSTM(100, return_sequences=True, activation='tanh'))
        model.add(LSTM(100, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2, 3)

    # ES5 tests
    run = 163
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(9):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 2000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(100, return_sequences=True, activation='tanh'))
        model.add(LSTM(100, return_sequences=True, activation='tanh'))
        model.add(LSTM(100, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2, 3)

    # ES5 tests
    run = 164
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([6, 7])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(5.2)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES5, pDF_Srxn1_Rmean_ES5)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES5)
    NNconfig = 2
    for i in range(9):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2, 3)

    # ES5 tests
    run = 165
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([6, 7])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(5.2)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES5, pDF_Srxn1_Rmean_ES5)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES5)
    NNconfig = 2
    for i in range(9):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 2000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plotting(trainX, trainY, NNconfig, 4, 2, 3)

    # ES1 tests
    run = 166
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(9):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plottingES1(trainX, trainY, NNconfig, 4, 2, 3)

    # SUL run
    run = 167
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist,
                                                     valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_SUL32)
    NNconfig = 2
    for i in range(3):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plottingES1(trainX, trainY, NNconfig, 4, 2, 3)

    # SUL run 2
    run = 168
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist,
                                                     valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_SUL32)
    NNconfig = 2
    for i in range(2):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 3000

        model = Sequential()
        model.add(
            LSTM(5, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(10, return_sequences=True, activation='tanh'))
        model.add(LSTM(10, return_sequences=True, activation='tanh'))
        model.add(LSTM(10, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plottingES1(trainX, trainY, NNconfig, 4, 2, 3)

    # SUL run 3
    run = 169
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist,
                                                     valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_SUL32)
    NNconfig = 2
    for i in range(2):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 3000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(100, return_sequences=True, activation='tanh'))
        model.add(LSTM(100, return_sequences=True, activation='tanh'))
        model.add(LSTM(100, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plottingES1(trainX, trainY, NNconfig, 4, 2, 3)

    run = 132.001
    traindoseindexlist = trainindex([1, 2, 3, 4])
    valdoseindexlist = valindex([5, 6])
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
    NNconfig = 2
    for i in range(9):
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=True, activation='tanh'))
        model.add(LSTM(50, return_sequences=False, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plottingES1(trainX, trainY, NNconfig, 4, 2, 3)
    # SUL size test, scaler test
    run = 170
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.1)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_SUL32)
    NNconfig = 2
    for i in range(2):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plottingES1(trainX, trainY, NNconfig, 4, 2, 3)

    # SUL size test, scaler test
    run = 171
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.12)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_SUL32)
    NNconfig = 2
    for i in range(2):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL size test, scaler test
    run = 172
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.12)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32)
    NNconfig = 2
    plt.clf()
    for i in range(2):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL size test, scaler test
    run = 173
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.12)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32)
    NNconfig = 2
    plt.clf()
    for i in range(2):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL size test, scaler test
    run = 174
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.12)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32)
    NNconfig = 2
    plt.clf()
    for i in range(2):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL size test, scaler test
    run = 175
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.12)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32)
    NNconfig = 2
    plt.clf()
    for i in range(2):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL size test
    run = 176
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.12)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32)
    NNconfig = 2
    plt.clf()
    for i in range(2):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(150, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        # model.add(
        #     LSTM(150, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
        #          activation='tanh'))
        # model.add(
        #     LSTM(150, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
        #          activation='tanh'))
        # model.add(
        #     LSTM(150, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
        #          activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL size test
    run = 177
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.12)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32)
    NNconfig = 2
    plt.clf()
    for i in range(2):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(150, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(150, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        # model.add(
        #     LSTM(150, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
        #          activation='tanh'))
        # model.add(
        #     LSTM(150, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
        #          activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL size test, ODE testing configuration (after dydt smoothing)
    run = 178
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.12)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32)
    NNconfig = 2
    plt.clf()
    for i in range(2):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL size test, ODE testing configuration (after dydt smoothing), adjusted scaler
    run = 179
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32)
    NNconfig = 2
    plt.clf()
    for i in range(2):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)

    # SUL size test, ODE testing configuration (after dydt smoothing), adjusted scaler, no time component
    run = 180
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1_notime(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32)
    NNconfig = 2
    plt.clf()
    for i in range(2):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)

    # SUL size test, ODE testing configuration (after dydt smoothing), adjusted scaler, no time component
    run = 181
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32)
    NNconfig = 2
    plt.clf()
    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 2000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL dydt smoothing tests
    run = 182
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32, 501, 5)
    NNconfig = 2
    plt.clf()
    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 2000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        SUL_integration_plotting()
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)




    # SUL dydt smoothing tests
    run = 183
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32, 1001, 5)
    NNconfig = 2
    plt.clf()
    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 2000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        SUL_integration_plotting()
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)

    # SUL dydt smoothing tests
    run = 184
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32, 1501, 9)
    NNconfig = 2
    plt.clf()
    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 2000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        SUL_integration_plotting()
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL dydt smoothing tests
    run = 185
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32, 1001, 9)
    NNconfig = 2
    plt.clf()
    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 2000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        SUL_integration_plotting()
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL dydt smoothing tests
    run = 186
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32, 501, 9)
    NNconfig = 2
    plt.clf()
    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 2000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        SUL_integration_plotting()
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL dydt smoothing tests
    run = 187
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32, 1501, 5)
    NNconfig = 2
    plt.clf()
    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 2000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        SUL_integration_plotting()
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)

    # SUL dydt smoothing tests
    run = 188
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32, 1501, 2)
    NNconfig = 2
    plt.clf()
    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 2000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        SUL_integration_plotting()
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL dydt smoothing tests
    run = 189
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32, 1901, 5)
    NNconfig = 2
    plt.clf()
    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 2000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        SUL_integration_plotting()
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL dydt smoothing tests
    run = 190
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32, 1701, 5)
    NNconfig = 2
    plt.clf()
    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 2000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        SUL_integration_plotting()
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL dydt smoothing tests
    run = 191
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32, 1501, 5)
    NNconfig = 2
    plt.clf()
    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 2000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        SUL_integration_plotting()
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL dydt smoothing tests
    run = 192
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32, 1501, 4)
    NNconfig = 2
    plt.clf()
    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 2000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        SUL_integration_plotting()
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL dydt smoothing tests
    run = 193
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32, 1501, 3)
    NNconfig = 2
    plt.clf()
    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 2000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        SUL_integration_plotting()
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL dydt smoothing tests
    run = 194
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32, 1501, 5)
    NNconfig = 2
    plt.clf()
    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 3000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        SUL_integration_plotting()
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL dydt smoothing tests
    run = 195
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32, 1501, 5)
    NNconfig = 2
    plt.clf()
    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 4000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        SUL_integration_plotting()
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)



    # SUL dydt smoothing tests
    run = 196
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32, 1501, 5)
    NNconfig = 2
    plt.clf()
    for i in range(1):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 6000

        model = Sequential()
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        SUL_integration_plotting()
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL dydt smoothing tests
    run = 197
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32, 1501, 5)
    NNconfig = 2
    plt.clf()
    for i in range(3):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        SUL_integration_plotting()
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)

    # SUL dydt smoothing tests
    run = 198
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32, 1501, 5)
    NNconfig = 2
    plt.clf()
    for i in range(3):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 2000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        SUL_integration_plotting()
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL dydt smoothing tests
    run = 199
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32, 1501, 5)
    NNconfig = 2
    plt.clf()
    for i in range(3):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 2000

        model = Sequential()
        model.add(
            LSTM(30, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(30, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(30, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(30, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        SUL_integration_plotting()
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)


    # SUL dydt smoothing tests
    run = 200
    traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54], 'SUL')
    valdoseindexlist = valindex([3.5], 'SUL')
    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist)
    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL32, 1501, 5)
    NNconfig = 2
    plt.clf()
    for i in range(3):
        print('seed = ' + str(i))
        seed = i
        tf.random.set_seed(seed=i)

        epch = 1000

        model = Sequential()
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(
            LSTM(100, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                 activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                            validation_data=(testX, testY), verbose=2,
                            shuffle=False)

        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
        posttraining(model, history)
        SUL_integration_plotting()
        # plottingES1(trainX, trainY, NNconfig, 4, 2, 3)

# Experimental multiple exposure tests (not part of final results)
for h in range(1):
    multiexpindex(0.35,35.00,'SUL')
    multiexpindex(35.00,35.00,'SUL')
    exposure1list = [0, 0.35]
    exposure2list = [0.35, 0.35]

    def gen_trainvalindexlistME(exposure1list, exposure2list, compound):
        indexlist = []
        for i in range(len(exposure1list)):
            indexlist.append(multiexpindex(exposure1list[i],exposure2list[i], compound))
        return indexlist

    trainexposure1list = [0,    0.35, 35.00]
    trainexposure2list = [0.35, 0.35, 0.35 ]
    valexposure1list = [0,    0.35, ]
    valexposure2list = [0.35, 0.35, ]

    traindoseindexlist = gen_trainvalindexlistME(trainexposure1list, trainexposure2list, 'SUL')
    valdoseindexlist = gen_trainvalindexlistME(valexposure1list, valexposure2list, 'SUL')

    # run 225, initial multiple exposure tests
    for h in range(1):
        # Multiple exposure training
        run = 225
        traindoseindexlist = [1,28,34]
        valdoseindexlist = [28,33]
        datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
        trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_SUL_8_24_ME, pDF_Srxn1_Rmean_SUL_8_24_ME, traindoseindexlist,
                                                         valdoseindexlist)
        trainY, testY = trainYfunc6_Nrf2_Srxn1_to_smoothed_dSrxn1(pDF_Srxn1_Rmean_SUL_8_24_ME, 1501, 5)
        plt.clf()
        NNconfig = 2
        # MSEmatrix = np.zeros((7, 4))
        MSEmatrix1 = np.zeros((7, 16))
        MSEmatrix2 = np.zeros((7, 16))
        plt.plot(trainY[:,0,0])
        # MSEmatrix = np.zeros((len(traindoseindexlist + valdoseindexlist + 1), 4))
        for i in range(3):
            print('seed = ' + str(i))
            seed = i
            tf.random.set_seed(seed=i)

            epch = 1000

            model = Sequential()
            model.add(
                LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                     activation='tanh'))
            model.add(
                LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                     activation='tanh'))
            model.add(
                LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                     activation='tanh'))
            model.add(
                LSTM(100, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                     activation='tanh'))
            model.add(Dense(1, activation='tanh'))
            model.compile(loss='mae', optimizer='adam')
            model.summary()
            history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                validation_data=(testX, testY), verbose=2,
                                shuffle=False)


        def ODE_integrator(Nrf2_data, y0Srxn1, timesteps):
            timelist = [listt[0, 0]]
            Nrf2list = [Nrf2_data[0]]
            Srxn1list = [y0Srxn1]
            dSrxn1list = []
            for i in range(timesteps):
                print('Integrating ' + str(i) + '/' + str(timesteps))
                Nrf2 = Nrf2list[-1]
                Nrf2 = Nrf2.reshape(1, 1)
                scaler.fit(datascalerlistNrf2)
                Nrf2 = scaler.transform(Nrf2)
                Srxn1 = Srxn1list[-1]
                Srxn1 = Srxn1.reshape(1, 1)
                scaler.fit(datascalerlistSrxn1)
                Srxn1 = scaler.transform(Srxn1)
                tensor = np.zeros((1, 1, 3))
                tensor[0, 0, 0] = timelist[-1]
                tensor[0, 0, 1] = Nrf2[0, 0]
                tensor[0, 0, 2] = Srxn1[0, 0]
                print(tensor)
                predic = model.predict(tensor)
                dSrxn1list.append(predic[0, 0])
                scaler.fit(diffscalerlist)
                predic = scaler.inverse_transform(predic)
                # scaler.fit(datascalerlistSrxn1)
                # predic = scaler.inverse_transform(predic)
                # predic = 10**predic
                nextSrxn1 = Srxn1list[-1]
                scaler.fit(datascalerlistSrxn1)
                nextSrxn1 = nextSrxn1.reshape(1, 1)
                nextSrxn1 = scaler.transform(nextSrxn1)
                nextSrxn1 = nextSrxn1[0, 0] + predic[0]
                nextSrxn1 = nextSrxn1.reshape(1, 1)
                nextSrxn1 = scaler.inverse_transform(nextSrxn1)
                Srxn1list.append(nextSrxn1[0])
                print(predic)
                timelist.append(listt[(i + 1), 0])
                Nrf2list.append(Nrf2_data[(i + 1)])
            # def integration_loss():
            #     integratedSrxn1_diff = np.diff()
            return Srxn1list, dSrxn1list, Nrf2list, timelist

        integratedtrain1, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL_8_24_ME[:, 1],
                                                                pDF_Srxn1_Rmean_SUL_8_24_ME[0, 1], 1998)
        integratedtrain2, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL_8_24_ME[:, 28],
                                                                pDF_Srxn1_Rmean_SUL_8_24_ME[0, 28], 1998)
        integratedtrain3, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL_8_24_ME[:, 34],
                                                                pDF_Srxn1_Rmean_SUL_8_24_ME[0, 34], 1998)
        integratedtest1, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL_8_24_ME[:, 28],
                                                                pDF_Srxn1_Rmean_SUL_8_24_ME[0, 28], 1998)
        integratedtest2, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL_8_24_ME[:, 33],
                                                                pDF_Srxn1_Rmean_SUL_8_24_ME[0, 33], 1998)
        integratedunseen1, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL_8_24_ME[:, 18],
                                                                pDF_Srxn1_Rmean_SUL_8_24_ME[0, 18], 1998)
        plt.plot(integratedtrain1, color='green', label='0, 0.75')
        plt.plot(pDF_Srxn1_Rmean_SUL_8_24_ME[:, 1], color='green', linestyle='dashed', label='0, 0.75')
        plt.plot(integratedtrain2, color=lighten_color('g', 0.8), label='1.62, 3.5')
        plt.plot(pDF_Srxn1_Rmean_SUL_8_24_ME[:, 28], color=lighten_color('g', 0.8), linestyle='dashed', label='1.62, 3.5')
        plt.plot(integratedtrain3, color=lighten_color('g', 0.3), label='3.5, 0.75')
        plt.plot(pDF_Srxn1_Rmean_SUL_8_24_ME[:, 34], color=lighten_color('g', 0.3), linestyle='dashed', label='3.5, 0.75')

        plt.plot(integratedtest2, color='orange', label='')
        plt.plot(pDF_Srxn1_Rmean_SUL_8_24_ME[:, 33], color='orange', linestyle='dashed', label='3.5, 0.35')
        plt.plot(integratedunseen1, color='red', label='0.75, 0.75')
        plt.plot(pDF_Srxn1_Rmean_SUL_8_24_ME[:, 18], color='red', linestyle='dashed', label='0.75, 0.75')
        plt.legend()

    # Improved indexing, baseline addition multiple exposure tests
    for h in range(1):
        # Multiple exposure training
        run = 226
        trainexposure1list = [0.35, 0.75, 1.62, 3.5, 7.54, 0.35]
        trainexposure2list = [0.35, 0.75, 1.62, 3.5, 7.54, 0.75]
        valexposure1list = [0.35, 7.54]
        valexposure2list = [7.54, 0.35]
        traindoseindexlist = gen_trainvalindexlistME(trainexposure1list, trainexposure2list, 'SUL')
        valdoseindexlist = gen_trainvalindexlistME(valexposure1list, valexposure2list, 'SUL')
        datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
        trainX, testX = trainXfunc12_Nrf2_Srxn1_to_dSrxn1_ME(pDF_Nrf2_Rmean_SUL_8_24_ME, pDF_Srxn1_Rmean_SUL_8_24_ME,traindoseindexlist, valdoseindexlist,
                                                                      include_baseline=True)
        trainY, testY = trainYfunc12_Nrf2_Srxn1_to_smoothed_dSrxn1_ME(pDF_Srxn1_Rmean_SUL_8_24_ME, 1501, 5,
                                                                      include_baseline=True)

        NNconfig = 2
        # MSEmatrix = np.zeros((7, 4))
        # MSEmatrix1 = np.zeros((7, 16))
        # MSEmatrix2 = np.zeros((7, 16))
        # plt.plot(trainY[:, 0, 0])
        # MSEmatrix = np.zeros((len(traindoseindexlist + valdoseindexlist + 1), 4))
        for i in range(1):
            print('seed = ' + str(i))
            seed = i
            tf.random.set_seed(seed=i)

            epch = 3000

            model = Sequential()
            model.add(
                LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                     activation='tanh'))
            model.add(
                LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                     activation='tanh'))
            model.add(
                LSTM(50, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                     activation='tanh'))
            model.add(
                LSTM(50, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                     activation='tanh'))
            model.add(Dense(1, activation='tanh'))
            model.compile(loss='mae', optimizer='adam')
            model.summary()
            history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                validation_data=(testX, testY), verbose=2,
                                shuffle=False)
            posttraining(model, history)


        def ODE_integrator(Nrf2_data, y0Srxn1, timesteps, include_time=True):
            timelist = [listt[0, 0]]
            Nrf2list = [Nrf2_data[0]]
            Srxn1list = [y0Srxn1]
            dSrxn1list = []
            for i in range(timesteps):
                print('Integrating ' + str(i) + '/' + str(timesteps))
                Nrf2 = Nrf2list[-1]
                Nrf2 = Nrf2.reshape(1, 1)
                scaler.fit(datascalerlistNrf2)
                Nrf2 = scaler.transform(Nrf2)
                Srxn1 = Srxn1list[-1]
                if len(Srxn1list) > 1:
                    Srxn1 = Srxn1.reshape(1, 1)
                scaler.fit(datascalerlistSrxn1)
                Srxn1 = scaler.transform(Srxn1)
                if include_time == True:
                    tensor = np.zeros((1, 1, 3))
                    tensor[0, 0, 0] = timelist[-1]
                    tensor[0, 0, 1] = Nrf2[0, 0]
                    tensor[0, 0, 2] = Srxn1[0, 0]
                if include_time == False:
                    tensor = np.zeros((1, 1, 2))
                    tensor[0, 0, 0] = Nrf2[0, 0]
                    tensor[0, 0, 1] = Srxn1[0, 0]
                print(tensor)
                predic = model.predict(tensor)
                dSrxn1list.append(predic[0, 0])
                scaler.fit(diffscalerlist)
                predic = scaler.inverse_transform(predic)
                # scaler.fit(datascalerlistSrxn1)
                # predic = scaler.inverse_transform(predic)
                # predic = 10**predic
                nextSrxn1 = Srxn1list[-1]
                scaler.fit(datascalerlistSrxn1)
                nextSrxn1 = nextSrxn1.reshape(1, 1)
                nextSrxn1 = scaler.transform(nextSrxn1)
                nextSrxn1 = nextSrxn1[0, 0] + predic[0]
                nextSrxn1 = nextSrxn1.reshape(1, 1)
                nextSrxn1 = scaler.inverse_transform(nextSrxn1)
                Srxn1list.append(nextSrxn1[0])
                print(predic)
                timelist.append(listt[(i + 1), 0])
                Nrf2list.append(Nrf2_data[(i + 1)])
                # def integration_loss():
                #     integratedSrxn1_diff = np.diff()
            return Srxn1list, dSrxn1list, Nrf2list, timelist


        model = load_model(str(directory) + 'Versions/NN2/run226 seed=3.h5')


        traindoseindexlist
        valdoseindexlist
        for h in range(1):
            integratedtrain1, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL_8_24_ME[:, 9],
                                                                    pDF_Srxn1_Rmean_SUL_8_24_ME[0, 9], 1998)
            integratedtrain2, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL_8_24_ME[:, 18],
                                                                    pDF_Srxn1_Rmean_SUL_8_24_ME[0, 18], 1998)
            integratedtrain3, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL_8_24_ME[:, 27],
                                                                    pDF_Srxn1_Rmean_SUL_8_24_ME[0, 27], 1998)
            integratedtrain4, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL_8_24_ME[:, 36],
                                                                     pDF_Srxn1_Rmean_SUL_8_24_ME[0, 36], 1998)
            integratedtest1, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL_8_24_ME[:, 45],
                                                                    pDF_Srxn1_Rmean_SUL_8_24_ME[0, 45], 1998)
            integratedtest2, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL_8_24_ME[:, 36],
                                                                    pDF_Srxn1_Rmean_SUL_8_24_ME[0, 36], 1998)
            integratedunseen1, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL_8_24_ME[:, 13],
                                                                    pDF_Srxn1_Rmean_SUL_8_24_ME[0, 13], 1998)
            integratedunseen2, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL_8_24_ME[:, 41],
                                                                    pDF_Srxn1_Rmean_SUL_8_24_ME[0, 41], 1998)
        zero = np.array(0)
        zero = zero.reshape(1,1)
        for h in range(1):
            integratedtrain1, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL_8_24_ME[:, 9],zero, 1998, include_time=False)
            integratedtrain2, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL_8_24_ME[:, 18],
                                                                    zero, 1998, include_time=False)
            integratedtrain3, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL_8_24_ME[:, 27],
                                                                    zero, 1998)
            integratedtrain4, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL_8_24_ME[:, 36],
                                                                     zero, 1998)
            integratedtest1, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL_8_24_ME[:, 45],
                                                                    zero, 1998)
            integratedtest2, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL_8_24_ME[:, 36],
                                                                    zero, 1998)
            integratedunseen1, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL_8_24_ME[:, 13],
                                                                    zero, 1998)
            integratedunseen2, dSrxn1control, nrf2, time = ODE_integrator(pDF_Nrf2_Rmean_SUL_8_24_ME[:, 41], zero, 1998)


        plt.plot(integratedtrain1, color='green', label='0, 0.75', linestyle='dashed')
        plt.plot(pDF_Srxn1_Rmean_SUL_8_24_ME[:, 9], color='green', label='0, 0.75')
        plt.plot(integratedtrain2, color=lighten_color('g', 0.8), label='1.62, 3.5', linestyle='dashed')
        plt.plot(pDF_Srxn1_Rmean_SUL_8_24_ME[:, 18], color=lighten_color('g', 0.8), label='1.62, 3.5')
        plt.plot(integratedtrain3, color=lighten_color('g', 0.5), label='3.5, 0.75', linestyle='dashed')
        plt.plot(pDF_Srxn1_Rmean_SUL_8_24_ME[:, 27], color=lighten_color('g', 0.5), label='3.5, 0.75')
        plt.plot(integratedtrain4, color=lighten_color('g', 0.3), label='3.5, 0.75', linestyle='dashed')
        plt.plot(pDF_Srxn1_Rmean_SUL_8_24_ME[:, 36], color=lighten_color('g', 0.3), label='3.5, 0.75')
        plt.plot(integratedtest1, color=lighten_color('g', 0.7), label='1.62, 3.5', linestyle='dashed')
        plt.plot(pDF_Srxn1_Rmean_SUL_8_24_ME[:, 45], color=lighten_color('g', 0.8), label='1.62, 3.5')



        plt.plot(integratedtest1, color='orange', label='', linestyle='dashed')
        plt.plot(pDF_Srxn1_Rmean_SUL_8_24_ME[:, 49], color='orange', label='3.5, 0.35')
        plt.plot(integratedtest2, color='orange', label='', linestyle='dashed')
        plt.plot(pDF_Srxn1_Rmean_SUL_8_24_ME[:, 36], color='orange', label='3.5, 0.35')
        plt.plot(integratedunseen1, color='red', label='0.75, 0.75', linestyle='dashed')
        plt.plot(pDF_Srxn1_Rmean_SUL_8_24_ME[:, 13], color='red', label='0.75, 0.75')
        plt.plot(integratedunseen2, color='red', label='0.75, 0.75', linestyle='dashed')
        plt.plot(pDF_Srxn1_Rmean_SUL_8_24_ME[:, 41], color='red', label='0.75, 0.75')
        plt.legend()


        # data training
        run = 227
        traindoseindexlist = trainindex([0.35, 0.75, 1.62, 7.54, 16.25], 'SUL')
        valdoseindexlist = valindex([3.5], 'SUL')
        datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(0.13)
        trainX, testX = trainXfunc12_Nrf2_Srxn1_to_dSrxn1_ME(pDF_Nrf2_Rmean_SUL32, pDF_Srxn1_Rmean_SUL32, traindoseindexlist, valdoseindexlist, include_baseline=True, include_time=True)
        trainY, testY = trainYfunc12_Nrf2_Srxn1_to_smoothed_dSrxn1_ME(pDF_Srxn1_Rmean_SUL32,1501, 5, include_baseline=True)

        NNconfig = 2
        # MSEmatrix = np.zeros((7, 4))
        MSEmatrix1 = np.zeros((7, 16))
        MSEmatrix2 = np.zeros((7, 16))

        # MSEmatrix = np.zeros((len(traindoseindexlist + valdoseindexlist + 1), 4))
        for i in range(15):
            print('seed = ' + str(i))
            seed = i
            tf.random.set_seed(seed=i)

            epch = 1000

            model = Sequential()
            model.add(
                LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                     activation='tanh'))
            model.add(
                LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                     activation='tanh'))
            model.add(
                LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                     activation='tanh'))
            model.add(
                LSTM(100, return_sequences=False, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                     activation='tanh'))
            model.add(Dense(1, activation='tanh'))
            model.compile(loss='mae', optimizer='adam')
            model.summary()
            history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                validation_data=(testX, testY), verbose=2,
                                shuffle=False)

            # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)

            # MSEmatrix1 = np.array(MSEmatrix1)
            posttraining(model, history)
            seedMSE = SUL_integration_plotting(True)
            if i == 0:
                MSEmatrix1[1:7, i] = seedMSE[:, 0]
            MSEmatrix1[0, i + 1] = seed
            MSEmatrix1[1:7, i + 1] = seedMSE[:, 1]
            np.shape(MSEmatrix1)


            # MSEmatrix2 = np.array(MSEmatrix2)
            seedMSE = CDDOtoSUL_integration_plotting(True)
            if i == 0:
                MSEmatrix2[1:7, i] = seedMSE[:, 0]
            MSEmatrix2[0, i + 1] = seed
            MSEmatrix2[1:7, i + 1] = seedMSE[:, 1]

        MSEmatrix1 = pd.DataFrame(MSEmatrix1)
        MSEmatrix1.to_csv(str(directory) + 'MSE matrix/run' + str(run) + 'SUL .txt')

        MSEmatrix2 = pd.DataFrame(MSEmatrix2)
        MSEmatrix2.to_csv(str(directory) + 'MSE matrix/run' +str(run) + 'CDDO .txt')

# Old design (runs 97-127)
for h in range(1):
    # Seed looped neural network
    Use_specificseed = False
    specific_seed = 1

    run = 97 + runiter
    # epch = 5000
    nseeds = 3
    second_run = False

    maxPredictime = 32
    minPredictime = 0
    for k in range(1):
        if Use_specificseed == True:
            nseeds = 1
            for i in range(specific_seed + 1):
                tf.random.set_seed(seed=i)
                seed = i
                print(seed)
        print('run=' + str(run))
        # print('epochs=' + str(epch))
        print('nseeds=' + str(nseeds))
        # lossdict = {}
        # val_lossdict = {}
        for i in range(nseeds):
            # Multiple nseeds for training
            if Use_specificseed == False:
                seed = i
                tf.random.set_seed(seed=i)

            # # compiling model

            # Run 68
            # if runiter == 0:
            #     model = Sequential()
            #     model.add(LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1],np.shape(trainX)[2]), activation='tanh'))
            #     model.add(LSTM(10, return_sequences=True, activation='tanh'))
            #     model.add(LSTM(10, return_sequences=True, activation='tanh'))
            #     model.add(LSTM(1, return_sequences=True, activation='tanh'))
            #     model.compile(loss='mse', optimizer='adam')
            #     model.summary()
            #     history = model.fit(trainX, trainY, epochs=epch, batch_size=1999, validation_data=(testX, testY), verbose=2,
            #                         shuffle=False)
            # Run 69
            # if runiter == 1:
            #     model = Sequential()
            #     model.add(LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1],np.shape(trainX)[2]), activation='tanh'))
            #     model.add(LSTM(50, return_sequences=True, activation='tanh'))
            #     model.add(LSTM(50, return_sequences=True, activation='tanh'))
            #     model.add(LSTM(1, return_sequences=True, activation='tanh'))
            #     model.compile(loss='mse', optimizer='adam')
            #     model.summary()
            #     history = model.fit(trainX, trainY, epochs=epch, batch_size=1999, validation_data=(testX, testY), verbose=2,
            #                         shuffle=False)
            # Run 70
            # if runiter == 0:
            #     model = Sequential()
            #     model.add(LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1],np.shape(trainX)[2]), activation='tanh'))
            #     model.add(LSTM(50, return_sequences=True, activation='tanh'))
            #     model.add(LSTM(50, return_sequences=True, activation='tanh'))
            #     model.add(LSTM(1, return_sequences=False, activation='tanh'))
            #     model.add(Dense(1, activation='tanh'))
            #     model.compile(loss='mse', optimizer='adam')
            #     model.summary()
            #     history = model.fit(trainX, trainY, epochs=epch, batch_size=1999, validation_data=(testX, testY), verbose=2,
            #                         shuffle=False)
            #
            # # Run 71
            # if runiter == 1:
            #     model = Sequential()
            #     model.add(LSTM(30, return_sequences=True, input_shape=(np.shape(trainX)[1],np.shape(trainX)[2]), activation='tanh'))
            #     model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #     model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #     model.add(LSTM(30, return_sequences=False, activation='tanh'))
            #     model.add(Dense(1, activation='tanh'))
            #     model.compile(loss='mse', optimizer='adam')
            #     model.summary()
            #     history = model.fit(trainX, trainY, epochs=epch, batch_size=1999, validation_data=(testX, testY), verbose=2,
            #                         shuffle=False)
            #     # Run 72
            # if runiter == 2:
            #     model = Sequential()
            #     model.add(LSTM(10, return_sequences=True, input_shape=(np.shape(trainX)[1],np.shape(trainX)[2]), activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(10, return_sequences=True, activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(10, return_sequences=True, activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(10, return_sequences=False, activation='tanh'))
            #     model.add(Dense(1, activation='tanh'))
            #     model.compile(loss='mse', optimizer='adam')
            #     model.summary()
            #     history = model.fit(trainX, trainY, epochs=epch, batch_size=1999, validation_data=(testX, testY), verbose=2,
            #                         shuffle=False)
            #     # Run 73
            # if runiter == 3:
            #     model = Sequential()
            #     model.add(LSTM(5, return_sequences=True, input_shape=(np.shape(trainX)[1],np.shape(trainX)[2]), activation='tanh'))
            #     model.add(LSTM(5, return_sequences=True, activation='tanh'))
            #     model.add(LSTM(5, return_sequences=True, activation='tanh'))
            #     model.add(LSTM(5, return_sequences=False, activation='tanh'))
            #     model.add(Dense(1, activation='tanh'))
            #     model.compile(loss='mse', optimizer='adam')
            #     model.summary()
            #     history = model.fit(trainX, trainY, epochs=epch, batch_size=1999, validation_data=(testX, testY), verbose=2,
            #                         shuffle=False)
            #
            #     # Run 74
            # if runiter == 0:
            #     model = Sequential()
            #     model.add(LSTM(10, return_sequences=True, input_shape=(np.shape(trainX)[1],np.shape(trainX)[2]), activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(10, return_sequences=True, activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(10, return_sequences=True, activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(10, return_sequences=False, activation='tanh'))
            #     model.add(Dense(1, activation='tanh'))
            #     model.compile(loss='mse', optimizer='adam')
            #     model.summary()
            #     history = model.fit(trainX, trainY, epochs=epch, batch_size=1999, validation_data=(testX, testY), verbose=2,
            #                         shuffle=False)
            #
            #     # Run 75
            # if runiter == 1:
            #     model = Sequential()
            #     model.add(LSTM(30, return_sequences=True, input_shape=(np.shape(trainX)[1],np.shape(trainX)[2]), activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(30, return_sequences=False, activation='tanh'))
            #     model.add(Dense(1, activation='tanh'))
            #     model.compile(loss='mse', optimizer='adam')
            #     model.summary()
            #     history = model.fit(trainX, trainY, epochs=epch, batch_size=1999, validation_data=(testX, testY), verbose=2,
            #                         shuffle=False)
            # run 76
            # if runiter == 0:
            #     model = Sequential()
            #     model.add(LSTM(5, return_sequences=True, input_shape=(np.shape(trainX)[1],np.shape(trainX)[2]), activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(1, return_sequences=True, activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(1, return_sequences=True, activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(1, return_sequences=False, activation='tanh'))
            #     model.add(Dense(1, activation='tanh'))
            #     model.compile(loss='mse', optimizer='adam')
            #     model.summary()
            #     history = model.fit(trainX, trainY, epochs=epch, batch_size=1999, validation_data=(testX, testY), verbose=2,
            #                         shuffle=False)
            #
            # # run 77
            # if runiter == 1:
            #     model = Sequential()
            #     model.add(LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1],np.shape(trainX)[2]), activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(1, return_sequences=False, activation='tanh'))
            #     model.add(Dense(1, activation='tanh'))
            #     model.compile(loss='mse', optimizer='adam')
            #     model.summary()
            #     history = model.fit(trainX, trainY, epochs=epch, batch_size=1999, validation_data=(testX, testY), verbose=2,
            #                         shuffle=False)
            # # run 78
            # if runiter == 2:
            #     model = Sequential()
            #     model.add(LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1],np.shape(trainX)[2]), activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(1, return_sequences=False, activation='tanh'))
            #     model.add(Dense(1, activation='tanh'))
            #     model.compile(loss='mse', optimizer='adam')
            #     model.summary()
            #     history = model.fit(trainX, trainY, epochs=epch, batch_size=1000, validation_data=(testX, testY), verbose=2,
            #                         shuffle=False)
            #
            # # run 79
            # if runiter == 3:
            #         model = Sequential()
            #         model.add(LSTM(30, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
            #                        activation='tanh'))
            #         model.add(Dropout(0.25))
            #         model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #         model.add(Dropout(0.25))
            #         model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #         model.add(Dropout(0.25))
            #         model.add(LSTM(30, return_sequences=False, activation='tanh'))
            #         model.add(Dense(1, activation='tanh'))
            #         model.compile(loss='mse', optimizer='adam')
            #         model.summary()
            #         history = model.fit(trainX, trainY, epochs=epch, batch_size=50, validation_data=(testX, testY),
            #                             verbose=2,
            #                             shuffle=False)

            # run 80 - 1500 epochs
            # if runiter == 0:
            #         model = Sequential()
            #         model.add(LSTM(30, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
            #                        activation='tanh'))
            #         model.add(Dropout(0.25))
            #         model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #         model.add(Dropout(0.25))
            #         model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #         model.add(Dropout(0.25))
            #         model.add(LSTM(30, return_sequences=False, activation='tanh'))
            #         model.add(Dense(1, activation='tanh'))
            #         model.compile(loss='mse', optimizer='adam')
            #         model.summary()
            #         history = model.fit(trainX, trainY, epochs=epch, batch_size=1, validation_data=(testX, testY), verbose=2,
            #                             shuffle=False)

            # Run 81/82/83
            # if runiter == 0:
            #     model = Sequential()
            #     model.add(LSTM(30, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
            #                    activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(30, return_sequences=False, activation='tanh'))
            #     model.add(Dense(1, activation='tanh'))
            #     model.compile(loss='mse', optimizer='adam')
            #     model.summary()
            #     history = model.fit(trainX, trainY, epochs=epch, batch_size=1000, validation_data=(testX, testY), verbose=2,
            #                         shuffle=False)

            # Run 84
            # if runiter == 0:
            #     model = Sequential()
            #     model.add(LSTM(30, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
            #                    activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(30, return_sequences=False, activation='tanh'))
            #     model.add(Dense(1, activation='tanh'))
            #     model.compile(loss='mse', optimizer='adam')
            #     model.summary()
            #     history = model.fit(trainX, trainY, epochs=epch, batch_size=1000, validation_data=(testX, testY),verbose=2,
            #                         shuffle=False)
            # # Run 85
            # if runiter == 1:
            #     model = Sequential()
            #     model.add(LSTM(30, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
            #                    activation='tanh'))
            #     model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #     model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #     model.add(LSTM(30, return_sequences=False, activation='tanh'))
            #     model.add(Dense(1, activation='tanh'))
            #     model.compile(loss='mse', optimizer='adam')
            #     model.summary()
            #     history = model.fit(trainX, trainY, epochs=epch, batch_size=1000, validation_data=(testX, testY),verbose=2,
            #                         shuffle=False)
            #
            # # Run 86
            # if runiter == 2:
            #     model = Sequential()
            #     model.add(LSTM(30, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
            #                    activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(1, return_sequences=False, activation='tanh'))
            #     model.add(Dense(1, activation='tanh'))
            #     model.compile(loss='mse', optimizer='adam')
            #     model.summary()
            #     history = model.fit(trainX, trainY, epochs=epch, batch_size=1000, validation_data=(testX, testY),verbose=2,
            #                         shuffle=False)
            #
            # # Run 87
            # if runiter == 3:
            #     model = Sequential()
            #     model.add(LSTM(30, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
            #                    activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(30, return_sequences=False, activation='tanh'))
            #     model.add(Dense(1, activation='tanh'))
            #     model.compile(loss='mse', optimizer='adam')
            #     model.summary()
            #     history = model.fit(trainX, trainY, epochs=epch, batch_size=50, validation_data=(testX, testY),verbose=2,
            #                         shuffle=False)
            #
            # # Run 88
            # if runiter == 4:
            #     model = Sequential()
            #     model.add(LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
            #                    activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(100, return_sequences=True, activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(100, return_sequences=True, activation='tanh'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(100, return_sequences=False, activation='tanh'))
            #     model.add(Dense(1, activation='tanh'))
            #     model.compile(loss='mse', optimizer='adam')
            #     model.summary()
            #     history = model.fit(trainX, trainY, epochs=epch, batch_size=1000, validation_data=(testX, testY),verbose=2,
            #                         shuffle=False)
            #
            # # Run 89
            # if runiter == 5:
            #     model = Sequential()
            #     model.add(LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),activation='tanh'))
            #     model.add(LSTM(100, return_sequences=True, activation='tanh'))
            #     model.add(LSTM(100, return_sequences=True, activation='tanh'))
            #     model.add(LSTM(100, return_sequences=False, activation='tanh'))
            #     model.add(Dense(1, activation='tanh'))
            #     model.compile(loss='mse', optimizer='adam')
            #     model.summary()
            #     history = model.fit(trainX, trainY, epochs=epch, batch_size=1000, validation_data=(testX, testY),verbose=2,
            #                         shuffle=False)
            #
            # # Run 90
            # if runiter == 6:
            #     model = Sequential()
            #     model.add(LSTM(30, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
            #                    activation='relu'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(30, return_sequences=True, activation='relu'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(30, return_sequences=True, activation='relu'))
            #     model.add(Dropout(0.25))
            #     model.add(LSTM(30, return_sequences=False, activation='relu'))
            #     model.add(Dense(1, activation='relu'))
            #     model.compile(loss='mse', optimizer='adam')
            #     model.summary()
            #     history = model.fit(trainX, trainY, epochs=epch, batch_size=1000, validation_data=(testX, testY),verbose=2,
            #                         shuffle=False)

            # Run 91
            # if runiter == 0:
            #     trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_36, pDF_Srxn1_Rmean_36)
            #     trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_36)
            #     NNconfig = 2
            #
            #     for i in range(3):
            #         # Multiple nseeds for training
            #         seed = i
            #         tf.random.set_seed(seed=i)
            #
            #         epch = 1500
            #
            #         model = Sequential()
            #         model.add(LSTM(30, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),activation='tanh'))
            #         model.add(Dropout(0.25))
            #         model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #         model.add(Dropout(0.25))
            #         model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #         model.add(Dropout(0.25))
            #         model.add(LSTM(30, return_sequences=False, activation='tanh'))
            #         model.add(Dense(1, activation='tanh'))
            #         model.compile(loss='mse', optimizer='adam')
            #         model.summary()
            #         history = model.fit(trainX, trainY, epochs=epch, batch_size=50, validation_data=(testX, testY), verbose=2,shuffle=False)
            #
            #         posttraining(model, history)
            #         plotting(trainX, trainY, NNconfig)

            # Run 92
            # if runiter == 0:
            #     trainX, testX = trainXfunc2(pDF_Nrf2_Rmean_36, pDF_Srxn1_Rmean_36)
            #     trainY, testY = trainYfunc2(pDF_Srxn1_Rmean_36)
            #     NNconfig = 1
            #
            #     for i in range(1):
            #         # Multiple nseeds for training
            #         seed = i
            #         tf.random.set_seed(seed=i)
            #
            #         epch = 2000
            #
            #         model = Sequential()
            #         model.add(LSTM(30, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
            #                        activation='tanh'))
            #         model.add(Dropout(0.25))
            #         model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #         model.add(Dropout(0.25))
            #         model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #         model.add(Dropout(0.25))
            #         model.add(LSTM(30, return_sequences=True, activation='tanh'))
            #         model.add(LSTM(1, return_sequences=True, activation='tanh'))
            #         model.compile(loss='mse', optimizer='adam')
            #         model.summary()
            #         history = model.fit(trainX, trainY, epochs=epch, batch_size=1000, validation_data=(testX, testY),
            #                             verbose=2, shuffle=False)
            #
            #
            #         posttraining(model, history)
            #         plotting(trainX, trainY, NNconfig)
            #
            # # Run 93
            # if runiter == 1:
            #     trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_36, pDF_Srxn1_Rmean_36)
            #     trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_36)
            #     NNconfig = 2
            #
            #     for i in range(6):
            #         seed = i
            #         tf.random.set_seed(seed=i)
            #
            #         if seed == 1:
            #             epch = 4300
            #         else:
            #             epch = 4500
            #
            #
            #         model = Sequential()
            #         model.add(LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
            #                        activation='tanh'))
            #         model.add(Dropout(0.25))
            #         model.add(LSTM(100, return_sequences=True, activation='tanh'))
            #         model.add(Dropout(0.25))
            #         model.add(LSTM(100, return_sequences=True, activation='tanh'))
            #         model.add(Dropout(0.25))
            #         model.add(LSTM(100, return_sequences=False, activation='tanh'))
            #         model.add(Dense(1, activation='tanh'))
            #         model.compile(loss='mse', optimizer='adam')
            #         model.summary()
            #         history = model.fit(trainX, trainY, epochs=epch, batch_size=1000, validation_data=(testX, testY),verbose=2,
            #                             shuffle=False)
            #
            #         posttraining(model, history)
            #         plotting(trainX, trainY, NNconfig)
            #
            # # Run 94
            # if runiter == 2:
            #
            #     trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_36, pDF_Srxn1_Rmean_36)
            #     trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_36)
            #     NNconfig = 2
            #
            #     # Multiple nseeds for training
            #     for i in range(3):
            #         seed = i
            #         tf.random.set_seed(seed=i)
            #
            #         epch = 3000 + 2000 * i
            #
            #         model = Sequential()
            #         model.add(
            #             LSTM(200, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
            #                  activation='tanh'))
            #         model.add(Dropout(0.25))
            #         model.add(LSTM(200, return_sequences=True, activation='tanh'))
            #         model.add(Dropout(0.25))
            #         model.add(LSTM(200, return_sequences=True, activation='tanh'))
            #         model.add(Dropout(0.25))
            #         model.add(LSTM(200, return_sequences=False, activation='tanh'))
            #         model.add(Dense(1, activation='tanh'))
            #         model.compile(loss='mse', optimizer='adam')
            #         model.summary()
            #         history = model.fit(trainX, trainY, epochs=epch, batch_size=1000, validation_data=(testX, testY), verbose=2, shuffle=False)
            #
            #         posttraining(model, history)
            #         plotting(trainX, trainY, NNconfig)

            # Run 95
            if runiter == 0:
                trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_36, pDF_Srxn1_Rmean_36)
                trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_36)
                NNconfig = 2
                run = 97

                for i in range(6):
                    seed = i
                    tf.random.set_seed(seed=i)

                    epch = 5000

                    model = Sequential()
                    model.add(
                        LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                             activation='tanh'))
                    model.add(Dropout(0.25))
                    model.add(LSTM(100, return_sequences=True, activation='tanh'))
                    model.add(Dropout(0.25))
                    model.add(LSTM(100, return_sequences=True, activation='tanh'))
                    model.add(Dropout(0.25))
                    model.add(LSTM(100, return_sequences=False, activation='tanh'))
                    model.add(Dense(1, activation='tanh'))
                    model.compile(loss='mse', optimizer='adam')
                    model.summary()
                    history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                        validation_data=(testX, testY), verbose=2,
                                        shuffle=False)

                    posttraining(model, history)
                    plotting(trainX, trainY, NNconfig)

                    # model.add(Dropout(0.25))
                    # model.add(Dropout(0.25))
                    # model.add(LSTM(50, return_sequences=True, activation='tanh'))
                    # model.add(Dropout(0.1))
                    # model.add(Dropout(0.1))
                    # model.add(LSTM(20, return_sequences=True, activation='tanh'))
                    # model.add(Dropout(0.1))
                    # model.add(Dense(10, activation='tanh'))
                    # model.add(Dropout(0.6))
                    # model.add(Dense(10, activation='tanh'))
                    # model.add(Dense(10, activation='tanh'))
                    # model.add(Dropout(0.6))
                    # model.add(LSTM(10, return_sequences=True, activation='tanh'))
                    # model.add(LSTM(10, kernel_regularizer=regularizers.l2(0.001), return_sequences=True, activation='tanh'))
                    # model.add(Dropout(0.5))
                    # model.add(LSTM(1, return_sequences=False, activation='tanh'))
                    # model.add(Dense(1, activation='tanh'))


                    # model = load_model(str(directory) + 'Versions/NN2/run51 seed=1 loss=10^-2.66.h5')

                    # history = model.fit(trainX, trainY, epochs=epch, batch_size=1999, validation_data=(testX, testY), verbose=2,
                    #                     shuffle=False)

                    # NN size tests
                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    run = 98
                    for i in range(6):
                        seed = i
                        tf.random.set_seed(seed=i)

                        epch = 5000

                        model = Sequential()
                        model.add(
                            LSTM(120, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(120, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(120, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(120, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mse', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig)

                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    run = 99
                    for i in range(6):
                        seed = i
                        tf.random.set_seed(seed=i)

                        epch = 5000

                        model = Sequential()
                        model.add(
                            LSTM(80, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(80, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(80, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(80, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mse', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig)

                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    run = 100
                    for i in range(6):
                        seed = i
                        tf.random.set_seed(seed=i)

                        epch = 5000

                        model = Sequential()
                        model.add(
                            LSTM(140, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(140, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(140, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(140, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mse', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig)

                    # Dropout tests
                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    run = 101
                    for i in range(4):
                        seed = i
                        tf.random.set_seed(seed=i)

                        epch = 5000

                        model = Sequential()
                        model.add(
                            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.10))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.10))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.10))
                        model.add(LSTM(100, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mse', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig)

                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    run = 102
                    for i in range(4):
                        seed = i
                        tf.random.set_seed(seed=i)

                        epch = 5000

                        model = Sequential()
                        model.add(
                            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.5))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.5))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.5))
                        model.add(LSTM(100, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mse', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig)

                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    run = 103
                    for i in range(4):
                        seed = i
                        tf.random.set_seed(seed=i)

                        epch = 5000

                        model = Sequential()
                        model.add(
                            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.01))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.01))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.01))
                        model.add(LSTM(100, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mse', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig)

                    # Note: many past runs (up to 98 and further in the past?) have been run with the Nrf2 scaler used for the testX Srxn1 data. Some of these tests may need to be redone.
                    # ODE integrator test run
                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    run = 104.2
                    for i in range(1):
                        seed = i
                        tf.random.set_seed(seed=i)

                        epch = 2000

                        model = Sequential()
                        model.add(
                            LSTM(120, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(120, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(120, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(120, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mse', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig)

                    # Nrf2 to dSrxn1 test
                    trainX, testX = trainXfunc7_Nrf2_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc7_Nrf2_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    run = 105
                    for i in range(1):
                        seed = i
                        tf.random.set_seed(seed=i)

                        epch = 5000

                        model = Sequential()
                        model.add(
                            LSTM(120, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(120, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(120, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(120, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mse', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        posttraining(model, history)
                        plotting_Nrf2_to_dSrxn1(trainX, trainY, NNconfig)

                    # Srxn1 train/test scaled by Nrf2 scaler test
                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    run = 107
                    for i in range(1):
                        seed = i
                        tf.random.set_seed(seed=i)

                        epch = 5000

                        model = Sequential()
                        model.add(
                            LSTM(120, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mse', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig)

                    # Proper scalers, scaler config 1
                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    run = 108
                    for i in range(1):
                        seed = i + 1
                        tf.random.set_seed(seed=i + 1)

                        epch = 2000

                        model = Sequential()
                        model.add(
                            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mse', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig)

                    # wrong scalers, scaler config 1
                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    run = 110
                    for i in range(1):
                        seed = i + 1
                        tf.random.set_seed(seed=i + 1)

                        epch = 250

                        model = Sequential()
                        model.add(
                            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mse', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig)

                    # proper scalers, scaler config 1
                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    run = 111.2
                    for i in range(1):
                        seed = i
                        tf.random.set_seed(seed=i)

                        epch = 1000

                        model = Sequential()
                        model.add(
                            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mse', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig)

                    # proper scalers, scaler config 1, empty Srxn1 vector
                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    run = 112
                    for i in range(1):
                        seed = i
                        tf.random.set_seed(seed=i)

                        epch = 1000

                        model = Sequential()
                        model.add(
                            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mse', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig)

                    # faulty scalers, scaler config 1, empty Srxn1 vector
                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1_backup(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1_backup(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    run = 113
                    for i in range(1):
                        seed = i
                        tf.random.set_seed(seed=i)

                        epch = 4500

                        model = Sequential()
                        model.add(
                            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mse', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)

                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig)

                    # proper scalers, scaler config 1, empty Srxn1 vector, dose 2 as test
                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1_backup(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1_backup(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    run = 114
                    for i in range(1):
                        seed = i
                        tf.random.set_seed(seed=i)

                        epch = 4500

                        model = Sequential()
                        model.add(
                            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mse', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)

                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig)

                    # proper scalers, scaler config 1, empty Srxn1 vector, dose 2 as test
                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1_backup(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1_backup(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    run = 115
                    for i in range(5):
                        seed = i + 1
                        tf.random.set_seed(seed=i + 1)

                        epch = 8000

                        model = Sequential()
                        model.add(
                            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mse', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)

                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig, 4, 4)

                    run = 116
                    traindoseindexlist = trainindex([1, 2, 3, 4])
                    valdoseindexlist = valindex([2, 4, 5, 6])
                    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
                    trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    for i in range(2):
                        seed = i
                        tf.random.set_seed(seed=i)

                        epch = 5000

                        model = Sequential()
                        model.add(
                            LSTM(10, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(10, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(10, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(10, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mse', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)

                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig, 4, 4)

                    # MAE instead of MSE
                    run = 117
                    traindoseindexlist = trainindex([1, 2, 3, 4])
                    valdoseindexlist = valindex([2, 4, 5, 6])
                    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
                    trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    for i in range(2):
                        seed = i
                        tf.random.set_seed(seed=i)

                        epch = 5000

                        model = Sequential()
                        model.add(
                            LSTM(10, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(10, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(10, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(10, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mae', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)

                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig, 4, 4)

                    run = 118.3
                    traindoseindexlist = trainindex([1, 2, 3, 4])
                    valdoseindexlist = valindex([2, 4, 5, 6])
                    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
                    trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    for i in range(2):
                        seed = i + 1
                        tf.random.set_seed(seed=i + 1)

                        epch = 10000

                        model = Sequential()
                        model.add(
                            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mae', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)

                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig, 4, 4)

                    run = 119
                    traindoseindexlist = trainindex([1, 2, 3, 4])
                    valdoseindexlist = valindex([2, 4, 5, 6])
                    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
                    trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    for i in range(2):
                        seed = i + 1
                        tf.random.set_seed(seed=i + 1)

                        epch = 10000

                        model = Sequential()
                        model.add(
                            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        # model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        # model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        # model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mae', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)

                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig, 4, 4)

                    run = 120
                    traindoseindexlist = trainindex([1, 2, 3, 4])
                    valdoseindexlist = valindex([5, 6])
                    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    for i in range(2):
                        seed = i
                        tf.random.set_seed(seed=i)

                        epch = 5000

                        model = Sequential()
                        model.add(
                            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mae', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig, 4, 2)

                    run = 120.1
                    traindoseindexlist = trainindex([1, 2, 3, 4])
                    valdoseindexlist = valindex([5, 6])
                    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    for i in range(2):
                        seed = i
                        tf.random.set_seed(seed=i)

                        epch = 319

                        model = Sequential()
                        model.add(
                            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mae', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig, 4, 2)

                    # Length test 1/2
                    run = 121
                    traindoseindexlist = trainindex([1, 2, 3, 4])
                    valdoseindexlist = valindex([5, 6])
                    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    for i in range(4):
                        seed = i
                        tf.random.set_seed(seed=i)

                        epch = 500

                        model = Sequential()
                        model.add(
                            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mae', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig, 4, 2)

                    # Length test 2/2
                    run = 122
                    traindoseindexlist = trainindex([1, 2, 3, 4])
                    valdoseindexlist = valindex([5, 6])
                    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    for i in range(4):
                        seed = i
                        tf.random.set_seed(seed=i)

                        epch = 500

                        model = Sequential()
                        model.add(
                            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mae', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig, 4, 2)

                    # Dropout tests 1/2
                    run = 123
                    traindoseindexlist = trainindex([1, 2, 3, 4])
                    valdoseindexlist = valindex([5, 6])
                    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    for i in range(3):
                        seed = i
                        tf.random.set_seed(seed=i)

                        epch = 500

                        model = Sequential()
                        model.add(
                            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        # model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        # model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        # model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mae', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig, 4, 2)

                    # Dropout tests 2/2
                    run = 124
                    traindoseindexlist = trainindex([1, 2, 3, 4])
                    valdoseindexlist = valindex([5, 6])
                    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    for i in range(3):
                        seed = i
                        tf.random.set_seed(seed=i)

                        epch = 500

                        model = Sequential()
                        model.add(
                            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(Dropout(0.25))
                        model.add(LSTM(100, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mae', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig, 4, 2)

                    # More length tests 1/3
                    run = 125
                    traindoseindexlist = trainindex([1, 2, 3, 4])
                    valdoseindexlist = valindex([5, 6])
                    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    for i in range(6):
                        seed = i + 3
                        tf.random.set_seed(seed=i + 3)

                        epch = 500

                        model = Sequential()
                        model.add(
                            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(LSTM(100, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mae', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig, 4, 2)

                    # More length tests 2/3
                    run = 126
                    traindoseindexlist = trainindex([1, 2, 3, 4])
                    valdoseindexlist = valindex([5, 6])
                    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    for i in range(6):
                        seed = i + 3
                        tf.random.set_seed(seed=i + 3)

                        epch = 500

                        model = Sequential()
                        model.add(
                            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(LSTM(100, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mae', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig, 4, 2)

                    # More length tests 3/3
                    run = 127
                    traindoseindexlist = trainindex([1, 2, 3, 4])
                    valdoseindexlist = valindex([5, 6])
                    datascalerlistNrf2, datascalerlistSrxn1, diffscalerlist, diffscalerlistNrf2 = scaler_config(1)
                    trainX, testX = trainXfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                    trainY, testY = trainYfunc6_Nrf2_Srxn1_to_dSrxn1(pDF_Srxn1_Rmean_ES1)
                    NNconfig = 2
                    for i in range(6):
                        seed = i + 3
                        tf.random.set_seed(seed=i + 3)

                        epch = 500

                        model = Sequential()
                        model.add(
                            LSTM(100, return_sequences=True, input_shape=(np.shape(trainX)[1], np.shape(trainX)[2]),
                                 activation='tanh'))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(LSTM(100, return_sequences=True, activation='tanh'))
                        model.add(LSTM(100, return_sequences=False, activation='tanh'))
                        model.add(Dense(1, activation='tanh'))
                        model.compile(loss='mae', optimizer='adam')
                        model.summary()
                        history = model.fit(trainX, trainY, epochs=epch, batch_size=1000,
                                            validation_data=(testX, testY), verbose=2,
                                            shuffle=False)

                        # trainX, testX = trainXfunc6_Nrf2_emptySrxn1_to_dSrxn1(pDF_Nrf2_Rmean_ES1, pDF_Srxn1_Rmean_ES1)
                        posttraining(model, history)
                        plotting(trainX, trainY, NNconfig, 4, 2)

