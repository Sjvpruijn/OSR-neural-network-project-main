import numpy as np
import pandas as pd

def dataprocessingtrans(dataset, dtraining=False, logscaled_training=True):
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

    return Urange, Lrange, Rmean

# Dose pair replicates
exp1list = [0,35,75,162,350,754,1625,3500]
exp2list = [0,35,75,162,350,754,1625,3500]
exp2listshort = [35,75,162,350,754,1625,3500]
pDF_Nrf2_Rmean_SUL_8_24_ME = np.zeros((2000,len(exp1list)*len(exp2list)))
for exp1 in range(len(exp1list)):
    if exp1list[exp1] == 0:
        for exp2 in range(len(exp2listshort)):
            dosepairDF = np.zeros((2000,6))
            for br in range(3):
                for tr in range(2):
                    print(exp1)
                    dosepairDF[:,(2*br + tr)] = np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/Causal model/Data/Training/Multiexp/Nrf2 SUL DN 8-24h BR' + str(br+1) + '/' + str(exp1list[exp1]) + ' ' + str(exp2listshort[exp2]) + '  tr' +str(tr+1) + '.txt'))[:, 1]
                    print('check1')
            urange, lrange, Rmean = dataprocessingtrans(dosepairDF, logscaled_training=True)
            pDF_Nrf2_Rmean_SUL_8_24_ME[:,exp2 + exp1*len(exp2list)] = Rmean[:,0]
            # print(exp1list[exp1] + ' ' + exp2list[exp2])
    else:
        print(exp1)
        for exp2 in range(len(exp2list)):
            dosepairDF = np.zeros((2000,6))
            for br in range(3):
                for tr in range(2):
                        dosepairDF[:,(2*br + tr)] = np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/Causal model/Data/Training/Multiexp/Nrf2 SUL DN 8-24h BR' + str(br+1) + '/' + str(exp1list[exp1]) + ' ' + str(exp2list[exp2]) + '  tr' +str(tr+1) + '.txt'))[:, 1]
            urange, lrange, Rmean = dataprocessingtrans(dosepairDF, logscaled_training=True)
            pDF_Nrf2_Rmean_SUL_8_24_ME[:,exp2 + exp1*len(exp2list)] = Rmean[:,0]
            # print(exp1list[exp1] + ' ' + exp2list[exp2])

pDF_Srxn1_Rmean_SUL_8_24_ME = np.zeros((2000,len(exp1list)*len(exp2list)))
for exp1 in range(len(exp1list)):
    if exp1list[exp1] == 0:
        for exp2 in range(len(exp2listshort)):
            dosepairDF = np.zeros((2000,6))
            for br in range(3):
                for tr in range(2):
                    print(exp1)
                    dosepairDF[:,(2*br + tr)] = np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/Causal model/Data/Training/Multiexp/Srxn1 SUL DN 8-24h BR' + str(br+1) + '/' + str(exp1list[exp1]) + ' ' + str(exp2listshort[exp2]) + '  tr' +str(tr+1) + '.txt'))[:, 1]
                    print('check1')
            urange, lrange, Rmean = dataprocessingtrans(dosepairDF, logscaled_training=True)
            pDF_Srxn1_Rmean_SUL_8_24_ME[:,exp2 + exp1*len(exp2list)] = Rmean[:,0]
            # print(exp1list[exp1] + ' ' + exp2list[exp2])
    else:
        print(exp1)
        for exp2 in range(len(exp2list)):
            dosepairDF = np.zeros((2000,6))
            for br in range(3):
                for tr in range(2):
                        dosepairDF[:,(2*br + tr)] = np.array(pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/Causal model/Data/Training/Multiexp/Srxn1 SUL DN 8-24h BR' + str(br+1) + '/' + str(exp1list[exp1]) + ' ' + str(exp2list[exp2]) + '  tr' +str(tr+1) + '.txt'))[:, 1]
            urange, lrange, Rmean = dataprocessingtrans(dosepairDF, logscaled_training=True)
            pDF_Srxn1_Rmean_SUL_8_24_ME[:,exp2 + exp1*len(exp2list)] = Rmean[:,0]
            # print(exp1list[exp1] + ' ' + exp2list[exp2])

