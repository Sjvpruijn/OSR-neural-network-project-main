import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import linregress
from scipy import interpolate
import pandas as pd
from pandas import read_csv
from pandas import read_fwf
from sklearn.preprocessing import MinMaxScaler


# BR1 = pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/Causal model/Data/Raw/20190710_Unilever3_expU6_NRF2_8_24_n1_SummaryData.txt', sep="	")
# BR2 = pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/Causal model/Data/Raw/20190722_Unilever3_expU6_NRF2_8_24_n2_SummaryData.txt', sep="	")
# BR3 = pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/Causal model/Data/Raw/20190731_Unilever3_expU6_NRF2_8_24_n3_SummaryData.txt', sep="	")
BR1 = pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/Causal model/Data/Raw/20190710_Unilever3_expU6_NRF2_8_24_n1_SummaryData.txt', sep="	")
BR2 = pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/Causal model/Data/Raw/20190722_Unilever3_expU6_NRF2_8_24_n2_SummaryData.txt', sep="	")
BR3 = pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/Causal model/Data/Raw/20190731_Unilever3_expU6_NRF2_8_24_n3_SummaryData.txt', sep="	")
BR1 = pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/Causal model/Data/Raw/20190605_unilever3_expU4_SRXN1_8_24_n1_SummaryData.txt', sep="	")
BR2 = pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/Causal model/Data/Raw/20190617_Unilever3_expU4_SRXN1_8_24_n2_SummaryData.txt', sep="	")
BR3 = pd.read_csv('E:/BFW/Master BPS/RP1/Neural Network/Causal model/Data/Raw/20190626_Unilever3_expU1_SRXN1_8_24_n3_SummaryData.txt', sep="	")

BRlist = [BR1, BR2, BR3]

# Treatment = Andrographolide, Ethacrynic Acid, DMSO, CDDO-me, Sulforaphane, DMEM
treatment = "Sulforaphane"
# Measurement = GFP or PI
measurement = 'GFP'
# Protein = Nrf2 or Srxn1
protein = 'Srxn1'
# Plot results of data filtering-smoothing-interpolation
plotting = True
# Export data files to drive as .txt (check directory at the bottom of the file/for-loop)
export = False
# Export directory: file path where the final datafiles (.txt files) are to be placed
exportdir = 'E:/BFW/Master BPS/RP1/Neural Network/Causal model/Data/filetest/'
data = BRlist[2].loc[BRlist[2]["treatment"] == treatment]

for f in range(3):
    plt.clf()
    data = BRlist[f].loc[BRlist[f]["treatment"] == treatment]
    # BRlist[f].loc[BRlist[f]["control"] == "ContinuousExposure"]
    # data = data.loc[data["variable"] == "Nuclei_Intensity_IntegratedIntensity_image_GFP"]
    if measurement == 'PI':
        data = data.loc[data["variable"] == "count_PI_masked_primaryID_AreaShape_Area.DIV.Nuclei_AreaShape_Area_larger_0.1_"]
    if measurement == 'GFP':
        if protein == 'Nrf2':
            data = data.loc[data["variable"] == "Nuclei_Intensity_IntegratedIntensity_image_GFP"]
        if protein == 'Srxn1':
            data = data.loc[data["variable"] == "Cytoplasm_Intensity_IntegratedIntensity_image_GFP"]
    data = data.loc[data["control"] == "ContinuousExposure"]
    data.dose_uM.unique()
    LocIDs = data.locationID.unique()
    doses = np.array(data.dose_uM.unique())
    data.locationID.unique()
    for i in range(len(data.dose_uM.unique())):
        data_tr1 = data.loc[data["locationID"] == LocIDs[i*2]]
        data_tr1 = np.array(data_tr1.loc[:,"value"])
        data_tr1 = savgol_filter(data_tr1, 9, 2)

        data_tr2 = data.loc[data["locationID"] == LocIDs[i*2+1]]
        data_tr2 = np.array(data_tr2.loc[:,"value"])
        data_tr2 = savgol_filter(data_tr2, 9, 2)

        DMSO = BRlist[f].loc[BRlist[f]["treatment"] == "DMSO"]
        if measurement == 'PI':
            DMSO = DMSO.loc[DMSO["variable"] == "count_PI_masked_primaryID_AreaShape_Area.DIV.Nuclei_AreaShape_Area_larger_0.1_"]
        if measurement == 'GFP':
            if protein == 'Nrf2':
                DMSO = DMSO.loc[DMSO["variable"] == "Nuclei_Intensity_IntegratedIntensity_image_GFP"]
            if protein == 'Srxn1':
                DMSO = DMSO.loc[DMSO["variable"] == "Cytoplasm_Intensity_IntegratedIntensity_image_GFP"]
        DMSO = DMSO.loc[DMSO["control"] == "ContinuousExposure"]
        # Might need to change in between datasets
        DMSO1 = np.array(DMSO.loc[DMSO["locationID"] == DMSO.locationID.unique()[4]])[:,10]
        DMSO2 = np.array(DMSO.loc[DMSO["locationID"] == DMSO.locationID.unique()[5]])[:,10]
        DMSO = np.array((DMSO1 + DMSO2)/2)
        DMSO = savgol_filter(DMSO, 9, 2)

        # Fitting to start at 0, tr1
        # Determine DMSO subtraction
        data_tr1_smoothed = (data_tr1 - DMSO)[0:32]
        data_tr1_smoothed = data_tr1_smoothed.reshape(len(data_tr1_smoothed),1)
        scaler = MinMaxScaler(feature_range=(0, np.amax(data_tr1_smoothed)-np.amin(data_tr1_smoothed)))
        scaler.fit(data_tr1_smoothed)
        data_tr1_smoothed = scaler.transform(data_tr1_smoothed)
        data_tr1_smoothed = data_tr1_smoothed.reshape(len(data_tr1_smoothed),)
        # plt.plot(data_tr1_smoothed)

        # Fitting to start at 0, tr2
        # data_tr2_smoothed = data_tr2 #for without DMSO
        # data_tr2_smoothed = savgol_filter(data_tr2_smoothed, 9, 5) #for without DMSO
        data_tr2_smoothed = (data_tr2 - DMSO)[0:32]
        data_tr2_smoothed = data_tr2_smoothed.reshape(len(data_tr2_smoothed),1)
        scaler = MinMaxScaler(feature_range=(0, np.amax(data_tr2_smoothed)-np.amin(data_tr2_smoothed)))
        scaler.fit(data_tr2_smoothed)
        data_tr2_smoothed = scaler.transform(data_tr2_smoothed)
        data_tr2_smoothed = data_tr2_smoothed.reshape(len(data_tr2_smoothed),)
        # plt.plot(data_tr2_smoothed)



        # Interpolation
        interpolation_length = 2000
        hours = 32
        interpolation_length = np.linspace(1,hours,interpolation_length)
        time = []
        for j in range(32):
            j = j + 1
            time.append(j)
        data_tr1_smoothed = interpolate.interp1d(time, data_tr1_smoothed, kind='cubic')(interpolation_length)
        data_tr2_smoothed = interpolate.interp1d(time, data_tr2_smoothed, kind='cubic')(interpolation_length)


        colorlist = ['green', 'red', 'blue']
        # Exporting
        if plotting == True:
            plt.plot(data_tr1_smoothed, label='BR'+str(f+1)+' TR1',
                     color=colorlist[f])
            plt.plot(data_tr2_smoothed, label='BR'+str(f+1)+' TR2',
                     color=colorlist[f])
            plt.legend()

        data_tr1_smoothed = pd.DataFrame(data_tr1_smoothed)
        data_tr2_smoothed = pd.DataFrame(data_tr2_smoothed)

        if export == True:

            data_tr1_smoothed.to_csv(str(exportdir) + str(protein) + ' ' + str(treatment) + ' ' + str(np.shape(data_tr1_smoothed)[0])+ 'dp ' + str(doses[i]) + 'uM br' + str(f+1) + '_tr1.txt')
            data_tr2_smoothed.to_csv(str(exportdir) + str(protein) + ' ' + str(treatment) + ' ' + str(np.shape(data_tr2_smoothed)[0])+ 'dp ' + str(doses[i]) + 'uM br' + str(f+1) + '_tr2.txt')

            # data_tr1_smoothed.to_csv('E:/BFW/Master BPS/RP1/Neural Network/Causal model/Data/filetest/' + str(protein) + ' ' + str(treatment) + ' ' + str(np.shape(data_tr1_smoothed)[0])+ 'dp n' +str(f+1)+ '/' + str(doses[i]) + '_tr1.txt')
            # data_tr2_smoothed.to_csv('E:/BFW/Master BPS/RP1/Neural Network/Causal model/Data/filetest/' + str(protein) + ' ' + str(treatment) + ' ' + str(np.shape(data_tr2_smoothed)[0])+ 'dp n' +str(f+1)+ '/' + str(doses[i]) + '_tr2.txt')

            # data_tr1_smoothed.to_csv('E:/BFW/Master BPS/RP1/Neural Network/Causal model/Data/filetest/' + str(protein) + ' ' + str(treatment) + ' ' + str(np.shape(data_tr1_smoothed)[0])+ 'dp ' + str(doses[i]) + 'uM br' + str(f+1) + '_tr1.txt')
            # data_tr2_smoothed.to_csv('E:/BFW/Master BPS/RP1/Neural Network/Causal model/Data/filetest/' + str(protein) + ' ' + str(treatment) + ' ' + str(np.shape(data_tr2_smoothed)[0])+ 'dp ' + str(doses[i]) + 'uM br' + str(f+1) + '_tr2.txt')
            # if measurement == 'PI':
            #     data_tr1_smoothed.to_csv('E:/BFW/Master BPS/RP1/Neural Network/Causal model/Data/Training/Nrf2cd CDDO DMSO normalized 2000 n' +str(f+1)+ '/' + str(doses[i]) + '_tr1.txt')
            #     data_tr2_smoothed.to_csv('E:/BFW/Master BPS/RP1/Neural Network/Causal model/Data/Training/Nrf2cd CDDO DMSO normalized 2000 n' +str(f+1)+ '/' + str(doses[i]) + '_tr2.txt')
            # else:
            #     data_tr1_smoothed.to_csv('E:/BFW/Master BPS/RP1/Neural Network/Causal model/Data/Training/Nrf2 CDDO DMSO normalized 2000 n' +str(f+1)+ '/' + str(doses[i]) + '_tr1.txt')
            #     data_tr2_smoothed.to_csv('E:/BFW/Master BPS/RP1/Neural Network/Causal model/Data/Training/Nrf2 CDDO DMSO normalized 2000 n' +str(f+1)+ '/' + str(doses[i]) + '_tr2.txt')



