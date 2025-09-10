#importing libraries
import matplotlib.pyplot as plt
import heartpy as hp
import numpy as np
import pandas as pd


high_anxiety= [14,23,34,39,46,51,57, 72, 77,78, 80, 82,83, 84, 87, 88, 89, 91, 92, 93, 4,8,10,12, 15, 16, 18, 22, 27,29, 31, 32, 42, 45, 48, 66, 69]
cllabel= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
# print(len(high_anxiety))
wrist_features_ha = np.zeros((18,len(high_anxiety)))

#VISUALISING THE DATA===============================================================================================================

sample_rate=250
def load_visualise(datafile):

    plt.figure(figsize=(12,3))
    plt.plot(datafile[0][100000:110000], datafile[1][100000:110000])
    return plt.show()
    return ecg

#ecg = load_visualise('C:/Users/uttar/Box/ML fall 2021/Project/15176082/electrocardiogram_data/electrocardiogram_data/bug_box_task/high_anxiety_group/avoidance_response/P23_Heart.xlsx')
#ecg=  pd.DataFrame(pd.read_excel('C:/Users/uuk58-admin/Box/ML fall 2021/Project/15176082/electrocardiogram_data/electrocardiogram_data/bug_box_task/high_anxiety_group/avoidance_response/P23_Heart.xlsx'))

for j in range(len(high_anxiety)):
    ecg= pd.DataFrame(pd.read_csv('C:/Users/uttar/Box/ML fall 2021/Project/15176082/electrocardiogram_data/electrocardiogram_data/bug_box_task/P' + str(high_anxiety[j]) + '_Heart.csv'))
    ecg= np.transpose(ecg.to_numpy())
    #ecg[1]= (ecg[1]- min(ecg[0])/(max(ecg[1])-min(ecg[1]))) #normalised data
#load_visualise(ecg)

#FILTERING THE DATA========================================================================================================================

    def filter_and_visualise(data, sample_rate):
        
        filtered = hp.remove_baseline_wander(data, sample_rate)
        #scd= hp.scale_data(data[200:1200])
        #And let's plot both original and filtered signal, and zoom in to show peaks are not moved
        #We'll also scale both signals with hp.scale_data
        # #This is so that they have the same amplitude so that the overlap is better visible
        # plt.figure(figsize=(12,3))
        # plt.title('zoomed in signal with baseline wander removed, original signal overlaid')
        # plt.plot(ecg[0][100000:105000], hp.scale_data(data[100000:105000]), color= 'red')
        # plt.plot(ecg[0][100000:105000], hp.scale_data(filtered[100000:105000]))
        # plt.show()
        return filtered 
    #visualise the filtered data
    filtered = filter_and_visualise(ecg[1], sample_rate)
    #working till here --------------------------------------------------------------------------------------------------------------------
    
    #highpass filtering
    filtered_hp = hp.filter_signal(filtered, cutoff=0.667, filtertype= 'highpass', sample_rate=sample_rate)

#8 point moving average filter and plotting it
# window_size = 8
# numbers_series = pd.Series(filtered_hp)
# windows = numbers_series.rolling(window_size)
# moving_averages = windows.mean()
# moving_averages_list = moving_averages.tolist()
# filtered_ma = moving_averages_list[window_size - 1:]
# plt.figure(figsize=(12,3))
# plt.title('Moving average')
# plt.plot(ecg[0][103000:104000], hp.scale_data(filtered[103000:104000]), color= 'red')
# plt.plot(ecg[0][103000:104000], hp.scale_data(filtered_ma[103000:104000]))
# plt.show()

    filtered_ma= filtered_hp
    
    #bandpass 0.75-2.5 Hz
    filtered_bp1 = hp.filter_signal(filtered_ma, cutoff=[0.75, 2.5], sample_rate=31.25, order=3, filtertype= 'bandpass')
    # plt.figure(figsize=(12,3))
    # plt.title('low bp')
    # plt.plot(ecg[0][103000:104000], hp.scale_data(filtered[103000:104000]), color= 'red')
    # plt.plot(ecg[0][103000:104000], hp.scale_data(filtered_bp1[103000:104000]))
    # plt.show()
    #bandpass 8-50 Hz
    filtered_bp2 = hp.filter_signal(filtered_bp1, cutoff=[8.0, 50.0], filtertype= 'bandpass', sample_rate=sample_rate)
#plotting the final filtered data
# plt.figure(figsize=(12,3))
# plt.title('All filters applied')
# plt.plot(ecg[0][103000:104000], hp.scale_data(filtered[103000:104000]), color= 'red')
# plt.plot(ecg[0][103000:104000], hp.scale_data(filtered_bp2[103000:104000]))
# plt.show()

    #=======================================================================================================================================================================
    def interval_interest(p_num):
        K=int(0.25*len(ecg[0]))
        K2=int(0.5*len(ecg[0]))
        filcropped = filtered_bp2[K:K2]
        return filcropped
          
    filblcrop= (interval_interest(23))
    
    #PEAK DETECTION AND FEATURE EXTRACTION ROLLING WINDOW
    
    wd, m= hp.process(filblcrop, sample_rate)
    lis= list(m.items())
    mea= np.transpose(np.array(lis))
    for meas in range(13):
        wrist_features_ha[meas+5][j]= mea[1][meas]

#====================================================================================================================================================================================

#ankle data
for i in range(len(high_anxiety)):
    wristdata= pd.DataFrame(pd.read_csv('C:/Users/uttar/Box/ML fall 2021/Project/15176082/ankle_movement_data/ankle_movement_data/P' +str(high_anxiety[i]) + '_RightAnkle.csv'))
    wristdata= np.transpose(wristdata.to_numpy())
    spwrist= int(0.25*len(wristdata[1]))
    epwrist= int(0.4*len(wristdata[1]))
    #wristdata[1]= (wristdata[1]-min(wristdata[1]))/(max(wristdata[1])-min(wristdata[1]))
    #wristdata[2]= (wristdata[2]-min(wristdata[2]))/(max(wristdata[2])-min(wristdata[2]))
    #wristdata[3]= (wristdata[3]-min(wristdata[3]))/(max(wristdata[3])-min(wristdata[3]))
    wrist_features_ha[0][i] = np.mean(wristdata[1][spwrist:epwrist])
    wrist_features_ha[1][i]  =np.mean( wristdata[2][spwrist:epwrist])
    wrist_features_ha[2][i]  = np.mean(wristdata[3][spwrist:epwrist])
    
#np.save('C:/Users/uttar/Box/ML fall 2021/Project/15176082/features_wrist', wrist_features_ha)


#torso data
for i in range(len(high_anxiety)):
    torso= pd.DataFrame(pd.read_csv('C:/Users/uttar/Box/ML fall 2021/Project/15176082/torso_posture_and_activity_data/torso_posture_and_activity_data/bug_box_task/P' +str(high_anxiety[i]) + '_Posture_Activity.csv'))
    torso= np.transpose(torso.to_numpy())
    sptorso= int(0.25*len(torso[1]))
    eptorso= int(0.4*len(torso[1]))
    #torso[1]= (torso[1]-min(torso[1]))/(max(torso[1])-min(torso[1]))
    #torso[2]= (torso[2]-min(torso[2]))/(max(torso[2])-min(torso[2]))
    wrist_features_ha[3][i] = np.mean(torso[1][sptorso:eptorso])
    wrist_features_ha[4][i]  =np.mean( torso[2][sptorso:eptorso])

np.save('C:/Users/uttar/Box/ML fall 2021/Project/15176082/features_ankle_torso', wrist_features_ha)


# #===============================================================================================================================================================================

# #SAVING THE FILES

# # # load json module
# # import json

# # # create json object from dictionary
# # measures = json.dumps(m)

# # # open file for writing, "w" 
# # f = open("C:/Users/uttar/Box/ML fall 2021/Project/15176082/electrocardiogram_data/electrocardiogram_data/bug_box_task/high_anxiety_group/heartmeasures/P' + str(p_num) + '__heart_measures.json","w")

# # # write json object to file
# # f.write(measures)

# # # close file
# # f.close()

# np.save('C:/Users/uttar/Box/ML fall 2021/Project/15176082/features_ankle_torso_heart.npy', wrist_features_ha)


#==================================================================================================================================================

#implementing SVM algorithm

import numpy as np

cllabel= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
features= np.load('C:/Users/uttar/Box/ML fall 2021/Project/15176082/features_ankle_torso_heart.npy')
for ro in range(18):
    features[ro]= (features[ro]- min(features[ro]))/max(features[ro]-min(features[ro]))
features=np.delete(features, 16, 0)
features= np.transpose(features)
from sklearn.svm import SVC
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

clf = SVC(kernel='linear', C=1)
scores = cross_val_score(clf, features, cllabel, cv=10)
print(scores)


clf = SVC(kernel='rbf', C=1)
scores = cross_val_score(clf, features, cllabel, cv=10)
print(scores)
print(mean(scores))
print



