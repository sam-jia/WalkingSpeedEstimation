import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.integrate as it
from statsmodels.nonparametric.smoothers_lowess import lowess

data = sys.argv[1]
df = pd.read_csv(data, sep=';', header=0)
df['velocity_x']=0
df['velocity_y']=0
df=df.shift(periods=1, axis=0, fill_value=0).reset_index()

'''
SOS Butterworth FIlter
sos_x = signal.butter(5, [0.1, 0.8], btype='band', output='sos')
df['filtered_x'] = signal.sosfilt(sos_x, df['x'])
smoothed= lowess(df['x'], df.index, frac=0.008) #smooth 0.008
df['filtered_x']=smoothed[:, 1]

sos_y = signal.butter(5, [0.05, 0.4], btype='band', output='sos')
df['filtered_y'] = signal.sosfilt(sos_y, df['y'])
smoothed_y= lowess(df['y'], df.index, frac=0.006) #smooth 0.008
df['filtered_y'] =smoothed_y[:, 1]
'''

#Butterworth Filer
b_x, a_x = signal.butter(3, 0.192, btype='lowpass', analog=False)  #0.2
low_passed_x = signal.filtfilt(b_x, a_x, df['x'])
df['filtered_x']=low_passed_x 

b_y, a_y = signal.butter(3, 0.192, btype='lowpass', analog=False)  #0.19
low_passed_y = signal.filtfilt(b_y, a_y, df['y'])
df['filtered_y']=low_passed_y 


'''
#LOWESS Filter
smoothed = lowess(df['x'], df.index, frac=0.008) #smooth 0.008
df['filtered_x']=smoothed[:, 1]

smoothed_y= lowess(df['y'], df.index, frac=0.006) #smooth 0.008
df['filtered_y'] =smoothed_y[:, 1]
'''

df['group']=0

row, col = df.shape

def split_col(df):
    flag = 0
    row, col = df.shape
    for i in range(1,row):
        if df.loc[i].at['level_0'] % 25 != 0:
            df.loc[df.index==i, 'group'] = flag
        else:
            #flag += 1
            df.loc[df.index==i, 'group'] = flag
            flag += 1
    return df
        
def find_vol(df):
    row, col = df.shape
    for i in range (0, row-1):

        if (df.loc[i].at['filtered_y'] * df.loc[i+1].at['filtered_y']) < 0 or (df.loc[i].at['filtered_x'] * df.loc[i+1].at['filtered_x']) < 0:
            df.loc[df.index==i, 'velocity_x'] = 0
            df.loc[df.index==i, 'velocity_y'] = 0
        
        #df.loc[df.index==i+1, 'velocity_y'] = df.loc[i].at['velocity_y']+(1*(df.loc[i+1].at['filtered_y']))
        #df.loc[df.index==i+1, 'velocity_x'] = df.loc[i].at['velocity_x']+(14*(df.loc[i+1].at['filtered_x']))
        df.loc[df.index==i+1, 'velocity_x'] = (((df.loc[i+1].at['filtered_x']+df.loc[i].at['filtered_x'])/2)*0.04) + df.loc[i].at['velocity_x']
        df.loc[df.index==i+1, 'velocity_y'] = (((df.loc[i+1].at['filtered_y']+df.loc[i].at['filtered_y'])/2)*0.04) + df.loc[i].at['velocity_y']

    return df

velocity_df = find_vol(df)
velocity_df_extra = find_vol(df)

velocity_df['angular_velocity']=np.sqrt(np.square(velocity_df['velocity_x']) + np.square(velocity_df['velocity_y']))

velocity_df = split_col(velocity_df)

velocity_df = velocity_df.groupby(['group'])['angular_velocity'].sum().reset_index()


average_speed = velocity_df['angular_velocity'].sum() / (velocity_df.shape[0]) #m/s / s 1010 
print("average speed is: ",average_speed)

#rint(velocity_df)
#velocity_df.to_csv('acceleration')


plt.title('Walking Speed Graph (Foot)')
plt.xlabel('Time (s)')
plt.ylabel('Speed(m/s)')

plt.plot(velocity_df.index, velocity_df['angular_velocity'], 'r-', alpha=0.5, label='Walking Speed')

#plt.plot(velocity_df_extra.index, velocity_df_extra['x'], 'g.', alpha=0.5, label='Raw Acceleration Data')
#plt.plot(velocity_df_extra.index, velocity_df_extra['filtered_x'], 'r-', alpha=0.5, label = 'Filtered Acceleration Data')

#plt.plot(velocity_df_extra.index, velocity_df_extra['y'], 'g.', alpha=0.5, label='Raw Acceleration Data')
#plt.plot(velocity_df_extra.index, velocity_df_extra['filtered_y'], 'r-', alpha=0.5, label = 'Filtered Acceleration Data')
plt.legend()

#plt.show()
plt.savefig('Velocity_Graph.png')


