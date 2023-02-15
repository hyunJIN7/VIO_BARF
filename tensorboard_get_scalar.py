"""
https://www.tensorflow.org/tensorboard/dataframe_api?hl=ko
"""
import numpy as np
from packaging import version
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
from bisect import bisect_left
from scipy.interpolate import UnivariateSpline


def nearest_index(s,ts):
    # Given a presorted list of timestamps:  s = sorted(index)
    time_list = list(map(lambda t: abs(ts - t), s))
    return time_list.index(min(time_list))


experiment_id = "4yT62ac3SBqokgGKiKBJMw" # hallway or lounge
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars(pivot=False)
# hall_arkit = df[df.run.str.endswith("arkit/hallway_1_01")]
# hall_iphone = df[df.run.str.endswith("iphone/hallway_1_01")]

arkit_df = df[df.run.str.endswith("arkit/lounge_1_01")]
iphone_df = df[df.run.str.endswith("iphone/lounge_1_01")]

step=[i for i in range(0,20200,400)]

hall_arkit_value=arkit_df['value'].to_numpy()
hall_arkit_step=arkit_df['step'].to_numpy()
hall_iphone_value=iphone_df["value"].to_numpy()
hall_iphone_step=iphone_df['step'].to_numpy()

# w=np.isnan(plot_iphone_data)
# y[w]=0

plot_arkit_data = []
plot_iphone_data = []
# for i in range(len(step)):
#     index = nearest_index(hall_arkit_step , step[i])
#     plot_arkit_data.append(hall_arkit_value[index])
#     index = nearest_index(hall_iphone_step , step[i])
#     plot_iphone_data.append(hall_iphone_value[index])



# spl_arkit = UnivariateSpline(step,plot_arkit_data)
# spl_iphone = UnivariateSpline(step,plot_iphone_data)

fig = plt.figure(figsize=(18,10))
# plt.plot(step,plot_arkit_data,'cx-', label='Ours')
# plt.plot(step,plot_iphone_data,'rx-', label='BARF', markersize=5)
# plt.plot(step,spl_arkit(step),'cx-', label='Ours')
# plt.plot(step,spl_iphone(step),'rx-', label='BARF', markersize=5)

# hall_arkit_value=hall_arkit['value'].to_numpy()
# hall_arkit_step=hall_arkit['step'].to_numpy()
# hall_iphone_value=hall_iphone["value"].to_numpy()
# hall_iphone_step=hall_iphone['step'].to_numpy()


plt.plot(hall_arkit_step,hall_arkit_value,'cx-', label='Ours')
# plt.plot(hall_iphone_step,hall_iphone_value,'rx-', label='BARF', markersize=5)
plt.title("Hallway PSNR")
plt.xlabel('step')
plt.ylabel('PSNR')
# plt.xticks(step,range(0,))
plt.yticks(range(32),range(32))
plt.legend(loc='best')
plt.show()
fname = "{}.png".format("Hallway")
plt.savefig(fname, dpi=75)
# clean up
# plt.close('all')

