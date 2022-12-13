import csv
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d,make_interp_spline

import numpy as np
import matplotlib.ticker as ticker

def load_csv(fname):
    arkit_step = []
    arkit_psnr = []
    with open(fname) as File:
        Line_reader = csv.reader(File, delimiter = ',')
        next(Line_reader) #header pass
        for row in Line_reader:
            arkit_step.append(float(row[1]))
            arkit_psnr.append(float(row[2]))
    return arkit_step,arkit_psnr


arkit_step,arkit_psnr = np.array(load_csv('arkit_hallway_PSNR.csv'))
iphone_step,iphone_psnr = np.array(load_csv('iphone_hallway_PSNR.csv'))

# arkit_step,arkit_psnr = load_csv('arkit_lounge_PSNR.csv')
# iphone_step,iphone_psnr = load_csv('iphone_lounge_PSNR.csv')

interval = 20
# cubic_interpol_model_arkit = interp1d(arkit_step,arkit_psnr, kind="cubic")
# cubic_interpol_model_iphone = interp1d(iphone_step,iphone_psnr, kind="cubic")
# arkit_step  = np.linspace(arkit_step.min(), arkit_step.max(), interval)
# arkit_psnr = cubic_interpol_model_arkit(arkit_step)
# iphone_step  = np.linspace(iphone_step.min(), iphone_step.max(), interval)
# iphone_psnr = cubic_interpol_model_iphone(iphone_step)

cubic_interpol_model_arkit = make_interp_spline(arkit_step,arkit_psnr,k=3)
cubic_interpol_model_iphone = make_interp_spline(iphone_step,iphone_psnr)
arkit_step  = np.linspace(arkit_step.min(), arkit_step.max(), interval)
arkit_psnr = cubic_interpol_model_arkit(arkit_step)
iphone_step  = np.linspace(iphone_step.min(), iphone_step.max(), interval)
iphone_psnr = cubic_interpol_model_iphone(iphone_step)
print(arkit_step)

plt.plot(arkit_step,arkit_psnr, label='Ours', c= (00,139/255.0,139/255.0))
plt.plot(iphone_step,iphone_psnr, label='BARF', c=(255/255.0,99/255.0,71/255.0))
plt.title("PSNR",fontsize=20)
plt.xlabel('Number of Steps',fontsize=20)
plt.ylabel('PSNR',fontsize=20)

# plt.xaxis.set_major_locator(ticker.MultipleLocator(5000))
# plt.xaxis.set_minor_locator(ticker.MultipleLocator(1000))
# plt.xlim(0,200000)
plt.ylim(min(iphone_psnr)-3,max(arkit_psnr)+3)
plt.legend(loc='upper left')
# plt.grid(True)
plt.show()
fname = "{}.png".format("PSNR")
plt.savefig(fname, dpi=75)
# clean up
# plt.close('all')




