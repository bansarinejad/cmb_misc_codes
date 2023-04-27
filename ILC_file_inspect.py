#!/usr/bin/env python
########################

########################
#load desired modules
import numpy as np, sys, os, scipy as sc, argparse
sys_path_folder='../python/'
sys.path.append(sys_path_folder)

import flatsky, tools, lensing, foregrounds, misc
from astropy.io import fits
from tqdm import tqdm
import ilc
from spt3g.ilc import ilc

from pylab import *
#for plotting
colordic = {'95GHz': 'navy', '150GHz': 'darkgreen', '220GHz': 'red'}
cmap = cm.RdYlBu_r

import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
print('\n')
########################

########################
if (0):
	ilc_residual = np.load('./debug/T_95_150_220_winterfield_fullsky_cl_residual.npy', allow_pickle = True).squeeze()#.item()
	print(ilc_residual)
	print(ilc_residual.shape)
	print( np.mean(ilc_residual))
	print( np.std(ilc_residual))
	print( np.min(ilc_residual))
	print( np.max(ilc_residual))

	el_res=np.arange(len(ilc_residual))
	plot(el_res,ilc_residual);show()

plot_weights=True
if plot_weights:
	weights_arr = np.load('./debug/T_95_150_220_winterfield_fullsky_weights.npy', allow_pickle = True)
	print(weights_arr.shape)
	print(weights_arr[0].shape)
	#plot weights now
	nx=40
	ny=40
	dx=0.5
	lx, ly = flatsky.get_lxly( [nx, ny, dx] )
	plt.subplot(111)
	weightsarr_for_sum = []
	freqs = [95, 150, 220]
	bands=['95GHz', '150GHz', '220GHz']
	experiments=['spt3g', 'spt3g', 'spt3g']

	for frqcntr, freq in enumerate( freqs ):
	    band = '%sGHz' %(freq)
	    print(band)
	    curr_weights = weights_arr[frqcntr]
	    el_=np.arange(len(weights_arr[1]))
	    print(el_.shape)
	    print(frqcntr)
	    print(curr_weights.shape) 
	    tit = 'MV CMB ILC'
	    acap = ilc.get_freq_response(bands, experiments, component='CMB').T
	    #rad_prf = flatsky.radial_profile(curr_weights, xy = None, bin_size = 100, minbin = 100, maxbin = 10000, to_arcmins = 0) # Behzad: original minbin = 100, maxbin = 10000
	    #el_, curr_weights = rad_prf[:,0], rad_prf[:,1]
	    plt.plot(el_, curr_weights, color = colordic[band], label = r'%s' %(freq))

	    acap = np.asarray(acap)[0]
	    weightsarr_for_sum.append( curr_weights * 1.0)#acap[frqcntr] )

	weightsarr_for_sum = np.asarray(weightsarr_for_sum)
	lmax=13000
	plt.plot(el_, np.sum(weightsarr_for_sum, axis = 0), 'k--', label = r'Sum')    
	plt.axhline(lw=0.3)
	plt.legend(loc = 4, fontsize = 7, ncol = 2)
	plt.ylim(-0.5, 1.5)
	plt.xlim(0, lmax-100)
	plt.title(tit, fontsize = 10)
	plt.xlabel(r'Multipole $\ell$'); ylabel(r'W$_{\ell}$')
	plt.show()
