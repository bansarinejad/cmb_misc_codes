#!/usr/bin/env python
########################
########################
#load desired modules
import sys
#casys.path.append('/home/rptd37/spt3g/RHEL_7_x86_64/')
import os
from numpy import average, split
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import Angle
from spt3g import core
from spt3g import maps
from spt3g import mapmaker
from spt3g import mapspectra, coordinateutils
import healpy as hp
import matplotlib.pyplot as plt
from focus_tools import *
import numpy as np, sys, os, scipy as sc, argparse
import scipy
from scipy import stats
sys_path_folder='../python/'
sys.path.append(sys_path_folder)

import time
start = time.time()

import flatsky, tools, lensing, foregrounds, misc

from tqdm import tqdm
from matplotlib import pyplot
from pylab import *
cmap = cm.RdYlBu_r

import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
print('\n')
########################

from spt3g import maps
import copy

########################
parser = argparse.ArgumentParser(description='')
parser.add_argument('-paramfile', dest='paramfile', action='store', help='paramfile', type=str, default='params_data.ini')

args = parser.parse_args()
args_keys = args.__dict__
for kargs in args_keys:
    param_value = args_keys[kargs]

    if isinstance(param_value, str):
        cmd = '%s = "%s"' %(kargs, param_value)
    else:
        cmd = '%s = %s' %(kargs, param_value)
    exec(cmd)

########################
print('\tread/get necessary params')
param_dict = misc.get_param_dict(paramfile)

cls_file = '%s/%s' %(param_dict['data_folder'], param_dict['cls_file'])
el, cl = tools.get_cmb_cls(cls_file, pol =False)
els=np.arange(300,13000,1)
dl_plot_camb=(cl[0][300:13000]*els*(els+1.))/(2.*np.pi)

dg_alm = hp.read_alm('/data/gpfs/projects/punim1720/spt_maps/sim_8192_v7_42/foregrounds/poisson_dg_150ghz_alms_0007.fits', hdu=[1,2,3])
dg_map1=hp.sphtfunc.alm2cl(dg_alm[0])

rg_alm = hp.read_alm('/data/gpfs/projects/punim1720/spt_maps/sim_8192_v7_42/foregrounds/poisson_rg_150ghz_alms_0007.fits', hdu=[1,2,3])
rg_map1=hp.sphtfunc.alm2cl(rg_alm[0])

gauss_alm= hp.read_alm('/data/gpfs/projects/punim1720/spt_maps/sim_8192_v7_42/foregrounds/combined_gaussian_150ghz_alms_0007.fits', hdu=[1,2,3])
guass_map1=hp.sphtfunc.alm2cl(gauss_alm[0])

ps_alm= hp.read_alm('/data/gpfs/projects/punim1720/spt_maps/sim_8192_v7_42/foregrounds/detected_point_sources_150ghz_alms_0000.fits', hdu=[1,2,3])
ps_map1=hp.sphtfunc.alm2cl(ps_alm[0])

cmb_alm= hp.read_alm('/data/gpfs/projects/punim1720/spt_maps/sim_8192_v7_42/cmb/cmb_alms_0007.fits', hdu=[1,2,3])
cmb_map1=hp.sphtfunc.alm2cl(cmb_alm[0])

dl_plot_dg=((dg_map1[300:13000]/1e-6)*els*(els+1.))/(2.*np.pi)
bin_means, bin_edges, binnumber = stats.binned_statistic(els, dl_plot_dg, statistic='mean', bins=300)
bin_stds, bin_edges, binnumber = stats.binned_statistic(els, dl_plot_dg, statistic='std', bins=300)
bin_count, bin_edges, binnumber = stats.binned_statistic(els, dl_plot_dg, statistic='count', bins=300)
errors=bin_stds/np.sqrt(bin_count)
bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2
pyplot.errorbar(bin_centers,bin_means,yerr=errors,fmt='bo',label='dg')
print('el:', bin_centers[64])
print('dg:', bin_means[64])

dl_plot_rg=((rg_map1[300:13000]/1e-6)*els*(els+1.))/(2.*np.pi)
bin_means, bin_edges, binnumber = stats.binned_statistic(els, dl_plot_rg, statistic='mean', bins=300)
bin_stds, bin_edges, binnumber = stats.binned_statistic(els, dl_plot_rg, statistic='std', bins=300)
bin_count, bin_edges, binnumber = stats.binned_statistic(els, dl_plot_rg, statistic='count', bins=300)
errors=bin_stds/np.sqrt(bin_count)
bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2
pyplot.errorbar(bin_centers,bin_means,yerr=errors,fmt='ro',label='rg')
print('rg:', bin_means[64])

dl_plot_ps=((ps_map1[300:13000]/1e-6)*els*(els+1.))/(2.*np.pi)
bin_means, bin_edges, binnumber = stats.binned_statistic(els, dl_plot_ps, statistic='mean', bins=300)
bin_stds, bin_edges, binnumber = stats.binned_statistic(els, dl_plot_ps, statistic='std', bins=300)
bin_count, bin_edges, binnumber = stats.binned_statistic(els, dl_plot_ps, statistic='count', bins=300)
errors=bin_stds/np.sqrt(bin_count)
bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2
pyplot.errorbar(bin_centers,bin_means,yerr=errors,fmt='yo',label='detected point sources >6mJy')
print('ps:', bin_means[64])

dl_plot_gauss=((guass_map1[300:13000]/1e-6)*els*(els+1.))/(2.*np.pi)
bin_means, bin_edges, binnumber = stats.binned_statistic(els, dl_plot_gauss, statistic='mean', bins=300)
bin_stds, bin_edges, binnumber = stats.binned_statistic(els, dl_plot_gauss, statistic='std', bins=300)
bin_count, bin_edges, binnumber = stats.binned_statistic(els, dl_plot_gauss, statistic='count', bins=300)
errors=bin_stds/np.sqrt(bin_count)
bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2
pyplot.errorbar(bin_centers,bin_means,yerr=errors,fmt='go', label='combined gaussian')
print('gauss:', bin_means[64])

dl_plot_cmb=((cmb_map1[300:13000]/1e-6)*els*(els+1.))/(2.*np.pi)
bin_means, bin_edges, binnumber = stats.binned_statistic(els, dl_plot_cmb, statistic='mean', bins=300)
bin_stds, bin_edges, binnumber = stats.binned_statistic(els, dl_plot_cmb, statistic='std', bins=300)
bin_count, bin_edges, binnumber = stats.binned_statistic(els, dl_plot_cmb, statistic='count', bins=300)
errors=bin_stds/np.sqrt(bin_count)
bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2
pyplot.errorbar(bin_centers,bin_means,yerr=errors,fmt='co',label='cmb')
print('cmb:', bin_means[64])

total=np.sum((dl_plot_cmb,dl_plot_ps,dl_plot_gauss,dl_plot_rg,dl_plot_dg),axis=0)
total=np.asarray(total)
bin_means, bin_edges, binnumber = stats.binned_statistic(els, total, statistic='mean', bins=300)
bin_stds, bin_edges, binnumber = stats.binned_statistic(els, total, statistic='std', bins=300)
bin_count, bin_edges, binnumber = stats.binned_statistic(els, total, statistic='count', bins=300)
errors=bin_stds/np.sqrt(bin_count)
bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2
pyplot.errorbar(bin_centers,bin_means,yerr=errors,fmt='ko',label='total')
print('total:', bin_means[64])

pyplot.plot(els,dl_plot_camb,label='CAMB')

pyplot.xscale('log')
pyplot.yscale('log')
pyplot.legend()
pyplot.show() 

