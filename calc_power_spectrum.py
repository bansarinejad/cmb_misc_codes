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

# input g3 file
#data_dir = '/sptlocal/user/rptd37/SPT_maps/spt_maps/'
#example_file = os.path.join(data_dir,'no_signflip_bundle_000.g3')
#mask_file= np.load('/sptlocal/user/rptd37/SPT_maps/spt_maps/masks/mask_0p4medwt_6mJy150ghzv2.npy', allow_pickle= True)

# core.G3File returns an iterator, which iterates over the frames in the file
#g3file= core.G3File(example_file)

# Access the first frame:
#frame1 = g3file.next()
#maps.RemoveWeights(frame1, zero_nans = True)
#map1=frame1*mask_file
#print(frame1)

#hmap_T = hp.fitsfunc.read_map('/data/gpfs/projects/punim1720/spt_maps/sim_8192_v7_4/total/total_150ghz_map_3g_0000.fits',field=0)
hmap_T = hp.fitsfunc.read_map('/data/gpfs/projects/punim1720/spt_maps/no_signflip_bundle_000_150GHz_nw.fits',field=0)
hmap_T = np.float32(hmap_T)
hmap_T[np.where(np.isnan(hmap_T))] = 0.0
print(hmap_T.shape)

#map_stub_T = coordinateutils.FlatSkyMap(x_len=5000,y_len=2500, alpha_center=0.0*core.G3Units.deg,delta_center=-57.5*core.G3Units.deg, res=1.0*core.G3Units.arcmin,proj=maps.MapProjection(0))
reso_arcmin = 0.5
ra0, dec0 = 0, -57.5
proj = 0
ny, nx = 3000, 4500
map_stub = maps.FlatSkyMap(
    nx, ny, 
    reso_arcmin*core.G3Units.arcmin, 
    weighted = False, 
    proj = maps.MapProjection(proj),
    alpha_center = ra0*core.G3Units.degrees, 
    delta_center = dec0*core.G3Units.degrees,
    coord_ref=maps.MapCoordReference.Equatorial, 
    units = core.G3TimestreamUnits.Tcmb, 
    pol_type=maps.MapPolType.T)

t_map = maps.maputils.healpix_to_flatsky(hmap_T, map_stub=map_stub, interp=True)

frame = core.G3Frame(core.G3FrameType.Map)
frame['T']=t_map

WeightsMap = maps.G3SkyMapWeights(t_map, polarized = False)
smTw = np.ones(t_map.shape)
np.asarray(WeightsMap.TT)[:,:] = smTw
frame['Wunpol'] = WeightsMap

#t_map = maps.maputils.healpix_to_flatsky(hmap_T,  proj = maps.MapProjection(proj),coord_ref=maps.MapCoordReference.Equatorial, units = core.G3TimestreamUnits.Tcmb, pol_type=maps.MapPolType.T, interp=True)

#in_frame = maps.fitsio.load_skymap_fits('/data/gpfs/projects/punim1720/spt_maps/sim_8192_v7_4/total/total_150ghz_map_3g_0000.fits')#snakemake.input[0])
#frame = core.G3Frame(core.G3FrameType.Map)
#for k in in_frame.keys():
#    if not (in_frame[k]).any():
#        m = in_frame.pop(k)
    #np.asarray(m)[masked_pixels] = 0
#        m.ringsparse = True
    # maybe check npix_allocated and shift_ra here to make sure sparsification worked. might have to set m.shift_ra = True manually
#        frame[k] = m
#WeightsMap = maps.G3SkyMapWeights(t_map, polarized = False)
#sptpol_weights = data.initArgs()['weight']
#smTw = np.ones(t_map.shape)

#np.asarray(WeightsMap.TT)[:,:] = smTw

#mapkey = 'cmb_map'
#opfname = '/data/gpfs/projects/punim1720/spt_maps/sim_8192_v7_4/total/total_150ghz_map_3g_0000.g3'
#pipe = core.G3Pipeline()
#pipe.Add(core.G3InfiniteSource,type= core.G3FrameType.Map,n=0)
#pipe.Add(maps.InjectMaps, map_id=mapkey, maps_in=[t_map, WeightsMap])
#pipe.Add(core.Dump)
#pipe.Add(core.G3Writer, filename = opfname)
#pipe.Run(profile=True)        
#core.G3Writer(frame)
##T_map=frame
#T_map=maps.G3SkyMap(t_map, polarized = False)
#writer = core.G3Writer(snakemake.output[0])
#writer(frame)

#map1, map2, map3 = frame1_josh, frame1_sam, frame1_eduardo
##maps.RemoveWeights(map1, zero_nans = True) #remove weights from Joshua's map if not already done

#copy joshua's weights into eduardo's map
##map2_mod = copy.deepcopy(map1)
##np.asarray(map2_mod['T'])[:] = np.asarray(map2['T']) #map is copied now

cls_file = '%s/%s' %(param_dict['data_folder'], param_dict['cls_file'])
el, cl = tools.get_cmb_cls(cls_file, pol =False)

####apodization###

gen_apod_mask=False#True
if gen_apod_mask:
  def gaussian(x, mu, sig):
  	return (np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)
  
  x=np.arange(0,120,1)
  y=np.zeros(120)
  for i in range(len(x)):
  	y[i]=gaussian(i,0,60)
  
  pad=y/max(y)
  pad_right=np.tile(pad,(4000,1))
  pad_left=np.tile(np.flip(pad,axis=0),(4000,1))
  pad_down=np.tile(pad,(6240,1))
  pad_up=np.tile(np.flip(pad,axis=0),(6240,1))
  
  small_mask=np.ones((4000,6000))
  
  small_mask_padded=np.append(small_mask, pad_right,axis=1)
  small_mask_padded=np.append(pad_left, small_mask_padded,axis=1)
  small_mask_padded=np.append(small_mask_padded,np.transpose(pad_down),axis=0)
  small_mask_padded=np.append(np.transpose(pad_up),small_mask_padded,axis=0)
  
  big_field_mask=np.pad(small_mask_padded,(12000,12000),mode='constant',constant_values=0.)
  
  #big_field_mask=np.pad(small_mask,(120,120),mode='constant',constant_values=0.)
  
  big_field_mask_del=np.delete(big_field_mask,np.s_[:6120],1)
  big_field_mask_del=np.delete(big_field_mask_del,np.s_[4000:],1)
  big_field_mask_del=np.delete(big_field_mask_del,np.s_[:8120],0)
  big_field_mask_del=np.delete(big_field_mask_del,np.s_[4000:],0)
  
  #masked_field=(big_field*big_field_mask_del)/np.mean(big_field_mask_del)
  ##test=(map1['T']*big_field_mask_del)
  test=(t_map*big_field_mask_del)
  
  map1_test = copy.deepcopy(frame)
  np.asarray(map1_test['T'])[:] = np.asarray(test) #map is copied now

##################

print('\t calculate power spectra')
#cl_map1=mapspectra.map_analysis.calculate_powerspectra(frame, delta_l=1, lmin=300, lmax=13000)#, apod_mask = 'from_weight') 
cl_map1=mapspectra.map_analysis.calculate_powerspectra(frame, delta_l=1, lmin=300, lmax=13000, apod_mask = 'from_weight') 
#cl_map2=mapspectra.map_analysis.calculate_powerspectra(map2, delta_l=1, lmin=500, lmax=5000, apod_mask = 'from_weight')
#cl_test=mapspectra.map_analysis.calculate_powerspectra(map1_test, delta_l=1, lmin=2, lmax=5000)#, apod_mask = 'from_weight') #Joshua's map with mask
#cl_map2=mapspectra.map_analysis.calculate_powerspectra(map2, delta_l=1, lmin=2, lmax=5000) #Eduardo's map w/o mask (wrong normalisation)
#cl_map2_mod=mapspectra.map_analysis.calculate_powerspectra(map2_mod, delta_l=1, lmin=2, lmax=5000, apod_mask = 'from_weight') #Eduardo's map with mask

els=np.arange(300,13000,1)

#cl_plot=cl_test['TT']/(1e-6*np.mean(big_field_mask_del**2))
dl_plot_camb=(cl[0][300:13000]*els*(els+1.))/(2.*np.pi)
dl_plot=((cl_map1['TT']/1e-6)*els*(els+1.))/(2.*np.pi)
#dl_plot_SG=((cl_map2['TT']/1e-6)*els*(els+1.))/(2.*np.pi)

bin_means, bin_edges, binnumber = stats.binned_statistic(els, dl_plot, statistic='mean', bins=500)
bin_stds, bin_edges, binnumber = stats.binned_statistic(els, dl_plot, statistic='std', bins=500)
bin_count, bin_edges, binnumber = stats.binned_statistic(els, dl_plot, statistic='count', bins=500)
errors=bin_stds/np.sqrt(bin_count)

##pyplot.plot(els,dl_plot,'bo',label='Data')
bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2
pyplot.errorbar(bin_centers,bin_means,yerr=errors,fmt='bo')#,label='Data')
pyplot.plot(els,dl_plot_camb,label='CAMB')

pyplot.xscale('log')
pyplot.yscale('log')
pyplot.legend()
pyplot.show() 

bin_means, bin_edges, binnumber = stats.binned_statistic(els, dl_plot/dl_plot_camb, statistic='mean', bins=500)
bin_stds, bin_edges, binnumber = stats.binned_statistic(els, dl_plot/dl_plot_camb, statistic='std', bins=500)
bin_count, bin_edges, binnumber = stats.binned_statistic(els, dl_plot/dl_plot_camb, statistic='count', bins=500)
errors=bin_stds/np.sqrt(bin_count)

pyplot.errorbar(bin_centers,bin_means,yerr=errors,fmt='bo')#,label='Data')

#pyplot.plot(els,dl_plot/dl_plot_camb,'bo',label='data/camb')
pyplot.axhline(y = 1.0, color = 'r', linestyle = '-')
pyplot.yscale('log')
pyplot.legend()
pyplot.show() 
