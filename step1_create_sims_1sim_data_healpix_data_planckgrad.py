#!/usr/bin/env python
########################
########################
#load desired modules
import sys
import os
from numpy import average, split
from astropy.io import fits
from spt3g import core
from spt3g import maps
from spt3g import mapmaker
from spt3g import mapspectra
import matplotlib.pyplot as plt
from focus_tools import *
import numpy as np, sys, os, scipy as sc, argparse
import pickle as pkl
from skimage import restoration
sys_path_folder='../python/'
sys.path.append(sys_path_folder)

import flatsky, tools, tools_data, lensing, foregrounds, misc

from tqdm import tqdm

from pylab import *
cmap = cm.RdYlBu_r

import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
print('\n')
########################

# set width of the cutout in arcmin
box_width = 200.0*core.G3Units.arcmin

# input g3 file
#t_temp=[None]
tp_temp=np.load('./planck/Planck_cutouts_T_nw_masked_WFv3_COMMANDER.npy', allow_pickle= True)
tp=np.asarray(tp_temp)#*1000000
t_temp=np.load('./data_cutouts/ILC_cutouts_T_nw.npy', allow_pickle= True)
t=np.asarray(t_temp)*1000
mask=np.load('./data_cutouts/mask_0p4medwt_6mJy150ghzv2_cutouts.npy', allow_pickle= True)

nx = 400#int(box_width / t.res)          ### get number of pixels on the side of the cutout
nxp = 400#int(box_width / t.res)          ### get number of pixels on the side of the cutout for planck map

########################
parser = argparse.ArgumentParser(description='')
parser.add_argument('-clusters_or_randoms', dest='clusters_or_randoms', action='store', help='clusters_or_randoms', type=str, default='clusters')
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

########################
print('\tread/get necessary params')
param_dict = misc.get_param_dict(paramfile)

data_folder = param_dict['data_folder']
results_folder = param_dict['results_folder']

#params or supply a params file
dx = param_dict['dx'] #pixel resolution in arcmins
dxp= 0.5 #5.0
mapparams = [nx, nx, dx]
mapparamsp = [nx, nx, dx]
verbose = 0
pol = False #True #param_dict['pol']
debug = param_dict['debug']

###beam and noise levels
noiseval = param_dict['noiseval'] #uK-arcmin
if pol:
    noiseval = [noiseval, noiseval * np.sqrt(2.), noiseval * np.sqrt(2.)]

#CMB power spectrum
cls_file = '%s/%s' %(param_dict['data_folder'], param_dict['cls_file'])
el, cl = tools.get_cmb_cls(cls_file, pol = pol)
nl_dic = tools.get_nl_dic(noiseval, el, pol = pol)

#if not pol:
#    tqulen = 1
#else:
#    tqulen = 3
#tqu_tit_arr = ['T', 'Q', 'U']
tqulen = 1

#cosmology
h=param_dict['h']
omega_m=param_dict['omega_m']
omega_lambda=param_dict['omega_lambda']
z_lss=param_dict['z_lss']
T_cmb=param_dict['T_cmb']

#cutouts specs 
cutout_size_am = param_dict['cutout_size_am'] #arcmins

#for estimating cmb gradient
apply_wiener_filter = param_dict['apply_wiener_filter']
lpf_gradient_filter = param_dict['lpf_gradient_filter']
cutout_size_am_for_grad = param_dict['cutout_size_am_for_grad'] #arcminutes
########################

########################
#generating sims
sim_dic={}
if clusters_or_randoms == 'clusters': #cluster lensed sims
###    do_lensing=True
    total_clusters=len(t)
    nclustersorrandoms=total_clusters  ##tochange
    sim_type='clusters'
elif clusters_or_randoms == 'randoms':
###    do_lensing=False
    total_randoms=len(t)
    nclustersorrandoms=total_randoms   ##tochange
    sim_type='randoms'
sim_dic[sim_type]={}
sim_dic[sim_type]['sims'] = {}
sim_dic[sim_type]['simsp'] = {}

sim_arr=[]
sim_arrp=[]
#N=2                                  ### setting downsampling factor
for i in tqdm(range(nclustersorrandoms)):

    # convert cluster RA+DEC to pixel coordinates and extract patched centered on the pixel
    #x, y = maps.FlatSkyMap.angle_to_xy(t, 200, 200)
    #t_cutout = t[i]#.extract_patch(200, 200, nx, nx)
    #print(t_cutout.shape) #testing
    
    #width_temp = t_cutout.shape[0]
    #height_temp= t_cutout.shape[1]
    #sim_map=average(split(average(split(t_cutout, width_temp // N, axis=1), axis=-1), height_temp // N, axis=1), axis=-1)
    t[i][t[i]<-1000]=0.0
    t[i][t[i]>1000]=0.0
    #sim_map=t[i]
    sim_map=t[i]##*mask[i]
    sim_mapp=tp[i] ##planck cutouts
    mask_two=np.copy(mask[i])
    mask_two=np.where(mask_two<0.1,1.0,0.0)
    sim_map=restoration.inpaint_biharmonic(sim_map, mask_two)
    
    ##for tqu in range(tqulen):#mean subtraction for T(/Q/U)
    sim_map -= np.mean(sim_map)
    sim_mapp -= np.mean(sim_mapp)
    sim_arr.append( sim_map )
    sim_arrp.append( sim_mapp )
sim_dic[sim_type]['sims']=np.asarray( sim_arr )
sim_dic[sim_type]['simsp']=np.asarray( sim_arrp )
#mapparams= [int(nx/N), int(nx/N), dx*N]
########################

#'''apodization''''
###ker=np.hanning(10)
###ker2d=np.asarray( np.sqrt(np.outer(ker,ker)) )
do_apod=1
if do_apod:
  def gaussian(x, mu, sig):
  	return (np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)
  
  x=np.arange(0,30,1)
  y=np.zeros(30)
  for i in range(len(x)):
  	y[i]=gaussian(i,0,15)
  
  pad=y/max(y)
  pad_right=np.tile(pad,(120,1))
  pad_left=np.tile(np.flip(pad,axis=0),(120,1))
  pad_down=np.tile(pad,(180,1))  #60 added in the lines above (2x30) so need to be 180 for the first dimension now.
  pad_up=np.tile(np.flip(pad,axis=0),(180,1))
  
  small_mask=np.ones((120,120))  # the bit with ones. 
  
  small_mask_padded=np.append(small_mask, pad_right,axis=1)  #gaussian padding bits
  small_mask_padded=np.append(pad_left, small_mask_padded,axis=1)
  small_mask_padded=np.append(small_mask_padded,np.transpose(pad_down),axis=0)
  small_mask_padded=np.append(np.transpose(pad_up),small_mask_padded,axis=0)
  
  big_field_mask=np.pad(small_mask_padded,(240,240),mode='constant',constant_values=0.)
  
  #big_field_mask=np.pad(small_mask,(120,120),mode='constant',constant_values=0.)
  
  big_field_mask_del=np.delete(big_field_mask,np.s_[:130],1)  # for 400x400 set these to 130 and 400. for 200x200 set these to 230 and 200.
  big_field_mask_del=np.delete(big_field_mask_del,np.s_[400:],1)
  big_field_mask_del=np.delete(big_field_mask_del,np.s_[:130],0)
  big_field_mask_del=np.delete(big_field_mask_del,np.s_[400:],0)

apply_wiener_filter=False #set to false for planck

########################
#get gradient information for all cluster cutouts
print('\tget gradient information for all cluster cutouts')
for sim_type in sim_dic:
    sim_dic[sim_type]['cutouts_rotated'] = {}
    sim_dic[sim_type]['grad_mag'] = {}
###    for simcntr in range( start, end ):ty
###        print('\t\tmock dataset %s of %s' %(simcntr+1, end-start))
    if do_apod:
      sim_arr=sim_dic[sim_type]['sims']*(big_field_mask_del)#/np.mean(big_field_mask_del))
      sim_arrp=sim_dic[sim_type]['simsp']
    else:
      sim_arr=sim_dic[sim_type]['sims']
      sim_arrp=sim_dic[sim_type]['simsp']
    if (0):
      #cl_simarr,alm_simarr = hp.anafast(sim_arr, lmax=13000, alm=True)
      #ell = np.arange(len(cl_simarr))
      #cl_sigmoid_hpf=1/(1+np.exp(0.05*(-ell+1000)))
      hpf_filter=flatsky.get_lpf_hpf(mapparams, 1000., filter_type=1)
      sim_arr=np.fft.ifft2( np.fft.fft2( sim_arr ) * hpf_filter ).real
    nclustersorrandoms=len(t)
    if apply_wiener_filter:
        if pol:
            cl_signal_arr=[cl[0], cl[1]/2., cl[1]/2.]
            cl_noise_arr=[nl_dic['T'], nl_dic['P'], nl_dic['P']]
        else:
            cl_signal_arr=cl #cl[0] if using camb cl
            cl_noise_arr=[nl_dic['T']]

    #get median gradient direction and magnitude for all cluster cutouts + rotate them along median gradient direction.
    #grad_mag_arr, grad_orien_arr, cutouts_rotated_arr = tools_data.get_rotated_tqu_cutouts(sim_arr, sim_arrp, nclustersorrandoms, tqulen, mapparams, mapparamsp, cutout_size_am, apply_wiener_filter=True, cl_signal = cl_signal_arr[0], cl_noise = cl_noise_arr[0], lpf_gradient_filter = lpf_gradient_filter, cutout_size_am_for_grad = cutout_size_am_for_grad)
    grad_mag_arr, grad_orien_arr, cutouts_rotated_arr = tools_data.get_rotated_tqu_cutouts(sim_arr, sim_arrp, nclustersorrandoms, tqulen, mapparams, mapparamsp, cutout_size_am, apply_wiener_filter=False, cl_signal = None, cl_noise = None, lpf_gradient_filter = None, cutout_size_am_for_grad = cutout_size_am_for_grad)
    
    #block below added as there are some cutouts/grad_mags that are 10^16 etc... need to investigate, this is a temporary fix
    cutouts_rotated_arr_new=[]
    grad_mag_arr_new=[]
    grad_orien_arr_new=[]
    for i in tqdm(range(len(cutouts_rotated_arr))):
        if (np.sum(cutouts_rotated_arr[i])<10 and grad_mag_arr[i]<100):
            cutouts_rotated_arr_new.append(cutouts_rotated_arr[i])
            grad_mag_arr_new.append(grad_mag_arr[i])
            grad_orien_arr_new.append(grad_orien_arr[i])

    sim_dic[sim_type]['cutouts_rotated']=np.asarray(cutouts_rotated_arr_new)
    sim_dic[sim_type]['grad_mag']=np.asarray(grad_mag_arr_new)  
    sim_dic[sim_type]['grad_orien']=np.asarray(grad_orien_arr_new)

########################

########################
#stack rotated cutouts + apply gradient magnitude weights
print('\tstack rotated cutouts + apply gradient magnitude weights')
sim_dic[sim_type]['stack'] = {}
for sim_type in sim_dic:
     ###for simcntr in range( start, end ):
     ###print('\t\tmock dataset %s of %s' %(simcntr+1, end-start))
     cutouts_rotated_arr=sim_dic[sim_type]['cutouts_rotated']
     grad_mag_arr=sim_dic[sim_type]['grad_mag']
     
     stack = tools_data.stack_rotated_tqu_cutouts(cutouts_rotated_arr, weights_for_cutouts = grad_mag_arr)
     sim_dic[sim_type]['stack']=stack
        
########################

########################
#save results
sim_dic[sim_type].pop('sims')
if clusters_or_randoms == 'randoms':
    sim_dic[sim_type].pop('cutouts_rotated')
    sim_dic[sim_type].pop('grad_mag')
sim_dic['param_dict']=param_dict

#np.save('./clusters_m200cut_T_150GHz_healpix_nw_apod_zeros_masked_inpaint_v4_planckgrad.npy', sim_dic)
with open('./stacks/clusters_m200cut_T_ILC_healpix_nw_apod_zeros_masked_inpaint_planckgrad_WFv3_COMMANDER_v2.pkl','wb') as f:
  pkl.dump(sim_dic, f)
sys.exit()
