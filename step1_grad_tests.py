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
from skimage import restoration
sys_path_folder='../python/'
sys.path.append(sys_path_folder)

import flatsky, tools, tools_data, tools_data_gradtest, tools_data_backup, lensing, foregrounds, misc

from tqdm import tqdm
import objgraph
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
t_temp =  np.load('./data_cutouts/ILC_cutouts_U_nw.npy', allow_pickle= True)
t=np.asarray(t_temp)*1000
#t_temp2=np.asarray(t_temp)*1000
#print(t_temp2.shape, t_temp2.nbytes) 
#t = np.rot90(t_temp2, k=1, axes=(1, 2))
#print(t.shape, t.nbytes)
mask=np.load('./mask_0p4medwt_6mJy150ghzv2_cutouts.npy', allow_pickle= True)

nx = 400#int(box_width / t.res)          ### get number of pixels on the side of the cutout

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
mapparams = [nx, nx, dx]
verbose = 0
pol = True #param_dict['pol']
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

if (0):
    #This is to stack without rotation and see if there's T->P dipole leakage that could contaminate the results.
    cl_signal_arr=cl #cl[0] if using camb cl
    cl_noise_arr=[nl_dic['T']]
    sims=t#sim_dic[sim_type]['sims']
    print(sims.shape)
    stacked_arr = np.mean(sims, axis=0)
    wiener_filter=flatsky.wiener_filter(mapparams, cl_signal=cl_signal_arr[0], cl_noise=cl_noise_arr[0])
    test= np.fft.ifft2( np.fft.fft2(stacked_arr) * wiener_filter ).real

    #imshow(data_stack_dic[0,10:30,10:30],interpolation='gaussian',cmap=cmap);colorbar();show()
    #imshow(stack[0][10:30,10:30],cmap=cmap);colorbar();show()
    #imshow(random_stack,cmap=cmap);colorbar();show()
    #imshow(data_stack_dic[0],cmap=cmap);colorbar();show()
    print(np.mean(stacked_arr[199:201,199:201]))
    imshow(stacked_arr[190:210,190:210],cmap=cmap);colorbar();show()
    imshow(stacked_arr[190:210,190:210],interpolation='gaussian',cmap=cmap);colorbar();show()
    imshow(test[190:210,190:210],cmap=cmap, vmin=-0.125, vmax=0.125);colorbar();show()

    #imshow(stacked_arr);colorbar();show()
    #imshow(stacked_arr[190:210,190:210]);colorbar();show()
    sys.exit()

sim_arr=[]
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
    #mask[i][mask[i]<1]=nan
    #sim_map=t[i]
    sim_map=t[i]##*mask[i]
    mask_two=np.copy(mask[i])
    mask_two=np.where(mask_two<0.1,1.0,0.0)
    #print(sim_map.shape)
    #print(mask_two.shape)
    sim_map=restoration.inpaint_biharmonic(sim_map, mask_two)
    
    ##for tqu in range(tqulen):c#mean subtraction for T(/Q/U)
    sim_map -= np.mean(sim_map)
    sim_arr.append( sim_map )
sim_dic[sim_type]['sims']=np.asarray( sim_arr )

if (0):
    cl_signal_arr=cl #cl[0] if using camb cl
    cl_noise_arr=[nl_dic['T']]
    sims=sim_dic[sim_type]['sims']
    print(sims.shape)
    stacked_arr = np.mean(sims, axis=0)
    wiener_filter=flatsky.wiener_filter(mapparams, cl_signal=cl_signal_arr[0], cl_noise=cl_noise_arr[0])
    test= np.fft.ifft2( np.fft.fft2(stacked_arr) * wiener_filter ).real

    #imshow(data_stack_dic[0,10:30,10:30],interpolation='gaussian',cmap=cmap);colorbar();show()
    #imshow(stack[0][10:30,10:30],cmap=cmap);colorbar();show()
    #imshow(random_stack,cmap=cmap);colorbar();show()
    #imshow(data_stack_dic[0],cmap=cmap);colorbar();show()
    imshow(stacked_arr[190:210,190:210],cmap=cmap);colorbar();show()
    print(np.mean(stacked_arr))
    print(np.mean(stacked_arr[190:210,190:210]))
    imshow(stacked_arr[190:210,190:210],interpolation='gaussian',cmap=cmap);colorbar();show()
    imshow(test[190:210,190:210],cmap=cmap, vmin=-0.125, vmax=0.125);colorbar();show()

    #imshow(stacked_arr);colorbar();show()
    #imshow(stacked_arr[190:210,190:210]);colorbar();show()
    sys.exit()
#mapparams= [int(nx/N), int(nx/N), dx*N]
########################

#'''apodization'''
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
objgraph.show_most_common_types()
objgraph.show_growth()
########################
#get gradient information for all cluster cutouts
print('\tget gradient information for all cluster cutouts')
for sim_type in sim_dic:
    sim_dic[sim_type]['cutouts_rotated'] = {}
    sim_dic[sim_type]['grad_mag'] = {}
    sim_dic[sim_type]['grad_orien'] = {}
    sim_dic[sim_type]['grad_orien_full'] = {}
###    for simcntr in range( start, end ):ty
###        print('\t\tmock dataset %s of %s' %(simcntr+1, end-start))
    if do_apod:
      sim_arr=sim_dic[sim_type]['sims']*(big_field_mask_del)#/np.mean(big_field_mask_del))
    else:
      sim_arr=sim_dic[sim_type]['sims']
    nclustersorrandoms=len(t)
    if apply_wiener_filter:
        if pol:
            cl_signal_arr=[cl[0], cl[1]/2., cl[1]/2.]
            cl_noise_arr=[nl_dic['T'], nl_dic['P'], nl_dic['P']]
        else:
            cl_signal_arr=cl #cl[0] if using camb cl
            cl_noise_arr=[nl_dic['T']]

    #get median gradient direction and magnitude for all cluster cutouts + rotate them along median gradient direction.
    #grad_mag_arr, grad_orien_arr, cutouts_rotated_arr = tools_data_backup.get_rotated_tqu_cutouts(sim_arr, sim_arr, nclustersorrandoms, tqulen, mapparams, cutout_size_am, apply_wiener_filter=True, cl_signal = cl_signal_arr[0], cl_noise = cl_noise_arr[0], lpf_gradient_filter = lpf_gradient_filter, cutout_size_am_for_grad = cutout_size_am_for_grad)
    grad_mag_arr, grad_orien_arr, cutouts_rotated_arr, grad_orien_full_arr = tools_data_gradtest.get_rotated_tqu_cutouts(sim_arr, sim_arr, nclustersorrandoms, tqulen, mapparams, cutout_size_am, apply_wiener_filter=True, cl_signal = cl_signal_arr[0], cl_noise = cl_noise_arr[0], lpf_gradient_filter = lpf_gradient_filter, cutout_size_am_for_grad = cutout_size_am_for_grad)
    #block below added as there are some cutouts/grad_mags that are 10^16 etc... need to investigate, this is a temporary fix
    cutouts_rotated_arr_new=[]
    grad_mag_arr_new=[]
    grad_orien_arr_new=[]
    grad_orien_full_arr_new=[]
    for i in tqdm(range(len(cutouts_rotated_arr))):
        if (np.sum(cutouts_rotated_arr[i])<1 and grad_mag_arr[i]<10):
            cutouts_rotated_arr_new.append(cutouts_rotated_arr[i])
            grad_mag_arr_new.append(grad_mag_arr[i])
            grad_orien_arr_new.append(grad_orien_arr[i])
            grad_orien_full_arr_new.append(grad_orien_full_arr[i])

    sim_dic[sim_type]['cutouts_rotated']=np.asarray(cutouts_rotated_arr_new)
    sim_dic[sim_type]['grad_mag']=np.asarray(grad_mag_arr_new)
    sim_dic[sim_type]['grad_orien']=np.asarray(grad_orien_arr_new)
    sim_dic[sim_type]['grad_orien_full'] =np.asarray(grad_orien_full_arr_new)
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
     sims=sim_dic[sim_type]['sims']

     stack = tools_data.stack_rotated_tqu_cutouts(cutouts_rotated_arr, weights_for_cutouts = grad_mag_arr)
     #stack = tools_data.stack_rotated_tqu_cutouts(sims, weights_for_cutouts = grad_mag_arr)
     sim_dic[sim_type]['stack']=stack
        
########################

########################
#save results
sim_dic[sim_type].pop('sims')
#if clusters_or_randoms == 'randoms':
#    sim_dic[sim_type].pop('cutouts_rotated')
#    sim_dic[sim_type].pop('grad_mag')
sim_dic['param_dict']=param_dict

np.save('./stacks/clusters_m200cut_U_ILC_healpix_nw_apod_zeros_masked_inpaint_v3_fullorient_circmean.npy', sim_dic)

sys.exit()

