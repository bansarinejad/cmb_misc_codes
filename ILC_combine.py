'''
Do this in the termianl first before running the rest of the code
[bansarinejad@spartan-login2 spt_maps]$ export PATH="/home/bansarinejad/anaconda3/bin:$PATH"
[bansarinejad@spartan-login2 spt_maps]$ /data/gpfs/projects/punim1720/spt3g-software/spt3g_software-master/build/env-shell.sh
'''

#folders
#camb / foregrounds file paths
spt3g_folder = '/data/gpfs/projects/punim1720/spt3g-software/'
datafolder = '%s/simulations/python/data/' %(spt3g_folder)
cambfolder = 'camb/planck18_TTEEEE_lowl_lowE_lensing_highacc' #'camb/planck18_TTEEEE_lowl_lowE_lensing'
import sys
builddir = '%sbuild' %(spt3g_folder)
sys.path.append(builddir)
ilcdir = '%s/ilc/python' %(spt3g_folder)
sys.path.append(ilcdir)
#sys_path_folder='../python/'
#sys.path.append(sys_path_folder)

from spt3g import core
from spt3g.simulations import foregrounds as fg
from spt3g.mapspectra import basicmaputils as utils
from spt3g.beams import beam_analysis as beam_stuff
import ilc
from spt3g.ilc import ilc
import healpy as hp
import flatsky, misc
import numpy as np, sys, os, scipy as sc, warnings#, healpy as H
import matplotlib.pyplot as plt
import matplotlib.cbook
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

#%pylab notebook
#%matplotlib inline
from pylab import *

#'''apodization - Behzad's code''''
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

#map resolution
full_sky = 0
#perform_analytic_ilc = True
if full_sky:
    nside = 2048
    mapparams = None
    lmax = 13000#5000
    res = None
    shape = None
else:
    nside = None
    dx = 0.5
    boxsize_x_am = 200. #boxsize in arcmins
    boxsize_y_am = 200. #boxsize in arcmins
    nx, ny = int(boxsize_x_am/dx), int(boxsize_y_am/dx)
    mapparams = [nx, ny, dx]
    x1,x2 = -nx/2. * dx, nx/2. * dx
    y1,y2 = -ny/2. * dx, ny/2. * dx
    lmax = 6000
    res = np.radians(dx/60.)
    shape = (ny, nx)

lstart = 1 #monopole
lmin = 20 #based on SPT-SZ foregounds
el = np.arange(lstart,lmax)
verbose = 0
tcmb = 2.73

#beam and noise levels for the three frequencies
freqs = [95, 150, 220]
bands=['95GHz', '150GHz', '220GHz']
experiments=['spt3g', 'spt3g', 'spt3g']

beam_noise_dic = {95: [1.57, 5.5], 150: [1.17, 4.5], 220: [1.04, 16.0]} #SPT-3G field 
opbeam = beam_noise_dic[150][0]
#get beams
bl_dic = {}
beam_norm = 1.

opbeam_deg = opbeam/60.
bl_eff = beam_stuff.Bl_gauss(el, opbeam_deg, beam_norm)
reso_rad = res / core.G3Units.rad
bl_eff_twod = utils.interp_cl_2d(bl_eff, reso_rad, shape, ell=el, real=False)
bl_dic['effective'] = bl_eff_twod

for freq in freqs:
    beamval, noiseval = beam_noise_dic[freq]
    beamval_deg = beamval/60. #arcmins to degrees
    bl = beam_stuff.Bl_gauss(el, beamval_deg, beam_norm)
    bl_twod = utils.interp_cl_2d(bl, res, shape, ell=el, real=False)
    bl_dic[freq] = bl_twod

#for plotting
colordic = {'95GHz': 'navy', '150GHz': 'darkgreen', '220GHz': 'red'}
cmap = cm.jet
#read map cutouts
t_temp_95=np.load('./data_cutouts/90GHz_cutouts_T_nw.npy', allow_pickle= True)
t_temp_150=np.load('./data_cutouts/150GHz_cutouts_T_nw.npy', allow_pickle= True)
t_temp_220=np.load('./data_cutouts/220GHz_cutouts_T_nw.npy', allow_pickle= True)
t_95=np.asarray(t_temp_95)*big_field_mask_del#*1000
t_150=np.asarray(t_temp_150)*big_field_mask_del#*1000
t_220=np.asarray(t_temp_220)*big_field_mask_del#*1000

map_dict={'95GHz': t_95[0],'150GHz': t_150[0], '220GHz': t_220[0]} #[0] element as test

real_nl=True
if real_nl:
  nl_95_temp = hp.read_cl("./nls/signflip_000_bundle_000_090ghz.fits")
  nl_150_temp = hp.read_cl("./nls/signflip_000_bundle_000_150ghz.fits")
  nl_220_temp = hp.read_cl("./nls/signflip_000_bundle_000_220ghz.fits")
  nl_oned_dic={}
  nl_oned_dic['TT']={}
  nl_oned_dic['TT']['95GHz'] = nl_95_temp[0] / (core.G3Units.uK*core.G3Units.arcmin)**2
  nl_oned_dic['TT']['150GHz'] = nl_150_temp[0] / (core.G3Units.uK*core.G3Units.arcmin)**2
  nl_oned_dic['TT']['220GHz'] = nl_220_temp[0] / (core.G3Units.uK*core.G3Units.arcmin)**2
  tf_twod = None
else:
  #atmospheric noise stuff for SPT
  elknee_t_90, alpha_knee_90 = 1200., 3.
  elknee_t_150, alpha_knee_150 = 2200., 4.
  elknee_t_220, alpha_knee_220 = 2300., 4.
  elknee_dic = {95: [elknee_t_90, alpha_knee_90], 150: [elknee_t_150, alpha_knee_150], 220: [elknee_t_220, alpha_knee_220]}
  
  #get beam deconvolved noise nls 
  nl_oned_dic= {}
  ax = subplot(111, yscale = 'log')
  which_spec = 'TT'
  nl_oned_dic[which_spec] = {}
  for freq in freqs:
      beamval, noiseval = beam_noise_dic[freq]
      nl_oned = misc.get_nl(noiseval, el, beamval, elknee_t=elknee_dic[freq][0], alpha_knee=elknee_dic[freq][1])#, mapparams = None)
      nl_oned *= (core.G3Units.uK ** 2.0)
      plt.plot(el, nl_oned/core.G3Units.uK ** 2.0, label = r'%s: %s arcmins, %.2f uK-arcmin' %(freq, beamval, noiseval), color = colordic['%sGHz' %(freq)])
      #nl_twod = misc.get_nl(noiseval, el, beamval, elknee_t=elknee_dic[freq][0], alpha_knee=elknee_dic[freq][1], make_2d = 1, mapparams = mapparams)
      #nl_twod *= (core.G3Units.uK ** 2.0)
      '''
      nl_twod[el<=lmin] = 0.
      nl_twod[nl_twod == 0.] = np.min(nl_twod[nl_twod!=0.])/1e3
      '''
      band = '%sGHz' %(freq)
      #nl_twod_dic[which_spec][band] = nl_twod
      nl_oned_dic[which_spec][band] = nl_oned
  plt.legend(loc = 1)
  plt.xlabel(r'Multipole $\ell$'); ylabel(r'N$_{\ell}$ [$\mu$K$^{2}$]')
  plt.show()
  
  #supply some simple transfer function
  lx_min = 20
  ell_max = 13000
  lx, ly = flatsky.get_lxly(mapparams)
  ell = np.sqrt(lx**2 + ly**2)
  tf_twod = np.ones(lx.shape) #None
  if (1):
      tf_twod = np.ones(lx.shape)
  if (0):
      tf_twod[abs(lx)<=lx_min] = 0.
      tf_twod[ell>ell_max] = 0.
  if (0):
      tf_twod = tf_twod * 0. + 1.
  if tf_twod is not None:
      imshow(np.fft.fftshift(tf_twod), cmap = cmap, extent = [lx.min(), lx.max(), ly.min(), ly.max()]); colorbar(); 
      axhline(lw = 0.5); axvline(lw=0.5); title(r'2D filtering'); show()
  
  #convolve nl_twod_dic with tf_twod
  #if tf_twod is not None:
  #    for freq in nl_twod_dic[which_spec]:
  #        nl_twod_dic[which_spec][freq] *= tf_twod
  #else:
  #    nl_twod_dic = nl_oned_dic
  nl_twod_dic = nl_oned_dic

        
ilc_op_dic = {}
#for keyname in comp_dic_for_ilc:
curr_weights_arr, curr_cl_residual_arr, curr_ilc_map = ilc.perform_ilc(component='CMB', bands=bands, experiments=experiments, lmax=lmax, lmin = lmin, cl_dict = None, nl_dict = nl_oned_dic, null_components = None, ignore_fg = ['CMB', 'kSZ'], tf_2d = tf_twod, shape = shape, res = res, map_dict = map_dict, bl_dict = bl_dic, nside = None, full_sky = 0, apod_mask=big_field_mask_del)
#curr_weights_arr, curr_cl_residual_arr, curr_ilc_map = ilc.perform_ilc(final_comp, bands, explist, lmax, lmin = lmin, cl_dict = None, nl_dict = nl_twod_dic, null_components = null_comp, ignore_fg = ignore_fg, tf_2d = tf_twod, shape = shape, res = res, map_dict = map_dic, bl_dict = bl_dic, nside = nside, full_sky = full_sky)
ilc_op_dic['cmbmv'] = [curr_weights_arr, curr_cl_residual_arr, curr_ilc_map]

#simply store weights, residuals, and maps under different variable names for plotting.
weights_arr, cl_residual_arr, ilc_map = ilc_op_dic['cmbmv']

#plot now
figure(figsize=(10., 5.))
subplots_adjust(wspace = 0.25, hspace = 0.25)
vmax = 300
vmin = -vmax
plt.subplot(121);imshow(t_150[0]/ core.G3Units.uK, cmap = cmap, extent = [x1,x2,y1,y2], vmin = vmin, vmax = vmax); colorbar(); title(r'150 GHz map', fontsize = 10);
plt.subplot(122);imshow(ilc_map/ core.G3Units.uK, cmap = cmap, extent = [x1,x2,y1,y2], vmin = vmin, vmax = vmax); colorbar(); title(r'ILC CMB map', fontsize = 10);
plt.show()
plt.subplot(131);imshow(weights_arr[0,:,:]/ core.G3Units.uK, cmap = cmap, extent = [x1,x2,y1,y2]); colorbar(); title(r'95 weights', fontsize = 10); #, vmin = vmin, vmax = vmax
plt.subplot(132);imshow(weights_arr[1,:,:]/ core.G3Units.uK, cmap = cmap, extent = [x1,x2,y1,y2]); colorbar(); title(r'150 weights', fontsize = 10); #, vmin = vmin, vmax = vmax
plt.subplot(133);imshow(weights_arr[2,:,:]/ core.G3Units.uK, cmap = cmap, extent = [x1,x2,y1,y2]); colorbar(); title(r'220 weights', fontsize = 10); #, vmin = vmin, vmax = vmax
plt.show()

#plot weights now
lx, ly = flatsky.get_lxly( [nx, ny, dx] )
plt.subplot(111)
weightsarr_for_sum = []
for frqcntr, freq in enumerate( freqs ):
    band = '%sGHz' %(freq)
    curr_weights = weights_arr[frqcntr] 
    tit = 'MV CMB ILC'
    acap = ilc.get_freq_response(bands, experiments, component='CMB').T
    rad_prf = flatsky.radial_profile(curr_weights, (lx,ly), bin_size = 100, minbin = 100, maxbin = 10000, to_arcmins = 0)
    el_, curr_weights = rad_prf[:,0], rad_prf[:,1]
    plt.plot(el_, curr_weights, color = colordic[band], label = r'%s' %(freq))

    acap = np.asarray(acap)[0]
    weightsarr_for_sum.append( curr_weights * acap[frqcntr] )

weightsarr_for_sum = np.asarray(weightsarr_for_sum)

plt.plot(el_, np.sum(weightsarr_for_sum, axis = 0), 'k--', label = r'Sum')    
plt.axhline(lw=0.3)
plt.legend(loc = 4, fontsize = 7, ncol = 2)
plt.ylim(-3., 3.)
plt.xlim(300, lmax-100)
plt.title(tit, fontsize = 10)
plt.xlabel(r'Multipole $\ell$'); ylabel(r'W$_{\ell}$')
plt.show()
