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
import spt3g
from spt3g import core, maps, mapspectra
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
from draw_field_map import plot_flat_sky_maps, plot_healpix_maps
from spt3g.maps.maputils import healpix_to_flatsky
from spt3g.std_processing import CreateFieldMapStub
from scipy import ndimage
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

#%pylab notebook
#%matplotlib inline
from matplotlib import pyplot
from pylab import *

#Apodization mask
def get_apod_mask(weight_map, weight_threshold = 0.1, hanning_rad = 60.):
    weight_map = weight_map / np.max(weight_map)
    apod_mask = np.ones( weight_map.shape )
    #apod_mask[weight_map<weight_threshold] = 0.
    indices = np.where(weight_map < weight_threshold)
    apod_mask[indices] = 0

    hanning=np.hanning(hanning_rad)
    hanning=np.sqrt(np.outer(hanning,hanning))

    apod_mask=ndimage.convolve(apod_mask, hanning)
    apod_mask=apod_mask/np.max(apod_mask)

    #imshow(apod_mask); colorbar(); show(); #sys.exit()
    
    return apod_mask

#map resolution
full_sky = 0
#perform_analytic_ilc = True
nside = None
dx = 0.5
boxsize_x_am = 4800. #boxsize in arcmins
boxsize_y_am = 2100. #boxsize in arcmins
nx, ny = int(boxsize_x_am/dx), int(boxsize_y_am/dx)
mapparams = [ny, nx, dx]
x1,x2 = -nx/2. * dx, nx/2. * dx
y1,y2 = -ny/2. * dx, ny/2. * dx
lmax = 13000
res = np.radians(dx/60.)
reso_rad = res / core.G3Units.rad
shape = (ny, nx)

lstart = 1 #monopole
lmin = 20 #based on SPT-SZ foregounds
el = np.arange(lstart,lmax)
verbose = 0
tcmb = 2.73

freqs = [95, 150, 220]
bands=['95GHz', '150GHz', '220GHz']
experiments=['spt3g', 'spt3g', 'spt3g']

calc_beam=False
if calc_beam:
    #beam and noise levels for the three frequencies

    beam_noise_dic = {95: [1.57, 5.5], 150: [1.57, 4.5], 220: [1.57, 16.0]} #SPT-3G field ; org: {95: [1.57, 5.5], 150: [1.17, 4.5], 220: [1.04, 16.0]}
    opbeam = beam_noise_dic[150][0]
    #get beams
    bl_dic = {}
    beam_norm = 1.

    opbeam_deg = opbeam/60.
    bl_eff = beam_stuff.Bl_gauss(el, opbeam_deg, beam_norm)
    reso_rad = res / core.G3Units.rad
    bl_eff_twod = utils.interp_cl_2d(bl_eff, reso_rad, shape, ell=el, real=False)
    bl_dic['effective'] = bl_eff_twod
    print(bl_eff_twod.shape)

    for freq in freqs:
        beamval, noiseval = beam_noise_dic[freq]
        beamval_deg = beamval/60. #arcmins to degrees
        bl = beam_stuff.Bl_gauss(el, beamval_deg, beam_norm)
        bl_twod = utils.interp_cl_2d(bl, res, shape, ell=el, real=False)
        bl_dic[freq] = bl_twod
else:
    el_bl= np.genfromtxt('./bls/compiled_2020_beams.txt', usecols=0)
    bl_95 = np.genfromtxt('./bls/compiled_2020_beams.txt', usecols=1)
    bl_150 = np.genfromtxt('./bls/compiled_2020_beams.txt', usecols=2)
    bl_220 = np.genfromtxt('./bls/compiled_2020_beams.txt', usecols=3)    
    bl_eff = bl_150
    bl_dic = {}
    bl_dic['effective'] = utils.interp_cl_2d(bl_eff, reso_rad, shape, ell=el_bl, real=False)
    bl_dic[95] = utils.interp_cl_2d(bl_95, reso_rad, shape, ell=el_bl, real=False)
    bl_dic[150] = utils.interp_cl_2d(bl_150, reso_rad, shape, ell=el_bl, real=False)
    bl_dic[220] = utils.interp_cl_2d(bl_220, reso_rad, shape, ell=el_bl, real=False)

#for plotting
colordic = {'95GHz': 'navy', '150GHz': 'darkgreen', '220GHz': 'red'}
cmap = cm.jet

#convert maps from healpix to flatsky, mask point sources and save files to disk 
read_save_maps=False
if read_save_maps:
    #read full maps
    t_95 = np.float32(hp.fitsfunc.read_map('./winter/no_signflip_bundle_000_90GHz_nw.fits',field=0))
    t_150 = np.float32(hp.fitsfunc.read_map('./winter/no_signflip_bundle_000_150GHz_nw.fits',field=0))
    t_220 = np.float32(hp.fitsfunc.read_map('./winter/no_signflip_bundle_000_220GHz_nw.fits',field=0))
    t_95[np.where(np.isnan(t_95))] = 0.
    t_150[np.where(np.isnan(t_150))] = 0.
    t_220[np.where(np.isnan(t_220))] = 0.

    q_95 = np.float32(hp.fitsfunc.read_map('./winter/no_signflip_bundle_000_90GHz_nw.fits',field=1))
    q_150 = np.float32(hp.fitsfunc.read_map('./winter/no_signflip_bundle_000_150GHz_nw.fits',field=1))
    q_220 = np.float32(hp.fitsfunc.read_map('./winter/no_signflip_bundle_000_220GHz_nw.fits',field=1))
    q_95[np.where(np.isnan(q_95))] = 0.
    q_150[np.where(np.isnan(q_150))] = 0.
    q_220[np.where(np.isnan(q_220))] = 0.

    u_95 = np.float32(hp.fitsfunc.read_map('./winter/no_signflip_bundle_000_90GHz_nw.fits',field=2))
    u_150 = np.float32(hp.fitsfunc.read_map('./winter/no_signflip_bundle_000_150GHz_nw.fits',field=2))
    u_220 = np.float32(hp.fitsfunc.read_map('./winter/no_signflip_bundle_000_220GHz_nw.fits',field=2))
    u_95[np.where(np.isnan(u_95))] = 0.
    u_150[np.where(np.isnan(u_150))] = 0.
    u_220[np.where(np.isnan(u_220))] = 0.

    weights_mask = np.float32(hp.fitsfunc.read_map('./winter/no_signflip_bundle_000_150GHz_weights.fits',field=4))
    weights_mask[np.where(np.isnan(weights_mask))] = 0.

    ps_mask=np.load('./masks/mask_0p4medwt_6mJy150ghzv2.npy')
    hmap_ps_mask =np.float32(spt3g.maps.HealpixSkyMap(ps_mask))
    hmap_ps_mask[np.where(np.isnan(hmap_ps_mask))] = 0. 

    frame_flat_T_95 = healpix_to_flatsky(t_95,map_stub=CreateFieldMapStub(res=0.5 * core.G3Units.arcmin, width=80 * core.G3Units.deg, height=35 * core.G3Units.deg, pol_type=maps.MapPolType.T, proj = spt3g.maps.MapProjection(0)), interp=True, rebin=1,)
    frame_flat_T_150 = healpix_to_flatsky(t_150,map_stub=CreateFieldMapStub(res=0.5 * core.G3Units.arcmin, width=80 * core.G3Units.deg, height=35 * core.G3Units.deg, pol_type=maps.MapPolType.T, proj = spt3g.maps.MapProjection(0)), interp=True, rebin=1,)
    frame_flat_T_220 = healpix_to_flatsky(t_220,map_stub=CreateFieldMapStub(res=0.5 * core.G3Units.arcmin, width=80 * core.G3Units.deg, height=35 * core.G3Units.deg, pol_type=maps.MapPolType.T, proj = spt3g.maps.MapProjection(0)), interp=True, rebin=1,)
    frame_flat_Q_95 = healpix_to_flatsky(q_95,map_stub=CreateFieldMapStub(res=0.5 * core.G3Units.arcmin, width=80 * core.G3Units.deg, height=35 * core.G3Units.deg, pol_type=maps.MapPolType.Q, proj = spt3g.maps.MapProjection(0)), interp=True, rebin=1,)
    frame_flat_Q_150 = healpix_to_flatsky(q_150,map_stub=CreateFieldMapStub(res=0.5 * core.G3Units.arcmin, width=80 * core.G3Units.deg, height=35 * core.G3Units.deg, pol_type=maps.MapPolType.Q, proj = spt3g.maps.MapProjection(0)), interp=True, rebin=1,)
    frame_flat_Q_220 = healpix_to_flatsky(q_220,map_stub=CreateFieldMapStub(res=0.5 * core.G3Units.arcmin, width=80 * core.G3Units.deg, height=35 * core.G3Units.deg, pol_type=maps.MapPolType.Q, proj = spt3g.maps.MapProjection(0)), interp=True, rebin=1,)
    frame_flat_U_95 = healpix_to_flatsky(u_95,map_stub=CreateFieldMapStub(res=0.5 * core.G3Units.arcmin, width=80 * core.G3Units.deg, height=35 * core.G3Units.deg, pol_type=maps.MapPolType.U, proj = spt3g.maps.MapProjection(0)), interp=True, rebin=1,)
    frame_flat_U_150 = healpix_to_flatsky(u_150,map_stub=CreateFieldMapStub(res=0.5 * core.G3Units.arcmin, width=80 * core.G3Units.deg, height=35 * core.G3Units.deg, pol_type=maps.MapPolType.U, proj = spt3g.maps.MapProjection(0)), interp=True, rebin=1,)
    frame_flat_U_220 = healpix_to_flatsky(u_220,map_stub=CreateFieldMapStub(res=0.5 * core.G3Units.arcmin, width=80 * core.G3Units.deg, height=35 * core.G3Units.deg, pol_type=maps.MapPolType.U, proj = spt3g.maps.MapProjection(0)), interp=True, rebin=1,)

    frame_flat_hmap_ps_mask = healpix_to_flatsky(hmap_ps_mask,map_stub=CreateFieldMapStub(res=0.5 * core.G3Units.arcmin, width=80 * core.G3Units.deg, height=35 * core.G3Units.deg, pol_type=maps.MapPolType.T, proj = spt3g.maps.MapProjection(0)), interp=True, rebin=1,)

    frame_flat_T_95=frame_flat_T_95*frame_flat_hmap_ps_mask
    frame_flat_T_95=np.where(frame_flat_hmap_ps_mask<0.1,0.0006,frame_flat_T_95)
    frame_flat_T_150=frame_flat_T_150*frame_flat_hmap_ps_mask
    frame_flat_T_150=np.where(frame_flat_hmap_ps_mask<0.1,7.89E-5,frame_flat_T_150)
    frame_flat_T_220=frame_flat_T_220*frame_flat_hmap_ps_mask
    frame_flat_T_220=np.where(frame_flat_hmap_ps_mask<0.1,-0.00112,frame_flat_T_220)

    frame_flat_Q_95=frame_flat_Q_95*frame_flat_hmap_ps_mask
    frame_flat_Q_95=np.where(frame_flat_hmap_ps_mask<0.1,-2.16E-5,frame_flat_Q_95)
    frame_flat_Q_150=frame_flat_Q_150*frame_flat_hmap_ps_mask
    frame_flat_Q_150=np.where(frame_flat_hmap_ps_mask<0.1,4.42E-5,frame_flat_Q_150)
    frame_flat_Q_220=frame_flat_Q_220*frame_flat_hmap_ps_mask
    frame_flat_Q_220=np.where(frame_flat_hmap_ps_mask<0.1,0.00011,frame_flat_Q_220)

    frame_flat_U_95=frame_flat_U_95*frame_flat_hmap_ps_mask
    frame_flat_U_95=np.where(frame_flat_hmap_ps_mask<0.1,3.55E-5,frame_flat_U_95)
    frame_flat_U_150=frame_flat_U_150*frame_flat_hmap_ps_mask
    frame_flat_U_150=np.where(frame_flat_hmap_ps_mask<0.1,-9.01E-7,frame_flat_U_150)
    frame_flat_U_220=frame_flat_U_220*frame_flat_hmap_ps_mask
    frame_flat_U_220=np.where(frame_flat_hmap_ps_mask<0.1,0.00011,frame_flat_U_220)
     
    frame_flat_apod_mask_temp = healpix_to_flatsky(weights_mask,map_stub=CreateFieldMapStub(res=0.5 * core.G3Units.arcmin, width=80 * core.G3Units.deg, height=35 * core.G3Units.deg, pol_type=maps.MapPolType.T, proj = spt3g.maps.MapProjection(0)), interp=True, rebin=1,)
    frame_flat_apod_mask=get_apod_mask(frame_flat_apod_mask_temp, weight_threshold = 0.1, hanning_rad = 60.)

    np.save('./flatsky_maps/T_95GHz_winterfield_flat.npy', frame_flat_T_95)
    np.save('./flatsky_maps/T_150GHz_winterfield_flat.npy', frame_flat_T_150)
    np.save('./flatsky_maps/T_220GHz_winterfield_flat.npy', frame_flat_T_220)
    np.save('./flatsky_maps/Q_95GHz_winterfield_flat.npy', frame_flat_Q_95)
    np.save('./flatsky_maps/Q_150GHz_winterfield_flat.npy', frame_flat_Q_150)
    np.save('./flatsky_maps/Q_220GHz_winterfield_flat.npy', frame_flat_Q_220)
    np.save('./flatsky_maps/U_95GHz_winterfield_flat.npy', frame_flat_U_95)
    np.save('./flatsky_maps/U_150GHz_winterfield_flat.npy', frame_flat_U_150)
    np.save('./flatsky_maps/U_220GHz_winterfield_flat.npy', frame_flat_U_220)
    np.save('./flatsky_maps/apod_mask.npy',frame_flat_apod_mask)

read_flatsky_files=True
if read_flatsky_files:
    frame_flat_T_95=np.load('./flatsky_maps/T_95GHz_winterfield_flat.npy', allow_pickle = True)
    frame_flat_T_150=np.load('./flatsky_maps/T_150GHz_winterfield_flat.npy', allow_pickle = True)
    frame_flat_T_220=np.load('./flatsky_maps/T_220GHz_winterfield_flat.npy', allow_pickle = True)
    frame_flat_Q_95=np.load('./flatsky_maps/Q_95GHz_winterfield_flat.npy', allow_pickle = True)
    frame_flat_Q_150=np.load('./flatsky_maps/Q_150GHz_winterfield_flat.npy', allow_pickle = True)
    frame_flat_Q_220=np.load('./flatsky_maps/Q_220GHz_winterfield_flat.npy', allow_pickle = True)
    frame_flat_U_95=np.load('./flatsky_maps/U_95GHz_winterfield_flat.npy', allow_pickle = True)
    frame_flat_U_150=np.load('./flatsky_maps/U_150GHz_winterfield_flat.npy', allow_pickle = True)
    frame_flat_U_220=np.load('./flatsky_maps/U_220GHz_winterfield_flat.npy', allow_pickle = True)
    frame_flat_apod_mask=np.load('./flatsky_maps/apod_mask.npy', allow_pickle = True)
    map_dict_T={'95GHz': frame_flat_T_95,'150GHz': frame_flat_T_150, '220GHz': frame_flat_T_220}
    map_dict_Q={'95GHz': frame_flat_Q_95,'150GHz': frame_flat_Q_150, '220GHz': frame_flat_Q_220}
    map_dict_U={'95GHz': frame_flat_U_95,'150GHz': frame_flat_U_150, '220GHz': frame_flat_U_220}

tf_twod = None
do_ICL=False
if do_ICL:
    nl_95_temp = hp.read_cl("./nls/signflip_000_bundle_000_090ghz.fits")
    nl_150_temp = hp.read_cl("./nls/signflip_000_bundle_000_150ghz.fits")
    nl_220_temp = hp.read_cl("./nls/signflip_000_bundle_000_220ghz.fits")
    nl_oned_dic_TT={}
    nl_oned_dic_TT['TT']={}
    nl_oned_dic_TT['TT']['95GHz'] = nl_95_temp[0] / (core.G3Units.uK*core.G3Units.arcmin)**2
    nl_oned_dic_TT['TT']['150GHz'] = nl_150_temp[0] / (core.G3Units.uK*core.G3Units.arcmin)**2
    nl_oned_dic_TT['TT']['220GHz'] = nl_220_temp[0] / (core.G3Units.uK*core.G3Units.arcmin)**2
    nl_twod_dic_TT = nl_oned_dic_TT

    nl_oned_dic_EE={}
    nl_oned_dic_EE['EE']={}
    nl_oned_dic_EE['EE']['95GHz'] = nl_95_temp[0] / (core.G3Units.uK*core.G3Units.arcmin)**2
    nl_oned_dic_EE['EE']['150GHz'] = nl_150_temp[0] / (core.G3Units.uK*core.G3Units.arcmin)**2
    nl_oned_dic_EE['EE']['220GHz'] = nl_220_temp[0] / (core.G3Units.uK*core.G3Units.arcmin)**2
    nl_twod_dic_EE = nl_oned_dic_EE

    nl_oned_dic_BB={}
    nl_oned_dic_BB['BB']={}
    nl_oned_dic_BB['BB']['95GHz'] = nl_95_temp[0] / (core.G3Units.uK*core.G3Units.arcmin)**2
    nl_oned_dic_BB['BB']['150GHz'] = nl_150_temp[0] / (core.G3Units.uK*core.G3Units.arcmin)**2
    nl_oned_dic_BB['BB']['220GHz'] = nl_220_temp[0] / (core.G3Units.uK*core.G3Units.arcmin)**2
    nl_twod_dic_BB = nl_oned_dic_BB

    ilc_op_dic_T = {}
    ilc_op_dic_Q = {}
    ilc_op_dic_U = {}
    
    #for keyname in comp_dic_for_ilc:
    ell=list(range(lmin, lmax+1))
    curr_weights_arr_T, curr_cl_residual_arr_T, curr_ilc_map_T = ilc.perform_ilc(component='CMB', bands=bands, experiments=experiments, lmax=lmax, lmin = lmin, ell=ell, cl_dict = None, nl_dict = nl_oned_dic_TT, null_components = None, ignore_fg = ['CMB', 'kSZ'], tf_2d = tf_twod, shape = shape, res = res, map_dict = map_dict_T, bl_dict = bl_dic, nside = nside , full_sky = full_sky, apod_mask=frame_flat_apod_mask)
    ilc_op_dic_T['cmbmv'] = [curr_weights_arr_T, curr_cl_residual_arr_T, curr_ilc_map_T]
    weights_arr_T, cl_residual_arr_T, ilc_map_T = ilc_op_dic_T['cmbmv']
    #np.save('./T_95_150_220_winterfield_flat_ILC.npy', ilc_map_T)

    curr_weights_arr_Q, curr_cl_residual_arr_Q, curr_ilc_map_Q = ilc.perform_ilc(component='CMB', bands=bands, experiments=experiments, lmax=lmax, lmin = lmin, ell=ell, cl_dict = None, nl_dict = nl_oned_dic_QQ, null_components = None, ignore_fg = ['CMB', 'kSZ'], tf_2d = tf_twod, shape = shape, res = res, map_dict = map_dict_Q, bl_dict = bl_dic, nside = nside , full_sky = full_sky, apod_mask=frame_flat_apod_mask)
    ilc_op_dic_Q['cmbmv'] = [curr_weights_arr_Q, curr_cl_residual_arr_Q, curr_ilc_map_Q]
    weights_arr_Q, cl_residual_arr_Q, ilc_map_Q = ilc_op_dic_Q['cmbmv']
    #np.save('./Q_95_150_220_winterfield_flat_ILC.npy', ilc_map_Q)

    curr_weights_arr_U, curr_cl_residual_arr_U, curr_ilc_map_U = ilc.perform_ilc(component='CMB', bands=bands, experiments=experiments, lmax=lmax, lmin = lmin, ell=ell, cl_dict = None, nl_dict = nl_oned_dic_UU, null_components = None, ignore_fg = ['CMB', 'kSZ'], tf_2d = tf_twod, shape = shape, res = res, map_dict = map_dict_U, bl_dict = bl_dic, nside = nside , full_sky = full_sky, apod_mask=frame_flat_apod_mask)
    ilc_op_dic_U['cmbmv'] = [curr_weights_arr_U, curr_cl_residual_arr_U, curr_ilc_map_U]
    weights_arr_U, cl_residual_arr_U, ilc_map_U = ilc_op_dic_U['cmbmv']
    #np.save('./U_95_150_220_winterfield_flat_ILC.npy', ilc_map_U)

    # map_arr=np.asarray((frame_flat_T_95*1000*frame_flat_apod_mask, frame_flat_T_150*1000*frame_flat_apod_mask, frame_flat_T_220*1000*frame_flat_apod_mask)) #*frame_flat_apod_mask
    # weighted_maparr = []
    # for (m, w) in zip( map_arr, weights_arr):
    #     curr_weighted_map = np.fft.ifft2( np.fft.fft2( m ) * w ).real
    #     weighted_maparr.append( curr_weighted_map )

    # ilc_map_new = np.sum(weighted_maparr, axis = (0))
    # np.save('./T_95_150_220_winterfield_flat_ILC_sri.npy', ilc_map_new)
    #print(ilc_map.shape, map_arr.shape, weight_arr.shape)
else:
    ilc_map_T=np.load('./T_95_150_220_winterfield_flat_ILC.npy',allow_pickle=True)
    #ilc_map_new=np.load('./T_95_150_220_winterfield_flat_ILC_sri.npy',allow_pickle=True)

plot_maps=True
if plot_maps:
    #plot now
    figure(figsize=(10., 5.))
    subplots_adjust(wspace = 0.25, hspace = 0.25)
    vmax = 150
    vmin = -vmax
    plt.subplot(121);imshow(frame_flat_T_150/ core.G3Units.uK, cmap = cmap, extent = [x1,x2,y1,y2], vmin = vmin, vmax = vmax); colorbar(); title(r'150 GHz map', fontsize = 10);
    plt.subplot(122);imshow(ilc_map_T/ core.G3Units.uK, cmap = cmap, extent = [x1,x2,y1,y2], vmin = vmin, vmax = vmax); colorbar(); title(r'ILC CMB map', fontsize = 10); #, vmin = vmin, vmax = vmax
    plt.show()
    #plt.subplot(132);imshow(frame_flat_apod_mask, cmap = cmap, extent = [x1,x2,y1,y2], vmin = 0, vmax = 1); colorbar(); title(r'Apod mask', fontsize = 10);
    #plt.subplot(133);imshow(ilc_map_new, cmap = cmap, extent = [x1,x2,y1,y2], vmin = vmin, vmax = vmax); colorbar(); title(r'ILC CMB map new', fontsize = 10); #, vmin = vmin, vmax = vmax
    # plt.subplot(131);imshow(weights_arr[0,:,:]/ core.G3Units.uK, cmap = cmap, extent = [x1,x2,y1,y2], vmin = 0, vmax = 600); colorbar(); title(r'95 weights', fontsize = 10); #, vmin = vmin, vmax = vmax
    # plt.subplot(132);imshow(weights_arr[1,:,:]/ core.G3Units.uK, cmap = cmap, extent = [x1,x2,y1,y2], vmin = 0, vmax = 600); colorbar(); title(r'150 weights', fontsize = 10); #, vmin = vmin, vmax = vmax
    # plt.subplot(133);imshow(weights_arr[2,:,:]/ core.G3Units.uK, cmap = cmap, extent = [x1,x2,y1,y2], vmin = 0, vmax = 600); colorbar(); title(r'220 weights', fontsize = 10); #, vmin = vmin, vmax = vmax
    # plt.show()

print('p-1', flush=True)
get_powerspec_org=True
if get_powerspec_org:
    print('p0', flush=True)
    #cl of input and output
    if tf_twod is not None:
        filter_2d_plus_beam = tf_twod * bl_dic['effective']
    else:
        filter_2d_plus_beam = bl_dic['effective']
    el_150, cl_150 = flatsky.map2cl(mapparams, (frame_flat_T_150/ core.G3Units.uK)*frame_flat_apod_mask, filter_2d = filter_2d_plus_beam**2.)
    print('p1', flush=True)
    el_ilc, cl_ilc = flatsky.map2cl(mapparams, (ilc_map/ core.G3Units.uK)*frame_flat_apod_mask, filter_2d = filter_2d_plus_beam**2.)
    el_ilc_sri, cl_ilc_sri = flatsky.map2cl(mapparams, ilc_map_new*frame_flat_apod_mask, filter_2d = filter_2d_plus_beam**2.)
    save_ps = np.column_stack((el_150, cl_150, cl_ilc, cl_ilc_sri))
    np.save('./cls_save.npy',save_ps)
    #el_cross_ip_op, cl_cross_ip_op = flatsky.map2cl(mapparams, cmb_map_input_beam_tf_conv, flatskymap2 = ilc_map, filter_2d = filter_2d_plus_beam**2.)
    print('p2', flush=True)
    subplot(111, yscale = 'log')#, xscale = 'log')
    el_ = el_ilc

    dl_fac = (el_ * (el_ + 1)) / 2 / np.pi
    plot(el_, dl_fac * cl_150/ core.G3Units.uK**2., 'purple', label = r'150 GHz')
    plot(el_, dl_fac * cl_ilc/ core.G3Units.uK**2., 'orangered', label = r'MV ILC CMB')
    plot(el_, dl_fac * cl_ilc_sri/ core.G3Units.uK**2., 'limegreen', label = r'MV ILC CMB new')
    print('p3', flush=True)
    #analytic curves
    ###plot(el_, dl_fac * res_ilc_curves['cmbmv']/ core.G3Units.uK**2., 'orangered', ls = '--')#, label = r'Analytic ILC')
    ###plot([],[], 'k--', label = r'Analytic')

    ###dl_fac = (el * (el + 1)) / 2 / np.pi
    ###plot(el, dl_fac * cl/ core.G3Units.uK**2., color = 'gray', label = r'CMB')

    legend(loc=1, ncol = 2, fontsize = 6)
    xlim(lmin+10, lmax-10); ylim(1e-2, 1e5)
    xlabel(r'Multipole $\ell$'); ylabel(r'D$_{\ell}$ [$\mu$K$^{2}$]')
    title(r'CMB/kSZ')

    suptitle(r'SPT 100d super-mega-deep field', y = 1.05, fontsize = 14)
    show()
    print('p4', flush=True)

plot_weights=False
if plot_weights:    
    #plot weights now
    lx, ly = flatsky.get_lxly( [nx, ny, dx] )
    plt.subplot(111)
    weightsarr_for_sum = []
    for frqcntr, freq in enumerate( freqs ):
        band = '%sGHz' %(freq)
        print(band)
        curr_weights = weights_arr[frqcntr]
        print(frqcntr)
        print(curr_weights.shape) 
        tit = 'MV CMB ILC'
        acap = ilc.get_freq_response(bands, experiments, component='CMB').T
        rad_prf = flatsky.radial_profile(curr_weights, (lx,ly), bin_size = 100, minbin = 100, maxbin = 10000, to_arcmins = 0) # Behzad: original minbin = 100, maxbin = 10000
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

get_powerspec_new=False
if get_powerspec_new:
    print('\t calculate power spectra')
    #cl_map1=mapspectra.map_analysis.calculate_powerspectra(frame, delta_l=1, lmin=300, lmax=13000)#, apod_mask = 'from_weight') 
    cl_150=mapspectra.basicmaputils.simple_cls(frame_flat_T_150*1000*frame_flat_apod_mask, apod_mask=None, delta_ell=10, ell_min=2000, ell_max=6000, res=0.5)#frame_flat_T_150*1000/ core.G3Units.uK
    print(cl_150)
    cl_ilc_map=mapspectra.basicmaputils.simple_cls(ilc_map, apod_mask=frame_flat_apod_mask, delta_ell=10, ell_min=2000, ell_max=6000, res=0.5)
    cl_ilc_map_new=mapspectra.basicmaputils.simple_cls(ilc_map_new, apod_mask=frame_flat_apod_mask, delta_ell=10, ell_min=2000, ell_max=6000, res=0.5)
    # simple_cls(map1, map2=None, apod_mask=None, smooth=True, ell_bins=None, delta_ell=None, ell_min=None, ell_max=None, res=None, return_2d=False)
    #cl_test=mapspectra.map_analysis.calculate_powerspectra(map1_test, delta_l=1, lmin=2, lmax=5000)#, apod_mask = 'from_weight') #Joshua's map with mask
    #cl_map2=mapspectra.map_analysis.calculate_powerspectra(map2, delta_l=1, lmin=2, lmax=5000) #Eduardo's map w/o mask (wrong normalisation)
    #cl_map2_mod=mapspectra.map_analysis.calculate_powerspectra(map2_mod, delta_l=1, lmin=2, lmax=5000, apod_mask = 'from_weight') #Eduardo's map with mask

    els=cl_150[0]#np.arange(300,13000,1)

    #cl_plot=cl_test['TT']/(1e-6*np.mean(big_field_mask_del**2))
    #dl_plot_camb=(cl[0][300:13000]*els*(els+1.))/(2.*np.pi)
    dl_150=((cl_150[1]/1e-6)*els*(els+1.))/(2.*np.pi)
    dl_ilc_map=((cl_ilc_map[1]/1e-6)*els*(els+1.))/(2.*np.pi)
    dl_ilc_map_new=((cl_ilc_map_new[1]/1e-6)*els*(els+1.))/(2.*np.pi)
    
    pyplot.plot(els,dl_150,label='150 GHz')
    pyplot.plot(els,dl_ilc_map,label='ilc map')
    pyplot.plot(els,dl_ilc_map_new,label='ilc map new')

    # bin_means, bin_edges, binnumber = stats.binned_statistic(els, dl_150, statistic='mean', bins=500)
    # bin_stds, bin_edges, binnumber = stats.binned_statistic(els, dl_150, statistic='std', bins=500)
    # bin_count, bin_edges, binnumber = stats.binned_statistic(els, dl_150, statistic='count', bins=500)
    # errors=bin_stds/np.sqrt(bin_count)

    # ##pyplot.plot(els,dl_plot,'bo',label='Data')
    # bin_width = (bin_edges[1] - bin_edges[0])
    # bin_centers = bin_edges[1:] - bin_width/2
    # pyplot.errorbar(bin_centers,bin_means,yerr=errors,fmt='bo')#,label='Data')
    # pyplot.plot(els,dl_plot_camb,label='CAMB')

    #pyplot.xscale('log')
    #pyplot.yscale('log')
    pyplot.legend()
    pyplot.show() 

    # bin_means, bin_edges, binnumber = stats.binned_statistic(els, dl_plot/dl_plot_camb, statistic='mean', bins=500)
    # bin_stds, bin_edges, binnumber = stats.binned_statistic(els, dl_plot/dl_plot_camb, statistic='std', bins=500)
    # bin_count, bin_edges, binnumber = stats.binned_statistic(els, dl_plot/dl_plot_camb, statistic='count', bins=500)
    # errors=bin_stds/np.sqrt(bin_count)

    # pyplot.errorbar(bin_centers,bin_means,yerr=errors,fmt='bo')#,label='Data')

    # #pyplot.plot(els,dl_plot/dl_plot_camb,'bo',label='data/camb')
    # pyplot.axhline(y = 1.0, color = 'r', linestyle = '-')
    # pyplot.yscale('log')
    # pyplot.legend()
    # pyplot.show()  
