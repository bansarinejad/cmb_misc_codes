import sys
#sys.path.append('/home/rptd37/spt3g/spt3g_software-master/')
import os
import spt3g
from spt3g import core,dfmux,std_processing, mapmaker, maps, mapspectra
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np , argparse
import healpy as hp
from pylab import *
from tqdm import tqdm
import pymaster as nmt
sys_path_folder='../python/'
sys.path.append(sys_path_folder)
import flatsky, tools, tools_data, lensing, foregrounds, misc
import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
print('\n')
import gc

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

print('\tread/get necessary params')
param_dict = misc.get_param_dict(paramfile)
data_folder = param_dict['data_folder']

pol = False

# input cluster catalog
hdul = fits.open('./y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_winter_lgt20_edge_ps_cut_use_m200cut.fit')
#hdul = fits.open('./randoms_50K_y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_winter_lgt20_edge_ps_cut_use.fit')
#hdul = fits.open('./y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_winter_lgt20_edge_use_m200cut_12arcmin_ps_cut.fit')
#hdul = fits.open('./randoms_50K_y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_winter_lgt20_edge_12arcmin_ps_cut_use.fit')
data = hdul[1].data
RA_use = data['RA']
DEC_use = data['DEC']
nclustersorrandoms=len(RA_use)

#read noise level
noiseval = param_dict['noiseval'] #uK-arcmin

#CMB power spectrum
cls_file = '%s/%s' %(param_dict['data_folder'], param_dict['cls_file'])
el_camb, cl_camb = tools.get_cmb_cls(cls_file, pol = pol)
nl_dic = tools.get_nl_dic(noiseval, el_camb, pol = pol)

#rotate maps from Galactic system to Celectial (equtorial) system
rot = hp.Rotator(coord=['G','C'])

#read Planck Point source & galactic plane masks and the CMB file
hmap_gp_mask = hp.fitsfunc.read_map('./lindsey_maps/SPTSZ_dust_mask_top_2p5percent.fits',field=0)
#hmap_gp_mask = hp.fitsfunc.read_map('./planck/COM_CMB_IQU-commander_2048_R3.00_full.fits',field=3)
#hmap_gp_mask = rot.rotate_map_pixel(hmap_gp_mask) # rot step not needed for Lindsey's maps

hmap_ps_mask = hp.fitsfunc.read_map('./lindsey_maps/SPTSZ_point_source_mask_nside_8192_binary_mask.fits',field=0)
#hmap_ps_mask = hp.fitsfunc.read_map('./planck/HFI_Mask_PointSrc_2048_R2.00.fits',field=1)
#hmap_ps_mask = rot.rotate_map_pixel(hmap_ps_mask) # rot step not needed for Lindsey's maps

aposcale = 0.03  #in degrees
mask_Sm = nmt.mask_apodization(hmap_ps_mask, aposcale, apotype="Smooth") #apodizing point source mask

#hmap_T = hp.fitsfunc.read_map('./planck/COM_CMB_IQU-commander_2048_R3.00_full.fits',field=5) #using inpainted map so dont have to apply point source mask
hmap_T = hp.fitsfunc.read_map('./lindsey_maps/SPTSZ_Planck_tsz_nulled_cmb_map.fits',field=0)
#hmap_T = hp.fitsfunc.read_map('./planck/COM_CMB_IQU-commander_2048_R3.00_full.fits',field=0)

#hmap_T = hp.fitsfunc.read_map('./planck/COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits',field=1) #using inpainted map so dont have to apply point source mask
#hmap_T = hp.fitsfunc.read_map('./planck/COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits',field=0)
#hmap_T = rot.rotate_map_pixel(hmap_T) # rot step not needed for Lindsey's maps

#Calculate Planck nl from the data 
hmap_T_HM1 = hp.fitsfunc.read_map('./lindsey_maps/SPTSZ_Planck_tsz_nulled_cmb_map_half1.fits',field=0)
#hmap_T_HM1 = hp.fitsfunc.read_map('./planck/COM_CMB_IQU-commander_2048_R3.00_hm1.fits',field=0)
#hmap_T_HM1 = hp.fitsfunc.read_map('./planck/COM_CMB_IQU-smica-nosz_2048_R3.00_hm1.fits',field=0)
#hmap_T_HM1 = rot.rotate_map_pixel(hmap_T_HM1) # rot step not needed for Lindsey's maps

hmap_T_HM2 = hp.fitsfunc.read_map('./lindsey_maps/SPTSZ_Planck_tsz_nulled_cmb_map_half2.fits',field=0)
#hmap_T_HM2 = hp.fitsfunc.read_map('./planck/COM_CMB_IQU-commander_2048_R3.00_hm2.fits',field=0)
#hmap_T_HM2 = hp.fitsfunc.read_map('./planck/COM_CMB_IQU-smica-nosz_2048_R3.00_hm2.fits',field=0)
#hmap_T_HM2 = rot.rotate_map_pixel(hmap_T_HM2) # rot step not needed for Lindsey's maps
hmap_T_diff=(hmap_T_HM1-hmap_T_HM2)/2
hmap_T_diff_masked=hmap_T_diff*hmap_gp_mask*1000000#*mask_Sm
#hmap_T_diff_masked=hmap_T_diff*hmap_gp_mask*1000000#*mask_Sm #for Planck
LMAX = 6000
cl_n,alm_n=hp.anafast(hmap_T_diff_masked, lmax=LMAX, alm=True)
ell_n = np.arange(len(cl_n))

hmap_T_masked=hmap_T*mask_Sm*hmap_gp_mask*1000000
#hmap_T_masked=hmap_T*hmap_gp_mask*1000000 #1000000 factor needed for commander but not for SIMCA INP
LMAX = 6000
cl,alm = hp.anafast(hmap_T_masked, lmax=LMAX, alm=True)
ell = np.arange(len(cl))

bl = tools_data.get_bl(5.0, ell, make_2d = 0)

#set cls to 0 for ell>2000
cl_sigmoid=1/(1+np.exp(0.11*(ell-2000)))  #v2 was 0.05*(ell...), v3 is 0.11*(ell...)
#cl_sigmoid_hpf=1/(1+np.exp(0.05*(-ell+300))) 
PW=hp.sphtfunc.pixwin(2048, pol=False, lmax=LMAX)

#camb_Wiener_filter
#cl_WF_camb=(np.asarray(cl_camb)[0][:6001]/(np.asarray(cl_camb)[0][:6001]+(np.asarray(nl_dic['T'])[:6001]/((PW**2)*(bl**2)))))*np.asarray(cl_sigmoid) #v4 with 40uk.arcmin noise in param file + same new sigmoid function as v2.
cl_WF_camb=(np.asarray(cl_camb)[0][:6001]/(np.asarray(cl_camb)[0][:6001]+(np.asarray(cl_n)/((PW**2)*bl**2))))*np.asarray(cl_sigmoid)#*np.asarray(cl_sigmoid_hpf) #v3 WF has new sigmoid function compared to v2!

#cl_WF_camb=(np.asarray(cl_camb)[0][:6001]/(np.asarray(cl_camb)[0][:6001]+(np.asarray(cl_n)/((PW**2)*bl**2))))*np.asarray(cl_sigmoid) #noHPF
#cl_WF_camb=(np.asarray(cl_camb)[0][:6001]/(np.asarray(cl_camb)[0][:6001]+(np.asarray(nl_dic['T'])[:6001]/(bl**2))))*np.asarray(cl_sigmoid)

#plt.plot(ell,cl_sigmoid); plt.show()
#plt.plot(ell,PW); plt.show()
#plt.plot(ell,bl); plt.show()
#plt.plot(ell,cl_WF_camb); 
#plt.yscale('log')
#plt.xscale('log')
#plt.show()

#filter_data by camb Wiener filter
camb_alm_filtered=hp.sphtfunc.almxfl(alm,cl_WF_camb)
camb_filtered_map=hp.sphtfunc.alm2map(camb_alm_filtered,nside=2048)
#hp.write_map("./planck/filtered_map_test_v10_commander_noINP.fits", camb_filtered_map, overwrite=True)

reso_arcmin = 0.5
ra0, dec0 = 0, -57.5
proj = 0
ny, nx = 400, 400

#data
if (0):
  hmap_T_masked = np.float32(camb_filtered_map)
  hmap_T_masked[np.where(np.isnan(camb_filtered_map))] = 0.
  cutout_T=[]
  for i in tqdm(range(nclustersorrandoms)):
    map_stub_T = maps.FlatSkyMap(
      nx, ny, 
      reso_arcmin*core.G3Units.arcmin, 
      weighted = False, 
      proj = spt3g.maps.MapProjection(proj),
      alpha_center = RA_use[i]*core.G3Units.degrees, 
      delta_center = DEC_use[i]*core.G3Units.degrees,
      coord_ref = maps.MapCoordReference.Equatorial, 
      units = spt3g.core.G3TimestreamUnits.Tcmb, 
      pol_type = maps.MapPolType.T)
    T_map = maps.maputils.healpix_to_flatsky(hmap_T_masked, map_stub=map_stub_T, interp=True)
    cutout_T.append(np.asarray(T_map))
  Cutout_T=np.asarray(cutout_T)
  #np.save('./planck/Planck_cutouts_T_nw_masked_WFv3_COMMANDER.npy', cutout_T)
  np.save('./lindsey_maps/LM_cutouts_T_nw_masked_WFv3.npy', cutout_T)

#randoms
if (1):
  hmap_T_masked = np.float32(camb_filtered_map)
  hmap_T_masked[np.where(np.isnan(camb_filtered_map))] = 0.
  for j in range(0,3,1):
    RA=RA_use[((j+1)*4900-4900):((j+1)*4900)]
    DEC=DEC_use[((j+1)*4900-4900):((j+1)*4900)]
    cutout_T=[]
    nclustersorrandoms=len(RA)
    for i in tqdm(range(nclustersorrandoms)):
      map_stub_T = maps.FlatSkyMap(
        nx, ny, 
        reso_arcmin*core.G3Units.arcmin, 
        weighted = False, 
        proj = spt3g.maps.MapProjection(proj),
        alpha_center = RA[i]*core.G3Units.degrees, 
        delta_center = DEC[i]*core.G3Units.degrees,
        coord_ref=maps.MapCoordReference.Equatorial, 
        units = spt3g.core.G3TimestreamUnits.Tcmb, 
        pol_type=maps.MapPolType.T)
      T_map = maps.maputils.healpix_to_flatsky(hmap_T, map_stub=map_stub_T, interp=True)
      cutout_T.append(np.asarray(T_map))
    cutout_T=np.asarray(cutout_T)
    #np.save('./planck/Planck_cutouts_T_nw_masked_rands_WFv3_COMMANDER_part%s.npy' %(j), cutout_T)
    np.save('./lindsey_maps/LM_cutouts_T_nw_masked_rands_WFv3_part%s.npy' %(j), cutout_T)
    del cutout_T
    del RA
    del DEC
    gc.collect()
  
'''  
#cl_one=np.ones(2000)
#cl_zero=np.zeros(4001)
#cl_onezero=np.concatenate([cl_one, cl_zero])

#data_Winer_filter
#cl_WF=(cl/(cl+cl_n))*cl_onezero
#cl_WF=(cl/(cl+(cl_n/bl**2)))*cl_sigmoid#*cl_onezero
#print(cl_sigmoid[1000],cl_sigmoid[3000])

#filter_data by data Wiener filter
#alm_filtered=hp.sphtfunc.almxfl(alm,cl_WF)
#filtered_map=hp.sphtfunc.alm2map(alm_filtered,nside=2048)
#hp.write_map("./planck/data_filtered_map_test_v8.fits", filtered_map, overwrite=True)

#hmap_ps_mask = hp.fitsfunc.read_map('./planck/HFI_Mask_PointSrc_2048_R2.00.fits',field=1)
#aposcale = 0.03  #in degrees
#mask_Sm = nmt.mask_apodization(hmap_ps_mask, aposcale, apotype="Smooth") #apodizing point source mask

#hmap_T_masked=hmap_T*mask_Sm*hmap_gp_mask*1000000

#beam=hp.sphtfunc.gauss_beam(0.00145444, lmax=LMAX, pol=False) # 0.00145444 is 5' in radians
#bl=hp.sphtfunc.beam2bl(beam,0.00145444, lmax=LMAX)

#alm_WF=hp.sphtfunc.synalm(cl_WF,lmax=LMAX)
#filter_map=hp.sphtfunc.alm2map(alm_WF,nside=2048)
#hp.write_map("./planck/data_filter_map_test_v5.fits", filter_map, overwrite=True)

#cl_WF_camb=(np.asarray(cl_camb)[0][:6001]/(np.asarray(cl_camb)[0][:6001]+(np.asarray(nl_dic['T'])[:6001]/bl**2)))*np.asarray(cl_onezero)
#alm_WF_camb=hp.sphtfunc.synalm(cl_WF_camb,lmax=LMAX)
#camb_filter_map=hp.sphtfunc.alm2map(alm_WF_camb,nside=2048)
#hp.write_map("./planck/camb_filter_map_test_v5.fits", camb_filter_map, overwrite=True)

#plt.scatter(ell_n,cl_n)
#plt.scatter(ell,cl)
#print(cl_camb[0].shape)
#plt.plot(el_camb, cl_camb[0])
#print(nl_dic.keys())
#print(np.asarray(nl_dic)[0].shape)
#plt.plot(el_camb,np.asarray(nl_dic['T'])) 
#plt.yscale('log')
#plt.xscale('log')
#plt.show()

#hp.write_map("./planck/mask_Smnest.fits", mask_Sm, nest=True, overwrite=True)
#hp.write_map("./planck/mask_Smring.fits", mask_Sm, nest=False, overwrite=True)
#hp.mollview(hmap_ps_mask, title='Binary mask', coord=['G', 'C'])
#hp.mollview(mask_Sm, title='Smooth apodization', coord=['G', 'C'])
#plt.show()

hmap_N = hp.fitsfunc.read_map('./planck/COM_CMB_IQU-commander-field-Int_2048_R2.00.fits',field=8)
LMAX = 5000
cl = hp.anafast((hmap_T*hmap_Tm), lmax=LMAX)
nl = hp.anafast((hmap_N*hmap_Nm), lmax=LMAX)
ell = np.arange(len(cl))
save_arr=np.array([ell,cl,nl])
np.save('./test_cl_nl_planck_masked.npy',save_arr)

plt.scatter(ell,cl)
plt.scatter(ell,nl)
plt.yscale('log')
plt.xscale('log')
plt.show()
''' 