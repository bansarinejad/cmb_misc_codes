import sys
#sys.path.append('/home/rptd37/spt3g/spt3g_software-master/')
import os
import spt3g
from spt3g import core,dfmux,std_processing, mapmaker, maps, mapspectra
from astropy.io import fits
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from tqdm import tqdm


# input cluster catalog
hdul = fits.open('./y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_winter_lgt20_edge_ps_cut_use_m200cut.fit')#./y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_winter_lgt20_edge_use_m200cut_12arcmin_ps_cut.fit')
data = hdul[1].data
RA = data['RA']
DEC = data['DEC']
nclustersorrandoms=len(RA)

#hmap_fname = './winter/no_signflip_bundle_000_150GHz_nw.fits'
mask=np.load('./masks/mask_0p4medwt_6mJy150ghzv2.npy')
hmap_m =spt3g.maps.HealpixSkyMap(mask)

reso_arcmin = 0.5
#ra0, dec0 = 0, -57.5
proj = 0
ny, nx = 400, 400
#map_stub_T = maps.FlatSkyMap(nx, ny, reso_arcmin*core.G3Units.arcmin, weighted = False, proj = spt3g.maps.MapProjection(proj),alpha_center = ra0*core.G3Units.degrees, delta_center = dec0*core.G3Units.degrees, coord_ref=maps.MapCoordReference.Equatorial, units = spt3g.core.G3TimestreamUnits.Tcmb, pol_type=maps.MapPolType.T)
'''
# calculate noise level
if (0):
  hmap_T = hp.fitsfunc.read_map(hmap_fname,field=0)
  hmap_T = np.float32(hmap_T)
  hmap_T[np.where(np.isnan(hmap_T))] = 0.
  hmap_Thp=spt3g.maps.HealpixSkyMap(hmap_T)
  print(hmap_T)
  print(hmap_Thp)
  #noise_test= mapspectra.map_analysis.calculateNoise(hmap_Thp,apod_mask='from_weight', ell_range=[5500.0, 6500.0], return_all=False, verbose=True, qu_eb='qu')
  noise_test= mapspectra.map_analysis.calculateNoise(hmap_Thp,apod_mask=None, ell_range=[5500.0, 6500.0], return_all=False, verbose=True, qu_eb='qu')
  #noise_test= mapspectra.map_analysis.calculateNoise(hmap_T, ell_range=[5500.0, 6500.0], return_all=False, verbose=True, qu_eb='qu')
  print(noise_test)
'''
if (0):
  hmap_T = hp.fitsfunc.read_map('./winter/T_95_150_220_winterfield_fullsky_ILC.fits',field=0, partial=True)
  hmap_T = np.float32(hmap_T)
  hmap_T[np.where(np.isnan(hmap_T))] = 0.
  cutout_T=[]
  for i in tqdm(range(nclustersorrandoms)):
    map_stub_T = maps.FlatSkyMap(
      nx, ny, 
      reso_arcmin*core.G3Units.arcmin, 
      weighted = False, 
      proj = spt3g.maps.MapProjection(proj),
      alpha_center = RA[i]*core.G3Units.degrees, 
      delta_center = DEC[i]*core.G3Units.degrees,
      coord_ref = maps.MapCoordReference.Equatorial, 
      units = spt3g.core.G3TimestreamUnits.Tcmb, 
      pol_type = maps.MapPolType.T)
    T_map = maps.maputils.healpix_to_flatsky(hmap_T, map_stub=map_stub_T, interp=True)
    cutout_T.append(np.asarray(T_map))
  Cutout_T=np.asarray(cutout_T)
  np.save('./ILC_cutouts_T_nw.npy', cutout_T)

if (0):
  hmap_Q = hp.fitsfunc.read_map('./winter/Q_95_150_220_winterfield_fullsky_ILC.fits',field=0, partial=True)
  hmap_Q = np.float32(hmap_Q)
  hmap_Q[np.where(np.isnan(hmap_Q))] = 0.
  cutout_Q=[]
  for i in tqdm(range(nclustersorrandoms)):
    map_stub_Q = maps.FlatSkyMap(
      nx, ny, 
      reso_arcmin*core.G3Units.arcmin, 
      weighted = False, 
      proj = spt3g.maps.MapProjection(proj),
      alpha_center = RA[i]*core.G3Units.degrees, 
      delta_center = DEC[i]*core.G3Units.degrees,
      coord_ref=maps.MapCoordReference.Equatorial, 
      units = spt3g.core.G3TimestreamUnits.Tcmb, 
      pol_type=maps.MapPolType.Q)
    Q_map = maps.maputils.healpix_to_flatsky(hmap_Q, map_stub=map_stub_Q, interp=True)
    cutout_Q.append(np.asarray(Q_map))
  Cutout_Q=np.asarray(cutout_Q)
  np.save('./ILC_cutouts_Q_nw.npy', cutout_Q)
  
if (1):
  hmap_U = hp.fitsfunc.read_map('./winter/U_95_150_220_winterfield_fullsky_ILC.fits',field=0, partial=True)
  hmap_U = np.float32(hmap_U)
  hmap_U[np.where(np.isnan(hmap_U))] = 0.
  cutout_U=[]
  for i in tqdm(range(nclustersorrandoms)):
    map_stub_U = maps.FlatSkyMap(
      nx, ny, 
      reso_arcmin*core.G3Units.arcmin, 
      weighted = False, 
      proj = spt3g.maps.MapProjection(proj),
      alpha_center = RA[i]*core.G3Units.degrees, 
      delta_center = DEC[i]*core.G3Units.degrees,
      coord_ref=maps.MapCoordReference.Equatorial, 
      units = spt3g.core.G3TimestreamUnits.Tcmb) 
      #pol_type=maps.MapPolType.U)
    U_map = maps.maputils.healpix_to_flatsky(hmap_U, map_stub=map_stub_U, interp=True)
    cutout_U.append(np.asarray(U_map))
  Cutout_U=np.asarray(cutout_U)
  np.save('./ILC_cutouts_U_nw.npy', cutout_U)
    
if (0):
  hmap_m = np.float32(hmap_m)
  hmap_m[np.where(np.isnan(hmap_m))] = 0.
  cutout_m=[]
  for i in tqdm(range(nclustersorrandoms)):
    map_stub_m = maps.FlatSkyMap(
      nx, ny, 
      reso_arcmin*core.G3Units.arcmin, 
      weighted = False, 
      proj = spt3g.maps.MapProjection(proj),
      alpha_center = RA[i]*core.G3Units.degrees, 
      delta_center = DEC[i]*core.G3Units.degrees,
      coord_ref=maps.MapCoordReference.Equatorial, 
      units = spt3g.core.G3TimestreamUnits.Tcmb, 
      pol_type=maps.MapPolType.T)
    m_map = maps.maputils.healpix_to_flatsky(hmap_m, map_stub=map_stub_m, interp=True)
    cutout_m.append(np.asarray(m_map))
  cutout_m=np.asarray(cutout_m)
  np.save('./mask_0p4medwt_6mJy150ghzv2_12arcmin_ps_cutouts.npy', cutout_m)


