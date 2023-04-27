import sys
#sys.path.append('/data/gpfs/projects/punim1720/spt3g-software/spt3g_software-master/')
import os
import spt3g
from spt3g import core,dfmux,std_processing, mapmaker, maps
from astropy.io import fits
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

# input cluster catalog
hdul = fits.open('./randoms_50K_y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_winter_lgt20_edge_ps_cut_use.fit')#./randoms_50K_y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_winter_lgt20_edge_12arcmin_ps_cut_use.fit')
data = hdul[1].data
RA_use = data['RA']
DEC_use = data['DEC']

hmap_fname = './winter/no_signflip_bundle_000_150GHz_nw.fits'
mask=np.load('./masks/mask_0p4medwt_6mJy150ghzv2.npy')
hmap_m =spt3g.maps.HealpixSkyMap(mask)

reso_arcmin = 0.5
#ra0, dec0 = 0, -57.5
proj = 0
ny, nx = 400, 400
#map_stub_T = maps.FlatSkyMap(nx, ny, reso_arcmin*core.G3Units.arcmin, weighted = False, proj = spt3g.maps.MapProjection(proj),alpha_center = ra0*core.G3Units.degrees, delta_center = dec0*core.G3Units.degrees, coord_ref=maps.MapCoordReference.Equatorial, units = spt3g.core.G3TimestreamUnits.Tcmb, pol_type=maps.MapPolType.T)

if (1):
  hmap_T = hp.fitsfunc.read_map('./winter/T_95_150_220_winterfield_fullsky_ILC.fits',field=0, partial=True)
  hmap_T = np.float32(hmap_T)
  hmap_T[np.where(np.isnan(hmap_T))] = 0.  
  for j in range(5,6,1):
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
    np.save('./merged_rand_parts_nw/ILC_cutouts_T_rands_nw_part%s.npy' %(j), cutout_T)
    del cutout_T
    del RA
    del DEC
    gc.collect()

if (0):
  hmap_Q = hp.fitsfunc.read_map('./winter/Q_95_150_220_winterfield_fullsky_ILC.fits',field=0, partial=True)
  hmap_Q = np.float32(hmap_Q)
  hmap_Q[np.where(np.isnan(hmap_Q))] = 0.
  for j in range(9,10,1):
    RA=RA_use[((j+1)*4900-4900):((j+1)*4900)]
    DEC=DEC_use[((j+1)*4900-4900):((j+1)*4900)]
    cutout_Q=[]
    nclustersorrandoms=len(RA)
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
    cutout_Q=np.asarray(cutout_Q)
    np.save('./merged_rand_parts_nw/ILC_cutouts_Q_rands_nw_part%s.npy' %(j), cutout_Q)
    del cutout_Q
    del RA
    del DEC
    gc.collect()

if (0):
  hmap_U = hp.fitsfunc.read_map('./winter/U_95_150_220_winterfield_fullsky_ILC.fits',field=0, partial=True)
  hmap_U = np.float32(hmap_U)
  hmap_U[np.where(np.isnan(hmap_U))] = 0.
  for j in range(0,1,1):
    RA=RA_use[((j+1)*4900-4900):((j+1)*4900)]
    DEC=DEC_use[((j+1)*4900-4900):((j+1)*4900)]
    cutout_U=[]
    nclustersorrandoms=len(RA)
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
    cutout_U=np.asarray(cutout_U)
    np.save('./merged_rand_parts_nw/ILC_cutouts_U_rands_nw_part%s.npy' %(j), cutout_U)
    del cutout_U
    del RA
    del DEC
    gc.collect()
    
if (0):
  hmap_m = np.float32(hmap_m)
  hmap_m[np.where(np.isnan(hmap_m))] = 0.  
  for j in range(8,10,1):
    RA=RA_use[((j+1)*4900-4900):((j+1)*4900)]
    DEC=DEC_use[((j+1)*4900-4900):((j+1)*4900)]
    cutout_m=[]
    nclustersorrandoms=len(RA)
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
    np.save('./merged_rand_parts_nw/rands_mask_0p4medwt_6mJy150ghzv2_12arcmin_ps_part%s.npy' %(j), cutout_m)
    del cutout_m
    del RA
    del DEC
    gc.collect()

    
    
