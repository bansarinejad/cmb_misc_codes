import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import spt3g
from spt3g import core, maps, mapspectra

do_T=False
do_Q=False
do_U=True

if do_T:
    map_150 = np.float32(hp.fitsfunc.read_map('./winter/no_signflip_bundle_000_150GHz_nw.fits',field=0, partial=True))
    ILC_map = np.float32(hp.fitsfunc.read_map('./winter/T_95_150_220_winterfield_fullsky_ILC.fits',field=0, partial=True))

if do_Q:
    map_150 = np.float32(hp.fitsfunc.read_map('./winter/no_signflip_bundle_000_150GHz_nw.fits',field=1, partial=True))
    ILC_map = np.float32(hp.fitsfunc.read_map('./winter/Q_95_150_220_winterfield_fullsky_ILC.fits',field=0, partial=True))

if do_U:
    map_150 = np.float32(hp.fitsfunc.read_map('./winter/no_signflip_bundle_000_150GHz_nw.fits',field=2, partial=True))
    ILC_map = np.float32(hp.fitsfunc.read_map('./winter/U_95_150_220_winterfield_fullsky_ILC.fits',field=0, partial=True))

apod_mask_curved=np.float32(hp.fitsfunc.read_map('./winter/apod_mask_curved_temp.fits',field=0, partial=True))

ps_mask=np.load('./masks/mask_0p4medwt_6mJy150ghzv2.npy')
hmap_ps_mask =np.float32(spt3g.maps.HealpixSkyMap(ps_mask))

map_150[np.where(np.isnan(map_150))] = 0.
ILC_map[np.where(np.isnan(ILC_map))] = 0.
apod_mask_curved[np.where(np.isnan(apod_mask_curved))] = 0.
hmap_ps_mask[np.where(np.isnan(hmap_ps_mask))] = 0.

map_150=map_150*hmap_ps_mask*apod_mask_curved
ILC_map=ILC_map*hmap_ps_mask*apod_mask_curved

plot_maps_hp=False
if plot_maps_hp:
    plt.figure(figsize=(10, 5))
    hp.mollview(ILC_map)
    plt.title('ILC map')
    plt.show()

plot_map_cutout=False
if plot_map_cutout:
    plt.figure(figsize=(10, 5))
    fov = 20.0 # 20 arcmin field of view
    center = [320.0, -45.0] # cutout centered on the north pole
    hp.visufunc.gnomview(map_150, coord='E', reso=0.5, rot=center, xsize=200)
    plt.show()
    hp.visufunc.gnomview(ILC_map, coord='E', reso=0.5, rot=center, xsize=200)
    plt.show()

calc_cls=False
if calc_cls:
    cl_150 = hp.sphtfunc.anafast(map_150,lmax=13000)
    cl_ILC = hp.sphtfunc.anafast(ILC_map,lmax=13000)

    ell = np.arange(len(cl_150))
    plt.plot(ell, cl_150, label='Q 150 map')
    plt.plot(ell, cl_ILC, label='Q ILC map')
    plt.xlabel('Multipole moment $\ell$')
    plt.ylabel('Power spectrum $C_\ell$')
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()

calc_nls=True
if calc_nls:
    noise_map_90=np.float32(hp.fitsfunc.read_map('./nls/new_signflip_000_bundle_000_090ghz.fits',field=0, partial=True))
    nl_90 = hp.sphtfunc.anafast(noise_map_90,lmax=13000)
    np.save('./new_maps_nl_90.npy', nl_90)

    noise_map_150=np.float32(hp.fitsfunc.read_map('./nls/new_signflip_000_bundle_000_150ghz.fits',field=0, partial=True))
    nl_150 = hp.sphtfunc.anafast(noise_map_150,lmax=13000)
    np.save('./new_maps_nl_150.npy', nl_150)

    noise_map_220=np.float32(hp.fitsfunc.read_map('./nls/new_signflip_000_bundle_000_220ghz.fits',field=0, partial=True))
    nl_220 = hp.sphtfunc.anafast(noise_map_220,lmax=13000)
    np.save('./new_maps_nl_220.npy', nl_220)