import sys
#sys.path.append('/home/rptd37/spt3g/RHEL_7_x86_64')
import os
import spt3g
from spt3g import core
import numpy as np
from spt3g import maps
from spt3g.mapspectra.map_analysis import calculate_powerspectra
import matplotlib.pyplot as plt
from spt3g.maps.maputils import healpix_to_flatsky
from spt3g.std_processing import CreateFieldMapStub

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

# input g3 file
data_dir = '/data/gpfs/projects/punim1720/spt_maps/nls/'
example_file = os.path.join(data_dir, 'signflip_000_bundle_000.g3')

# core.G3File returns an iterator, 
# which iterates over the frames in the file
g3file = core.G3File(example_file)

# Access the first frame:
frame1 = g3file.next()
#maps.RemoveWeights(frame1, zero_nans = True)
#print(frame1)
frame2 = g3file.next()
#maps.RemoveWeights(frame2, zero_nans = True)
#print(frame2)
frame3 = g3file.next()
T=frame1["T"]; Q=frame1["Q"]; U=frame1["U"]; W=frame1["Wpol"]
print(T.shape,Q.shape,U.shape,W.shape)
maps.save_skymap_fits("./nls/noise_map_95GHz_Wpol.fits", T=frame1["T"], Q=frame1["Q"], U=frame1["U"], W=frame1["Wpol"], overwrite=False, compress=False)
maps.save_skymap_fits("./nls/noise_map_150GHz_Wpol.fits", T=frame2["T"], Q=frame2["Q"], U=frame2["U"],W=frame2["Wpol"], overwrite=False, compress=False)
maps.save_skymap_fits("./nls/noise_map_220GHz_Wpol.fits", T=frame3["T"], Q=frame3["Q"], U=frame3["U"],W=frame3["Wpol"], overwrite=False, compress=False)
sys.exit()

weights_mask = np.float32(frame1["Wpol"])
weights_mask[np.where(np.isnan(weights_mask))] = 0.

frame_flat_apod_mask_temp = healpix_to_flatsky(weights_mask,map_stub=CreateFieldMapStub(res=0.5 * core.G3Units.arcmin, width=80 * core.G3Units.deg, height=35 * core.G3Units.deg, pol_type=maps.MapPolType.T, proj = spt3g.maps.MapProjection(0)), interp=True, rebin=1,)
frame_flat_apod_mask=get_apod_mask(frame_flat_apod_mask_temp, weight_threshold = 0.1, hanning_rad = 60.)

q_95 = np.float32(frame1["Q"])
q_150 = np.float32(frame2["Q"])
q_220 = np.float32(frame3["Q"])
q_95[np.where(np.isnan(q_95))] = 0.
#q_150[np.where(np.isnan(q_150))] = 0.
#q_220[np.where(np.isnan(q_220))] = 0.
frame_flat_Q_95 = healpix_to_flatsky(q_95,map_stub=CreateFieldMapStub(res=0.5 * core.G3Units.arcmin, width=80 * core.G3Units.deg, height=35 * core.G3Units.deg, pol_type=maps.MapPolType.Q, proj = spt3g.maps.MapProjection(0)), interp=True, rebin=1,)
# frame_flat_Q_150 = healpix_to_flatsky(q_150,map_stub=CreateFieldMapStub(res=0.5 * core.G3Units.arcmin, width=80 * core.G3Units.deg, height=35 * core.G3Units.deg, pol_type=maps.MapPolType.Q, proj = spt3g.maps.MapProjection(0)), interp=True, rebin=1,)
# frame_flat_Q_220 = healpix_to_flatsky(q_220,map_stub=CreateFieldMapStub(res=0.5 * core.G3Units.arcmin, width=80 * core.G3Units.deg, height=35 * core.G3Units.deg, pol_type=maps.MapPolType.Q, proj = spt3g.maps.MapProjection(0)), interp=True, rebin=1,)

# u_95 = np.float32(frame1["u"])
# u_150 = np.float32(frame2["u"])
# u_220 = np.float32(frame3["u"])
# u_95[np.where(np.isnan(u_95))] = 0.
# u_150[np.where(np.isnan(u_150))] = 0.
# u_220[np.where(np.isnan(u_220))] = 0.
# frame_flat_U_95 = healpix_to_flatsky(u_95,map_stub=CreateFieldMapStub(res=0.5 * core.G3Units.arcmin, width=80 * core.G3Units.deg, height=35 * core.G3Units.deg, pol_type=maps.MapPolType.U, proj = spt3g.maps.MapProjection(0)), interp=True, rebin=1,)
# frame_flat_U_150 = healpix_to_flatsky(u_150,map_stub=CreateFieldMapStub(res=0.5 * core.G3Units.arcmin, width=80 * core.G3Units.deg, height=35 * core.G3Units.deg, pol_type=maps.MapPolType.U, proj = spt3g.maps.MapProjection(0)), interp=True, rebin=1,)
# frame_flat_U_220 = healpix_to_flatsky(u_220,map_stub=CreateFieldMapStub(res=0.5 * core.G3Units.arcmin, width=80 * core.G3Units.deg, height=35 * core.G3Units.deg, pol_type=maps.MapPolType.U, proj = spt3g.maps.MapProjection(0)), interp=True, rebin=1,)

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
tf_twod = None

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

get_powerspec_org=True
if get_powerspec_org:
    print('p0', flush=True)
    #cl of input and output
    filter_2d_plus_beam = bl_dic['effective']
    el_95, cl_95 = flatsky.map2cl(mapparams, (frame_flat_Q_95/ core.G3Units.uK)*frame_flat_apod_mask, filter_2d = filter_2d_plus_beam**2.)
    print('p1', flush=True)
    save_ps = np.column_stack((el_95, cl_95))#, cl_ilc, cl_ilc_sri))
    np.save('./cls_save.npy',save_ps)
    print('p2', flush=True)
    subplot(111, yscale = 'log')#, xscale = 'log')
    el_ = el_95

    dl_fac = (el_ * (el_ + 1)) / 2 / np.pi
    plot(el_, dl_fac * cl_95/ core.G3Units.uK**2., 'purple', label = r'150 GHz')
    print('p3', flush=True)

    legend(loc=1, ncol = 2, fontsize = 6)
    xlim(lmin+10, lmax-10); ylim(1e-2, 1e5)
    xlabel(r'Multipole $\ell$'); ylabel(r'D$_{\ell}$ [$\mu$K$^{2}$]')
    title(r'CMB/kSZ')

    suptitle(r'SPT 100d super-mega-deep field', y = 1.05, fontsize = 14)
    show()
    print('p4', flush=True)

#nl_95=calculate_powerspectra(frame1, lmin=300, lmax=13000, apod_mask=None, qu_eb="both",) # apod_mask="from_weight" 
###nl_95=calculate_powerspectra(frame_flat_Q_95, lmin = 300, lmax = 13000, delta_l = 5, apod_mask = frame_flat_apod_mask, qu_eb = 'qu')
###print(nl_95)
###print(nl_95.shape)

# def get_noise_spectra(map1, map2, lmin = 0, lmax = 10000, delta_l = 5, apod_mask = 'from_weight', qu_eb = 'qu', verbose = False, rough_tf_fac = 1., return_2d = False, real = False, ell_range_for_white_noise = [3000.0, 5000.0]):
#     diffmap = map_analysis.subtract_two_maps(map1, map2, divide_by_two = True)

#     np.asarray(diffmap['T'])[:] = np.asarray(diffmap['T'])[:]/rough_tf_fac**0.5
#     nl = map_analysis.calculate_powerspectra(diffmap, lmin = lmin, lmax = lmax, delta_l = delta_l, apod_mask = apod_mask, qu_eb = qu_eb)
#     binned_el = nl['TT'].lbins[:-1] + (nl['TT'].delta_l/2.)
#     noise_uk_arcmin = map_analysis.calculateNoise(diffmap, verbose = verbose, ell_range=ell_range_for_white_noise, apod_mask = apod_mask, qu_eb = qu_eb)
#     if return_2d:
#         nl_2d = map_analysis.calculate_powerspectra(diffmap, lmin = lmin, lmax = lmax, delta_l = delta_l, apod_mask = apod_mask, qu_eb = qu_eb, return_2d  =return_2d, real = real)
#         return binned_el, nl, noise_uk_arcmin, nl_2d
#     else:
#         #noise_uk_arcmin = None
#         return binned_el, nl, noise_uk_arcmin

#maps.save_skymap_fits("/sptlocal/user/rptd37/SPT_maps/spt_maps/no_signflip_bundle_000_90GHz_nw_Wpol.fits", T=frame1["T"], Q=frame1["Q"], U=frame1["U"], W=frame1["Wpol"], overwrite=False, compress=False)
#maps.save_skymap_fits("/sptlocal/user/rptd37/SPT_maps/spt_maps/no_signflip_bundle_000_150GHz_nw_Wpol.fits", T=frame2["T"], Q=frame2["Q"], U=frame2["U"],W=frame2["Wpol"], overwrite=False, compress=False)
#maps.save_skymap_fits("/sptlocal/user/rptd37/SPT_maps/spt_maps/no_signflip_bundle_000_220GHz_nw_Wpol.fits", T=frame3["T"], Q=frame3["Q"], U=frame3["U"],W=frame3["Wpol"], overwrite=False, compress=False)


# import healpy as hp
# import numpy as np

# # Load the HEALPix map
# nside = 8192
# #map = hp.read_map("cmb_map.fits", verbose=False)

# # Compute the power spectrum
# nl_Q_95 = hp.anafast(frame1["Q"], lmax=13000, pol=False)

# # Plot the power spectrum
# import matplotlib.pyplot as plt
# ell = np.arange(len(nl_Q_95))
# plt.plot(ell, nl_Q_95)
# plt.xlabel(r'$\ell$')
# plt.ylabel(r'$N_{\ell}$')
# plt.show()
