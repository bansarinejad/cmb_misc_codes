#!/usr/bin/env python
########################

########################
#load desired modules
import numpy as np, sys, os, argparse, glob #data as sc,
sys_path_folder='../python/'
sys.path.append(sys_path_folder)

import flatsky, tools, lensing, foregrounds, misc

from tqdm import tqdm
from matplotlib import pylab
from pylab import *
cmap = cm.RdYlBu_r
import matplotlib.pyplot as plt
import pickle

import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
print('\n')
########################

########################
parser = argparse.ArgumentParser(description='')
#parser.add_argument('-dataset_fname', dest='dataset_fname', action='store', help='dataset_fname', type=str, default='./stacks/clusters_m200cut_T_ILC_healpix_nw_apod_zeros_masked_inpaint_planckgrad_WFv3_COMMANDER_v2_org_Behzad_fullorient.pkl')
#parser.add_argument('-dataset_fname', dest='dataset_fname', action='store', help='dataset_fname', type=str, default='./stacks/clusters_m200cut_U_ILC_healpix_nw_apod_zeros_masked_inpaint_v3_org_Behzad_fullorient.npy')
#parser.add_argument('-dataset_fname', dest='dataset_fname', action='store', help='dataset_fname', type=str, default='./stacks/clusters_m200cut_U_ILC_healpix_nw_apod_zeros_masked_inpaint_v3_fullorient_rot_test.npy')
#parser.add_argument('-dataset_fname', dest='dataset_fname', action='store', help='dataset_fname', type=str, default='./stacks/grad_tests/clusters_m200cut_Q_ILC_healpix_nw_apod_zeros_masked_inpaint_v3_grad_orient.npy')
#parser.add_argument('-dataset_fname', dest='dataset_fname', action='store', help='dataset_fname', type=str, default='../cmb_cluster_lensing/results//nx240_dx0.5/beam1.2/noise3.2/10amcutouts/grad_oreint_m200mean_2.3e+14/5474clusters/withgaussianfg_withclustertsz_withclusterksz/TQU/5474sims0to1-1.npy')
#parser.add_argument('-dataset_fname', dest='dataset_fname', action='store', help='dataset_fname', type=str, default='../results/nx240_dx0.5/beam1.17/noise9.4/20amcutouts/m200mean/5474clusters/withgaussianfg_withclustertsz_withclusterksz/TQU/clusters_m200cut_Q_ILC_healpix_nw_apod_zeros_masked_inpaint_v3.npy')
#parser.add_argument('-dataset_fname', dest='dataset_fname', action='store', help='dataset_fname', type=str, default='../results/nx240_dx0.5/beam1.17/noise9.4/20amcutouts/m200mean/5474clusters/withgaussianfg_withclustertsz_withclusterksz/TQU/clusters_m200cut_T_ILC_healpix_nw_apod_zeros_masked_inpaint_planckgrad_WFv3_COMMANDER_v2.pkl')#, allow_pickle= True)
parser.add_argument('-dataset_fname', dest='dataset_fname', action='store', help='dataset_fname', type=str, default='./stacks/clusters_m200cut_T_ILC_healpix_nw_apod_zeros_masked_inpaint_planckgrad_WFv3_COMMANDER_v2_org_Behzad_fullorient_rot_test.pkl')
parser.add_argument('-paramfile', dest='paramfile', action='store', help='paramfile', type=str, default='params_data.ini')
#parser.add_argument('-dataset_fname', dest='dataset_fname', action='store', help='dataset_fname', type=str, default='../results//real_data/test_random_points_SR_m200cut.npy')
parser.add_argument('-use_1d', dest='use_1d', action='store', help='use_1d', type=int, default=0)
#parser.add_argument('-totiters_for_model', dest='totiters_for_model', action='store', help='totiters_for_model', type=int, default=25)#25)

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

N=1 #set downsampling factor used in step1 here
data = np.load(dataset_fname, allow_pickle= True)#.item() #for pol add .item() and comment out for T?
#with open('./stacks/clusters_m200cut_T_ILC_healpix_nw_apod_zeros_masked_inpaint_planckgrad_WFv3_COMMANDER_v2_org_Behzad_fullorient.pkl', 'rb') as f:
    # Load the contents of the file using pickle.load()
#    data = pickle.load(f)
#data = np.load(dataset_fname, allow_pickle= True)
param_dict = misc.get_param_dict(paramfile)
#cutouts specs 
dx = param_dict['dx'] #pixel resolution in arcmins
pol = True
if not pol:
    tqulen = 1
else:
    tqulen = 3
tqu_tit_arr = ['T', 'Q', 'U']
cutout_size_am = param_dict['cutout_size_am'] #arcmins
ny = nx = int(cutout_size_am / dx)
#x1, x2 = -cutout_size_am, cutout_size_am
x1, x2 = -cutout_size_am*N/2., cutout_size_am*N/2.

print('\tread/get necessary params')
param_dict = misc.get_param_dict(paramfile)

#params or supply a params file
mapparams = [nx, nx, dx]

###beam and noise levels
noiseval = param_dict['noiseval'] #uK-arcmin
if pol:
    noiseval = [noiseval, noiseval * np.sqrt(2.), noiseval * np.sqrt(2.)]

#CMB power spectrum
cls_file = '%s/%s' %(param_dict['data_folder'], param_dict['cls_file'])
el, cl = tools.get_cmb_cls(cls_file, pol = pol)
nl_dic = tools.get_nl_dic(noiseval, el, pol = pol)

##########################################
#read mock data + remove tsz estimate if necessary
try:
    add_cluster_tsz = param_dict['add_cluster_tsz']
except:
    add_cluster_tsz = False
try:
    add_cluster_ksz = param_dict['add_cluster_ksz']
except:
    add_cluster_ksz = False

data_stack_dic = {}
totsims = len(data['clusters']['cutouts_rotated'])
cutouts_rotated_arr=data['clusters']['cutouts_rotated']
grad_mag_arr=data['clusters']['grad_mag']
if (1):
    #print(data['clusters'].keys())
    grad_orient_arr=data['clusters']['grad_orien']
    #print(grad_orient_arr.shape)
    #print(grad_orient_arr[10])
    #imshow(cutouts_rotated_arr[10]);colorbar();show()
    plt.hist(grad_orient_arr, bins=90)
    plt.xlabel("Q median gradient orientation")
    plt.show()
    #for i in arange(10):
    #    imshow(cutouts_rotated_arr[i].squeeze());colorbar();show()
    #sys.exit()
###stack = tools.stack_rotated_tqu_cutouts(cutouts_rotated_arr, weights_for_cutouts = grad_mag_arr)
if (1): 
    #estimate and remove tSZ from rotated stack
    cutouts_rotated_arr_for_tsz_estimation = np.copy(cutouts_rotated_arr)
    tsz_estimate = tools.stack_rotated_tqu_cutouts(cutouts_rotated_arr_for_tsz_estimation, weights_for_cutouts = None, perform_random_rotation =True)

    #fit tsz model
    tsz_fit_model = foregrounds.fit_fot_tsz(tsz_estimate[0], dx)
    #imshow(tsz_fit_model,cmap=cmap);colorbar();show()
    #imshow(data['clusters']['stack'][0],cmap=cmap);colorbar();show()

    if (1):
        print('\n\t\t\tfitting for tsz\n\n')
        tsz_estimate[0] = np.copy(tsz_fit_model)

    cutouts_rotated_arr[:,0] -= tsz_estimate[0]
    data['clusters']['cutouts_rotated'] = cutouts_rotated_arr
    stack_after_tsz_removal = tools.stack_rotated_tqu_cutouts(cutouts_rotated_arr, weights_for_cutouts = grad_mag_arr)

    stack = np.copy(stack_after_tsz_removal)
    #data_stack_dic[simcntr]=stack
    data_stack_dic=stack
    imshow(data_stack_dic[0],cmap=cmap);colorbar();show()
else:
    data_stack_dic=data['clusters']['stack']
    imshow(data_stack_dic[0],cmap=cmap);colorbar();show()
    print(np.mean(data_stack_dic[0,19:21,19:21]))
#data_stack_dic = np.load('./data_stack_dic_fitted_tsz.npy',allow_pickle=True)
#data_stack_dic = np.load('./data_stack_dic_jk_test_fitted_tsz.npy',allow_pickle=True)
#data_stack_dic=data_stack_dic#[:,10:30,10:30]
        
##########################################
##########################################

#get and read random stack
#fd = '/'.join( dataset_fname.split('/')[:-1] )
#random_dataset_fname = glob.glob( '%s/randoms*' %(fd) )[0]
#random_data = np.load('../results/nx240_dx0.5/beam1.17/noise9.4/20amcutouts/m200mean/5474clusters/withgaussianfg_withclustertsz_withclusterksz/TQU/randoms_m200cut_U_ILC_healpix_nw_apod_zeros_masked_inpaint_v3.npy', allow_pickle= True).item() #for pol add .item() and comment out for T?
random_data = np.load('../results/nx240_dx0.5/beam1.17/noise9.4/20amcutouts/m200mean/5474clusters/withgaussianfg_withclustertsz_withclusterksz/TQU/randoms_m200cut_T_ILC_healpix_nw_apod_zeros_masked_inpaint_planckgrad_WFv3_COMMANDER.pkl', allow_pickle= True)#.item()#['randoms']
if (0):
    random_data = np.load('../results/nx240_dx0.5/beam1.17/noise9.4/20amcutouts/m200mean/5474clusters/withgaussianfg_withclustertsz_withclusterksz/TQU/randoms_m200cut_T_150GHz_healpix_nw_apod_zeros_masked_inpaint_v5_planckgrad.pkl', allow_pickle= True)#.item()#['randoms']
    random_stack_dic = random_data['randoms']['stack']
    random_stack = random_stack_dic[0]
    #imshow(data_stack_dic[0],cmap=cmap);colorbar();show()
    data_stack_dic -= random_stack
else:
    random_stack_dic = random_data['randoms']['stack']
    random_stack = random_stack_dic[0]
    print(random_stack.shape)
    data_stack_dic -= random_stack
if pol:
    cl_signal_arr=[cl[0], cl[1]/2., cl[1]/2.]
    cl_noise_arr=[nl_dic['T'], nl_dic['P'], nl_dic['P']]
else:
    cl_signal_arr=cl #cl[0] if using camb cl
    cl_noise_arr=[nl_dic['T']]
wiener_filter=flatsky.wiener_filter(mapparams, cl_signal=cl_signal_arr[0], cl_noise=cl_noise_arr[0])
#np.save('./WF.npy', wiener_filter)
test= np.fft.ifft2( np.fft.fft2(data_stack_dic[0]) * wiener_filter ).real

imshow(random_stack,cmap=cmap);colorbar();show()
#imshow(data_stack_dic[0],cmap=cmap);colorbar();show()
imshow(data_stack_dic[0,10:30,10:30],cmap=cmap);colorbar();show()
imshow(data_stack_dic[0,10:30,10:30],interpolation='gaussian',cmap=cmap);colorbar();show()
imshow(test[10:30,10:30], cmap=cmap);colorbar();show()
#imshow(test[10:30,10:30],cmap=cmap, vmin=-0.125, vmax=0.125);colorbar();show()
#imshow(test,cmap=cmap);colorbar();show()
#np.save('./stacks/stack_m200cut_T_150GHz_healpix_nw_apod_zeros_masked_inpaint_v4_planckgrad.npy', data_stack_dic)
sys.exit()
#pylab.savefig('./T_220_nw_apod_zeros_masked_inpaint_v3.png')
##########################################
##########################################
do_JK=0
if do_JK:
    # #get JK based covariance from cluster cuouts
    #dummysimcntr = 1
    ###dummysimcntr=0 #behzad added this line on 21/02/22 for a test with only 1 data sim, replace with line above later.
    cluster_cutouts_rotated_arr=data['clusters']['cutouts_rotated']- random_stack
    #if use_1d:
    #    cluster_cutouts_rotated_arr_1d = np.zeros((total_clusters, tqulen, nx))
    #    for i in range(total_clusters):
    #        for tqu in range(tqulen):
    #            cluster_cutouts_rotated_arr_1d[i, tqu] = np.mean(cluster_cutouts_rotated_arr[i, tqu], axis = 0)
    #    cluster_cutouts_rotated_arr = np.copy( cluster_cutouts_rotated_arr_1d )
    cluster_grad_mag_arr=data['clusters']['grad_mag']
    
    howmany_jk_samples = int(5474 * 0.9)
    # #try:
    # #    howmany_jk_samples = param_dict['howmany_jk_samples']
    # #except:
    # #    howmany_jk_samples = min_howmany_jk_samples
    # #if howmany_jk_samples<min_howmany_jk_samples:
    # #    howmany_jk_samples = min_howmany_jk_samples
    # #np.random.seed(100)
    # ##jk_cov=tools.get_jk_covariance(cluster_cutouts_rotated_arr, howmany_jk_samples, weights=cluster_grad_mag_arr, T_or_Q_or_U='all')
    jk_cov=tools.get_jk_covariance(cluster_cutouts_rotated_arr, howmany_jk_samples, weights=cluster_grad_mag_arr, only_T=True)
    if (1): print(jk_cov.shape); clf(); imshow(jk_cov, cmap=cmap); colorbar(); show(); sys.exit()
