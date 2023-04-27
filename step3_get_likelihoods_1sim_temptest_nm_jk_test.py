#!/usr/bin/env python
########################

########################
#load desired modules
import numpy as np, sys, os, scipy as sc, argparse, glob
sys_path_folder='../python/'
sys.path.append(sys_path_folder)

import flatsky, tools, lensing, foregrounds, misc

from tqdm import tqdm

from pylab import *
cmap = cm.RdYlBu_r

import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
print('\n')
########################

########################
parser = argparse.ArgumentParser(description='')
parser.add_argument('-dataset_fname', dest='dataset_fname', action='store', help='dataset_fname', type=str, default='../results/nx240_dx0.5/beam1.17/noise9.4/20amcutouts/m200mean/5474clusters/withgaussianfg_withclustertsz_withclusterksz/TQU/clusters_TQU.npy')
parser.add_argument('-use_1d', dest='use_1d', action='store', help='use_1d', type=int, default=0)
parser.add_argument('-totiters_for_model', dest='totiters_for_model', action='store', help='totiters_for_model', type=int, default=25)#25)
parser.add_argument('-paramfile', dest='paramfile', action='store', help='paramfile', type=str, default='params_model.ini')

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
data = np.load(dataset_fname, allow_pickle= True).item()
param_dict = misc.get_param_dict(paramfile)
#cutouts specs 
dx = param_dict['dx'] #pixel resolution in arcmins
pol = param_dict['pol']
if not pol:
    tqulen = 1
else:
    tqulen = 3
tqu_tit_arr = ['T', 'Q', 'U']
cutout_size_am = param_dict['cutout_size_am'] #arcmins
ny = nx = int(cutout_size_am / dx)
x1, x2 = -cutout_size_am/2., cutout_size_am/2.
#sim stuffs
total_clusters = param_dict['total_clusters']

##########################################
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

if not add_cluster_tsz:
    data_stack_dic = data['clusters']['stack']
else: #handle tsz
    #stack rotated cutouts + apply gradient magnitude weights
    data_stack_dic = {}
    totsims = len(data['clusters']['cutouts_rotated'])
    for simcntr in range( totsims ):
        cutouts_rotated_arr=data['clusters']['cutouts_rotated'][simcntr]
        grad_mag_arr=data['clusters']['grad_mag'][simcntr]

        ###stack = tools.stack_rotated_tqu_cutouts(cutouts_rotated_arr, weights_for_cutouts = grad_mag_arr)

        #estimate and remove tSZ from rotated stack
        cutouts_rotated_arr_for_tsz_estimation = np.copy(cutouts_rotated_arr)
        tsz_estimate = tools.stack_rotated_tqu_cutouts(cutouts_rotated_arr_for_tsz_estimation, weights_for_cutouts = None, perform_random_rotation = True)
        
        #fit tsz model
        tsz_fit_model = foregrounds.fit_fot_tsz(tsz_estimate[0], dx)
        if (0):
            print('\n\t\t\tfitting for tsz\n\n')
            tsz_estimate[0] = np.copy(tsz_fit_model)

        cutouts_rotated_arr[:,0] -= tsz_estimate[0]
        data['clusters']['cutouts_rotated'][simcntr] = cutouts_rotated_arr
        stack_after_tsz_removal = tools.stack_rotated_tqu_cutouts(cutouts_rotated_arr, weights_for_cutouts = grad_mag_arr)

        if (0):
            subplot(131); imshow(stack[0], cmap=cmap, extent = [x1, x2, x1, x2]); colorbar(); 
            subplot(132); imshow(tsz_estimate[0], cmap=cmap, extent = [x1, x2, x1, x2]); colorbar();
            subplot(133); imshow(stack_after_tsz_removal[0], cmap=cmap, extent = [x1, x2, x1, x2]); colorbar(); show(); sys.exit()

        stack = np.copy(stack_after_tsz_removal)

        data_stack_dic[simcntr]=stack
        
##########################################
##########################################

#get and read random stack
#fd = '/'.join( dataset_fname.split('/')[:-1] )
#random_dataset_fname = glob.glob( '%s/randoms*' %(fd) )[0]
random_data = np.load('../results/nx240_dx0.5/beam1.17/noise9.4/20amcutouts/m200mean/5474clusters/withgaussianfg_withclustertsz_withclusterksz/TQU/randoms_TQU.npy', allow_pickle= True).item()['randoms']
random_stack_dic = random_data['stack']
random_stack = random_stack_dic[0]

#subtract background from data stack
for keyname in data_stack_dic:
    if (0):
        tmp_stack = data_stack_dic[keyname]
        tmp_stack_bg_sub = data_stack_dic[keyname] - random_stack
        sbpl=1
        for tqu in range(len(data_stack_dic[keyname])):
            subplot(tqulen, 3, sbpl); imshow(tmp_stack[tqu], cmap=cmap, extent = [x1, x2, x1, x2]); colorbar(); sbpl+=1
            subplot(tqulen, 3, sbpl); imshow(random_stack[tqu], cmap=cmap, extent = [x1, x2, x1, x2]); colorbar(); sbpl+=1
            subplot(tqulen, 3, sbpl); imshow(tmp_stack_bg_sub[tqu], cmap=cmap, extent = [x1, x2, x1, x2], vmin = -2.5, vmax = 2.5); colorbar(); sbpl+=1
            title('%s' %(tqu_tit_arr[tqu]))
        show(); sys.exit()
    data_stack_dic[keyname] -= random_stack
    ##data_stack_dic[keyname][np.isnan(data_stack_dic[keyname])] = 0.
    ##if np.sum(data_stack_dic[keyname][0]) == 0.: tqulen = 1
##########################################
##########################################
#total_clusters=6300

#get models
# model_fd = '%s/models/' %(fd)
# if not os.path.exists(model_fd):
#     tmp_fd = fd.replace('_withclusterksz','').replace('_withclustertsz','')
model_fd = '../results/nx240_dx0.5/beam1.17/noise9.4/20amcutouts/m200mean/5474clusters/withgaussianfg_withclustertsz_withclusterksz/TQU/models/'
#model_flist = sorted( glob.glob('%s/*_%ssims_*.npy' %(model_fd, totiters_for_model)) )
model_flist = sorted( glob.glob('%s/*.npy' %(model_fd)) )

#def get_model_keyname(model_fname):
#    model_keyname_tmp = '_'.join(model_fname.split('_')[-3:]).replace('mass', '').replace('z','').replace('.npy','').split('_')
#    model_mass, model_z = float(model_keyname_tmp[0]), float(model_keyname_tmp[1])
#    model_keyname = ( round(model_mass, 3), round(model_z, 3) )
#    return model_keyname   

def get_model_keyname(model_fname):
    model_keyname_tmp = '_'.join(model_fname.split('_')[-3:]).replace('mass', '').replace('z','').replace('.npy','').split('_') #model_fname.split('_')[-3:] replace -3 by -2 depending on file name
    model_mass, model_z = float(model_keyname_tmp[0]), float(model_keyname_tmp[1])
    model_keyname = ( round(model_mass, 3), round(model_z, 3) )
    return model_keyname   

#get gradient orientation first for each cluster in each (M,z) for each sim.
#next for every cluster we will get the median grad orientation across all (M,z) for each sim.
#this ensures that we rotate a given cluster lensed by all (M,z) by the same angle. Otherwise, there likelihoods can be shaky.
tmp_model_orien_dic = {}
for model_fname in model_flist:
    model_data = np.load(model_fname, allow_pickle=True).item()
    for simkeyname in model_data:
        if simkeyname not in tmp_model_orien_dic:
            tmp_model_orien_dic[simkeyname] = []
        cutouts_rotated_arr, grad_mag_arr, grad_orien_arr = model_data[simkeyname]['cutouts']
        tmp_model_orien_dic[simkeyname].append(grad_orien_arr)

#final gradient orientation is obtained in this step
model_orien_dic = {}
for simkeyname in tmp_model_orien_dic:
    model_orien_dic[simkeyname] = []
    grad_orien_arr = np.asarray( tmp_model_orien_dic[simkeyname] ) #vector with dimensions total_models x total_clusters x tqulen
    model_orien_dic[simkeyname] = np.mean(grad_orien_arr, axis = 0) #vector with dimensions total_clusters x tqulen

#models are computed here by rotating each cluster along the orientations estimated above
model_dic = {}
for model_fname in model_flist:
    model_data = np.load(model_fname, allow_pickle=True).item()
    '''
    #model_arr = np.asarray( list(model_data.values()) )
    model_arr = []
    for simkeyname in model_data:
        model_arr.append(model_data[simkeyname]['stack'])
    '''
    model_arr = []
    for simkeyname in model_data:
        cutouts_arr, grad_mag_arr, grad_orien_arr = model_data[simkeyname]['cutouts']
        ##grad_orien_arr_avg = model_orien_dic[simkeyname]

        ##cutouts_rotated_arr = tools.get_rotated_tqu_cutouts_simple(cutouts_arr, grad_orien_arr_avg, total_clusters, tqulen)
        cutouts_rotated_arr = tools.get_rotated_tqu_cutouts_simple(cutouts_arr, grad_orien_arr, total_clusters, tqulen)
        stack=tools.stack_rotated_tqu_cutouts(cutouts_rotated_arr, weights_for_cutouts = grad_mag_arr)
        model_arr.append( stack )

    model = np.mean(model_arr, axis = 0)
    ##model= model_arr #this is for n to n model/data comparison in "1sim" case, this is added and line above is commented out
    model_keyname = get_model_keyname(model_fname)
    model_dic[model_keyname] = model
    ###if model_keyname==(2.3,0.5):
        ###print('printing model')
        ###print(model[21])

##########################################
##########################################

#subtract M=0 from all
bg_model_keyname = (0., 0.5)#0.7)
for model_keyname in model_dic:
    if model_keyname == bg_model_keyname: continue
    model_dic[model_keyname] -= np.asarray(model_dic[bg_model_keyname])
    #model_dic[model_keyname] -= random_stack #model-random works better than M0? why?     
    if (0):#model_keyname[0]>0.8 and model_keyname[0]<1.2:#(1):
        for tqu in range(tqulen):
            print(model_dic[model_keyname][tqu])
            vmin, vmax = -2., 2.
            vmin, vmax = None, None
            subplot(1, tqulen, tqu+1); imshow(model_dic[model_keyname][tqu], cmap=cmap, extent = [x1, x2, x1, x2], vmin = vmin, vmax = vmax); 
            colorbar()
            title('(%s, %s): %s' %(model_keyname[0], model_keyname[1], tqu_tit_arr[tqu]))
            axhline(lw = 0.5); axvline(lw = 0.5)
        show(); sys.exit()
model_dic[bg_model_keyname] -= np.asarray(model_dic[bg_model_keyname])
#sys.exit()

##########################################
##########################################

if (1):
    #params or supply a params file
    noiseval = param_dict['noiseval'] #uK-arcmin
    beamval = param_dict['beamval'] #arcmins
    #foregrounds
    try:
        fg_gaussian = param_dict['fg_gaussian'] #Gaussian realisation of all foregrounds
    except:
        fg_gaussian = False

    #ILC
    try:
        ilc_file = param_dict['ilc_file'] #ILC residuals
        which_ilc = param_dict['which_ilc']
    except:
        ilc_file = None
        which_ilc = None

    #cluster info
    cluster_mass = 2.3e14#param_dict['cluster_mass']
    cluster_z = 0.5#param_dict['cluster_z']

    #cluster mass definitions
    delta=param_dict['delta']
    rho_def=param_dict['rho_def']
    profile_name=param_dict['profile_name']

########################
########################

#total_clusters=5474
#get JK based covariance from cluster cuouts
#dummysimcntr = 1
###dummysimcntr=0 #behzad added this line on 21/02/22 for a test with only 1 data sim, replace with line above later.
jk_cov=[]
howmany_jk_samples = int(total_clusters * 0.9)
for simcntr in sorted(data_stack_dic):
    cluster_cutouts_rotated_arr=data['clusters']['cutouts_rotated'][simcntr]-random_stack
    cluster_grad_mag_arr=data['clusters']['grad_mag'][simcntr]
    jk_cov_temparr=tools.get_jk_covariance(cluster_cutouts_rotated_arr, howmany_jk_samples, weights=cluster_grad_mag_arr, only_T=False)
    jk_cov.append(jk_cov_temparr)
#    print(np.asarray(jk_cov).shape)
########################
########################
#get likelihoods
def get_plname():
    plfolder = '%s/plots/' %(dataset_fd)
    if not os.path.exists(plfolder): os.system('mkdir -p %s' %(plfolder))
    plname = '%s/%sclusters_beam%s_noise%s' %(plfolder, total_clusters, beamval, noiseval)
    if tqulen == 3:
        plname = '%s_TQU' %(plname)
    else:
        plname = '%s_T' %(plname)

    if fg_gaussian:
        titstr = 'FG added'
        plname = '%s_withfg' %(plname)
    else:
        titstr = 'No FG'
        plname = '%s_nofg' %(plname)

    if ilc_file is not None:
        titstr = 'ILC: %s' %(which_ilc)
        plname = '%s_ilc_%s' %(plname, which_ilc)

    if add_cluster_tsz:
        plname = '%s_withclustertsz' %(plname)
        titstr = '%s + cluster tSZ' %(titstr)
    if add_cluster_ksz:
        plname = '%s_withclusterksz' %(plname)
        titstr = '%s + cluster kSZ' %(titstr)

    plname = plname.replace('plots//', 'plots/')
    opfname = '%s.npy' %(plname.replace('/plots/', '/results_'))
    plname = '%s.png' %(plname)
    
    return plname, opfname, titstr

res_dic = {}
res_dic['likelihood'] = {}
testing = 0
tr, tc = tqulen, 1
tqudic = {0: 'T', 1: 'Q', 2: 'U'}
jk_cov_temp=[]

if (1):
    np.save('./data_stack_dic_jk_test.npy',data_stack_dic)
    np.save('./model_dic_jk_test.npy',model_dic)
    np.save('./jk_cov_jk_test.npy',jk_cov)

for tqu in range(tqulen):
    res_dic['likelihood'][tqudic[tqu]] = {}
    master_loglarr = []
    ax = subplot(tr, tc, tqu+1)
    for simcntr in sorted(data_stack_dic):
        if tqu==0:
            jk_cov_temp=jk_cov[simcntr][0:400,0:400]
        elif tqu==1:
            jk_cov_temp=jk_cov[simcntr][400:800,400:800]
        elif tqu==2:
            jk_cov_temp=jk_cov[simcntr][800:1200,800:1200]
        loglarr = []
        massarr = []
        if use_1d:
            data_vec = np.mean(data_stack_dic[simcntr][tqu], axis = 0)
        else:
            data_vec = data_stack_dic[simcntr][tqu].flatten()
        if testing:colorarr = [cm.jet(int(d)) for d in np.linspace(0, 255, len(model_dic))]
        for modelcntr, model_keyname in enumerate( sorted( model_dic ) ):
            if use_1d:
                model_vec = np.mean(model_dic[model_keyname][tqu], axis = 0)
                if testing: plot(model_vec, color = colorarr[modelcntr])
            else:
                model_vec = model_dic[model_keyname][tqu].flatten()
                ###model_vec = model_dic[model_keyname][simcntr][tqu].flatten() ## this is for n to n model/data comparison in "1sim" case
            loglval = tools.get_lnlikelihood(data_vec, model_vec, jk_cov_temp)
            loglarr.append( loglval )
            massarr.append( model_keyname[0] )
        if testing: show(); sys.exit()
        massarr = np.asarray( massarr )
        massarr_mod, larr, recov_mass, snr = tools.lnlike_to_like(massarr, loglarr)
        #logl_dic[simcntr] = [massarr, loglarr, larr]
        if (recov_mass>0.1 and recov_mass<3.9):
            master_loglarr.append( loglarr )
            plot(massarr_mod, larr, lw = 0.5, color = 'orangered');#, label = simcntr, lw = 0.5);
            res_dic['likelihood'][tqudic[tqu]][simcntr] = [massarr, loglarr, massarr_mod, larr, recov_mass, snr]

    combined_loglarr = np.sum(master_loglarr, axis = 0)
    massarr_mod, combined_larr, combined_recov_mass, combined_snr = tools.lnlike_to_like(massarr, combined_loglarr)
    combined_mean_mass, combined_mean_mass_low_err, combined_mean_mass_high_err = tools.get_width_from_sampling(massarr_mod, combined_larr)
    combined_mean_mass_err = (combined_mean_mass_low_err + combined_mean_mass_high_err)/2.
    plot(massarr_mod, combined_larr, lw = 2.5, color = 'black')#, label = r'Combined: %.2f $\pm$ %.2f' %(combined_mean_mass, combined_mean_mass_err));
    axvline(cluster_mass/1e14, ls = '--', lw = 2.5, color = 'grey')
    dataset_fd = '/'.join(dataset_fname.split('/')[:-1])
    plname, opfname, titstr = get_plname()
    res_dic['likelihood'][tqudic[tqu]][-1] = [massarr, combined_loglarr, massarr_mod, combined_larr, combined_recov_mass, combined_snr]
    
    if tqu == 0:
        if tqulen == 1:
            legend(loc = 4, ncol = 4, fontsize = 8)
        else:
            legend(loc = 4, ncol = 8, fontsize = 6)
    if tqu+1 == tqulen:
        mdefstr = 'M$_{%s%s}$' %(param_dict['delta'], param_dict['rho_def'])
        xlabel(r'%s [$10^{14}$M$_{\odot}$]' %(mdefstr), fontsize = 14)
    ylabel(r'Normalised $\mathcal{L}$', fontsize = 14)
    title(r'%s clusters (SNR = %.2f); $\Delta_{\rm T} = %s \mu{\rm K-arcmin}$; %s' %(total_clusters, combined_snr/np.sqrt(14), noiseval, titstr), fontsize = 10)
res_dic['param_dict'] = param_dict
np.save(opfname, res_dic)
savefig(plname)
show();
print(plname)

plot_qu=False#True
master_loglarr=[]
#massarr = []
if plot_qu:
    jk_cov_QU=jk_cov[100:300,100:300]
    for simcntr in sorted(data_stack_dic):
        loglarr = []
        massarr = []
        if use_1d:
            data_vec_Q = np.mean(data_stack_dic[simcntr][1], axis = 0)
            data_vec_U = np.mean(data_stack_dic[simcntr][2], axis = 0)
        else:
            data_vec_Q = data_stack_dic[simcntr][1].flatten()
            data_vec_U = data_stack_dic[simcntr][2].flatten()
        data_vec_QU= np.concatenate([data_vec_Q, data_vec_U])
        for modelcntr, model_keyname in enumerate( sorted( model_dic ) ):
            if use_1d:
                model_vec_Q = np.mean(model_dic[model_keyname][1], axis = 0)
                model_vec_U = np.mean(model_dic[model_keyname][2], axis = 0)
            else:
                model_vec_Q = model_dic[model_keyname][1].flatten()#*0.85#behzad- temp test *0.85
                model_vec_U = model_dic[model_keyname][2].flatten()#*0.85#behzad- temp test *0.85
            model_vec_QU=np.concatenate([model_vec_Q, model_vec_U])
            loglval = tools.get_lnlikelihood(data_vec_QU, model_vec_QU, jk_cov_QU) #behzad jk_cov->jk_cov_temp
            loglarr.append( loglval )
            massarr.append( model_keyname[0] )
        if testing: show(); sys.exit()
        massarr = np.asarray( massarr )
        massarr_mod, larr, recov_mass, snr = tools.lnlike_to_like(massarr, loglarr)
        #logl_dic[simcntr] = [massarr, loglarr, larr]
        master_loglarr.append( loglarr )
        plot(massarr_mod, larr, label = simcntr, lw = 0.5);
    #np.savetxt("../results_tsz_test//nx120_dx1/beam1.2/noise5/10amcutouts/m200mean_2.3e+14//6300clusters/nogaussianfg//TQU//likelihood_0p1bins_25_sims_0to5e14_QU.txt", np.c_[massarr,np.asarray(master_loglarr).transpose()])
    combined_loglarr = np.sum(master_loglarr, axis = 0)
    massarr_mod, combined_larr, combined_recov_mass, combined_snr = tools.lnlike_to_like(massarr, combined_loglarr)
    combined_mean_mass, combined_mean_mass_low_err, combined_mean_mass_high_err = tools.get_width_from_sampling(massarr_mod, combined_larr)
    combined_mean_mass_err = (combined_mean_mass_low_err + combined_mean_mass_high_err)/2.
    plot(massarr_mod, combined_larr, lw = 1.5, color = 'black', label = r'Combined: %.2f $\pm$ %.2f' %(combined_mean_mass, combined_mean_mass_err));
    axvline(2.3e14/1e14, ls = '-.', lw = 2.)
 #   dataset_fd = '/'.join(dataset_fname.split('/')[:-1])
 #   plname, titstr = get_plname()
    mdefstr = 'M$_{%s%s}$' %(param_dict['delta'], param_dict['rho_def'])
    xlabel(r'%s [$10^{14}$M$_{\odot}$]' %(mdefstr), fontsize = 14)
    ylabel(r'Normalised $\mathcal{L}$', fontsize = 14)
    title(r'%s clusters (SNR = %.2f); $\Delta_{\rm T} = %s \mu{\rm K-arcmin}$; %s' %(total_clusters, combined_snr/np.sqrt(25), noiseval, titstr), fontsize = 10)
#savefig(plname)
show();
#print(plname)
sys.exit()

