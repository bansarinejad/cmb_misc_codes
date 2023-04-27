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
mapparams = [nx, nx, dx]
##########################################
##########################################

#get and read random stack
random_data = np.load('../results/nx240_dx0.5/beam1.17/noise9.4/20amcutouts/m200mean/5474clusters/withgaussianfg_withclustertsz_withclusterksz/TQU/randoms_TQU.npy', allow_pickle= True).item()['randoms']
random_stack_dic = random_data['stack']
random_stack = random_stack_dic.squeeze()

##########################################
##########################################
if (1):
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

        #data=np.load('./data_fitted_tsz.npy',allow_pickle=True).item()
        data=np.load('./data_25mean_tsz.npy',allow_pickle=True).item()
        data['clusters']['cutouts_rotated']=data['clusters']['cutouts_rotated'][:,:,10:30,10:30]

        #data_stack_dic = np.load('./data_stack_dic_fitted_tsz.npy',allow_pickle=True)
        data_stack_dic = np.load('./data_stack_dic_25mean_tsz.npy',allow_pickle=True)
        data_stack_dic=data_stack_dic[:,10:30,10:30]

        #model_dic=np.load('./model_dic_fitted_tsz.npy',allow_pickle=True).item()
        model_dic=np.load('./model_dic_25mean_tsz.npy',allow_pickle=True).item()
        model_dic={(round(x, 1), 0.5): model_dic[(round(x, 1), 0.5)][:,10:30, 10:30] for x in np.arange(0.0, 4.1, 0.1)}
        model_dic_test=np.load('/data/gpfs/projects/punim1720/cmb_cluster_lensing/scripts/models_TQU_desnz/model_randomseed67_mass2.300_z0.500.npy',allow_pickle=True).item()
        #print(model_dic_test[(2)].keys())
        #imshow(model_dic_test[(2)]['stack'][1],cmap=cmap, vmin=-0.125, vmax=0.125);colorbar();show()
        #imshow(model_dic_test[(2)]['stack'][2],cmap=cmap, vmin=-0.125, vmax=0.125);colorbar();show()
        #sys.exit()
        if (1):
            ###beam and noise levels
            pol=True
            noiseval = param_dict['noiseval'] #uK-arcmin
            if pol:
                noiseval = [noiseval, noiseval * np.sqrt(2.), noiseval * np.sqrt(2.)]

            #CMB power spectrum
            cls_file = '%s/%s' %(param_dict['data_folder'], param_dict['cls_file'])
            el, cl = tools.get_cmb_cls(cls_file, pol = pol)
            nl_dic = tools.get_nl_dic(noiseval, el, pol = pol)
            if pol:
                cl_signal_arr=[cl[0], cl[1]/2., cl[1]/2.]
                cl_noise_arr=[nl_dic['T'], nl_dic['P'], nl_dic['P']]
            mapparams = [nx//2, nx//2, dx]
            wiener_filter=flatsky.wiener_filter(mapparams, cl_signal=cl_signal_arr[0], cl_noise=cl_noise_arr[0])
            T_wf= np.fft.ifft2( np.fft.fft2(data_stack_dic[0]) * wiener_filter ).real
            Q_wf= np.fft.ifft2( np.fft.fft2(data_stack_dic[1]) * wiener_filter ).real
            U_wf= np.fft.ifft2( np.fft.fft2(data_stack_dic[2]) * wiener_filter ).real
            #imshow(T_wf,cmap=cmap);colorbar();show()
            #imshow(Q_wf,cmap=cmap);colorbar();show()
            #imshow(U_wf,cmap=cmap);colorbar();show()
            imshow(data_stack_dic[0],cmap=cmap);colorbar();show() #,interpolation='gaussian'
            imshow(data_stack_dic[1],cmap=cmap);colorbar();show()
            imshow(data_stack_dic[2],cmap=cmap);colorbar();show()
            imshow(model_dic[(2.3,0.5)][0],cmap=cmap,vmin=-1.25, vmax=1.25);colorbar();show() 
            imshow(model_dic[(2.3,0.5)][1],cmap=cmap, vmin=-0.125, vmax=0.125);colorbar();show()
            imshow(model_dic[(2.3,0.5)][2],cmap=cmap, vmin=-0.125, vmax=0.125);colorbar();show()
            #imshow(model_dic[(2.3,0.5)][2]/model_dic[(2.3,0.5)][1],cmap=cmap);colorbar();show()
            #imshow(model_dic[(2.3,0.5)][2]/model_dic[(1.8,0.5)][2],cmap=cmap);colorbar();show()
            #imshow((model_dic[(2.3,0.5)][2]/model_dic[(2.3,0.5)][1])/(model_dic[(2.3,0.5)][2]/model_dic[(1.8,0.5)][2]),cmap=cmap);colorbar();show()
            #sys.exit()

    ########################
    ########################

    cluster_cutouts_rotated_arr=data['clusters']['cutouts_rotated'] - random_stack[:,10:30, 10:30]
    cluster_grad_mag_arr=data['clusters']['grad_mag']

    howmany_jk_samples = int(total_clusters * 0.9)

    jk_cov=tools.get_jk_covariance(cluster_cutouts_rotated_arr, howmany_jk_samples, weights=cluster_grad_mag_arr, only_T=False)
    if (0): print(jk_cov.shape); clf(); imshow(jk_cov, cmap=cmap); colorbar(); show(); sys.exit()

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

if (0):
    data=np.load('./data.npy',allow_pickle=True).item()
    data['clusters']['cutouts_rotated']=data['clusters']['cutouts_rotated'][:,:,10:30,10:30]

    data_stack_dic = np.load('./data_stack_dic.npy',allow_pickle=True)
    data_stack_dic=data_stack_dic[:,10:30,10:30]

    model_dic=np.load('./model_dic.npy',allow_pickle=True).item()
    model_dic={(round(x, 1), 0.5): model_dic[(round(x, 1), 0.5)][:,10:30, 10:30] for x in np.arange(0.0, 4.1, 0.1)}

    jk_cov=np.load('./jk_cov.npy',allow_pickle=True)
    #print(jk_cov.shape)
    if (0):
        imshow(data_stack_dic[0],cmap=cmap);colorbar();show()    #,vmin=-3.0,vmax=3.0
        imshow(data_stack_dic[1],cmap=cmap);colorbar();show()
        imshow(data_stack_dic[2],cmap=cmap);colorbar();show()
        imshow(model_dic[(2.3,0.5)][0],cmap=cmap);colorbar();show() 
        imshow(model_dic[(2.3,0.5)][1],cmap=cmap);colorbar();show()
        imshow(model_dic[(2.3,0.5)][2],cmap=cmap);colorbar();show()

for tqu in range(tqulen):
    print('tqu=', tqu)
    if tqu==0:
        jk_cov_temp=jk_cov[0:400,0:400]#[600:1000,600:1000]#[0:1600,0:1600]#[0:400,0:400]
    elif tqu==1:
        jk_cov_temp=jk_cov[400:800,400:800]#[2200:2600,2200:2600]#[1600:3200,1600:3200]#[400:800,400:800]
    elif tqu==2:
        jk_cov_temp=jk_cov[800:1200,800:1200]#[3800:4200,3800:4200]#[3200:4800,3200:4800]#[800:1200,800:1200]
    #imshow(jk_cov_temp,cmap=cmap);colorbar();show()
    res_dic['likelihood'][tqudic[tqu]] = {}
    master_loglarr = []
    ax = subplot(tr, tc, tqu+1)
    #for simcntr in sorted(data_stack_dic):
    loglarr = []
    massarr = []
    if use_1d:
        data_vec = np.mean(data_stack_dic[tqu], axis = 0)
    else:
        data_vec = data_stack_dic[tqu].flatten()
    if testing:colorarr = [cm.jet(int(d)) for d in np.linspace(0, 255, len(model_dic))]
    for modelcntr, model_keyname in tqdm( enumerate( sorted( model_dic ) ) ):
        if use_1d:
            model_vec = np.mean(model_dic[model_keyname][tqu], axis = 0)
            if testing: plot(model_vec, color = colorarr[modelcntr])
        else:
            model_vec = model_dic[model_keyname][tqu].flatten()
            ###model_vec = model_dic[model_keyname][simcntr][tqu].flatten() ## this is for n to n model/data comparison in "1sim" case
        loglval = tools.get_lnlikelihood(data_vec, model_vec, jk_cov_temp)
        loglarr.append( loglval )
        massarr.append( model_keyname[0] )
    loglarr = np.asarray( loglarr )
    #print(loglarr.shape)
    #print(type(loglarr))
    if testing: show(); sys.exit()
    massarr = np.asarray( massarr )
    print('test1')
    massarr_mod, larr, recov_mass, snr = tools.lnlike_to_like(massarr, loglarr)
    #logl_dic[simcntr] = [massarr, loglarr, larr]
    if (recov_mass>=0.0 and recov_mass<4):
        master_loglarr.append( loglarr )
        plot(massarr_mod, larr, lw = 0.5);
        res_dic['likelihood'][tqudic[tqu]] = [massarr, loglarr, massarr_mod, larr, recov_mass, snr]

    #combined_loglarr = np.sum(master_loglarr, axis = 0)
    #massarr_mod, combined_larr, combined_recov_mass, combined_snr = tools.lnlike_to_like(massarr, combined_loglarr)
    #combined_mean_mass, combined_mean_mass_low_err, combined_mean_mass_high_err = tools.get_width_from_sampling(massarr_mod, combined_larr)
    #combined_mean_mass_err = (combined_mean_mass_low_err + combined_mean_mass_high_err)/2.
    #plot(massarr_mod, combined_larr, lw = 2.5, color = 'black')#, label = r'Combined: %.2f $\pm$ %.2f' %(combined_mean_mass, combined_mean_mass_err));
    #axvline(cluster_mass/1e14, ls = '--', lw = 2.5, color = 'grey')
    #dataset_fd = '/'.join(dataset_fname.split('/')[:-1])
    #plname, opfname, titstr = get_plname()
    #res_dic['likelihood'][tqudic[tqu]][-1] = [massarr, combined_loglarr, massarr_mod, combined_larr, combined_recov_mass, combined_snr]
    
    if tqu == 0:
        if tqulen == 1:
            legend(loc = 4, ncol = 4, fontsize = 8)
        else:
            legend(loc = 4, ncol = 8, fontsize = 6)
    if tqu+1 == tqulen:
        mdefstr = 'M$_{%s%s}$' %(param_dict['delta'], param_dict['rho_def'])
        xlabel(r'%s [$10^{14}$M$_{\odot}$]' %(mdefstr), fontsize = 14)
    ylabel(r'Normalised $\mathcal{L}$', fontsize = 14)
    #title(r'%s clusters (SNR = %.2f); $\Delta_{\rm T} = %s \mu{\rm K-arcmin}$; %s' %(total_clusters, combined_snr/np.sqrt(14), noiseval, titstr), fontsize = 10)
res_dic['param_dict'] = param_dict
#np.save(opfname, res_dic)
#savefig(plname)
show();
#print(plname)

plot_qu=False#True
master_loglarr=[]
#massarr = []
if plot_qu:
    jk_cov_QU=jk_cov[100:300,100:300]
    #for simcntr in sorted(data_stack_dic):
    loglarr = []
    massarr = []
    if use_1d:
        data_vec_Q = np.mean(data_stack_dic[1], axis = 0)
        data_vec_U = np.mean(data_stack_dic[2], axis = 0)
    else:
        data_vec_Q = data_stack_dic[1].flatten()
        data_vec_U = data_stack_dic[2].flatten()
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
    plot(massarr_mod, larr, lw = 0.5);
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

