import numpy as np, os

#################################################################################
#################################################################################
#################################################################################

def is_seq(o):
    """
    determine if the passed variable is an array.
    """
    return hasattr(o, '__len__')

################################################################################################################
################################################################################################################

def get_param_dict(paramfile):
    params, paramvals = np.genfromtxt(
        paramfile, delimiter = '=', unpack = True, autostrip = True, dtype='unicode')
    param_dict = {}
    for p,pval in zip(params,paramvals):
        if pval in ['T', 'True']:
            pval = True
        elif pval in ['F', 'False']:
            pval = False
        elif pval == 'None':
            pval = None
        else:
            try:
                pval = float(pval)
                if int(pval) == float(pval):
                    pval = int(pval)
            except:
                pass
        # replace unallowed characters in paramname
        p = p.replace('(','').replace(')','')
        param_dict[p] = pval
    return param_dict

################################################################################################################
################################################################################################################

def get_op_folder(results_folder, nx, dx, beamval, noiseval, cutout_size_am, mdef = None, ilc_file = None, which_ilc = None, nclustersorrandoms = None, pol = False, models = False, fg_str = None):
    if ilc_file is None:
        if is_seq(noiseval):
            tmpnoiseval = noiseval[0]
        else:
            tmpnoiseval = noiseval
        tmpnoiseval = 'noise%s' %(tmpnoiseval)
    else:
        tmpnoiseval = 'ilc/%s/%s/' %(ilc_file.split('/')[-1].replace('.npy',''), which_ilc)
    if mdef is None:
        mdeffd = ''
    else:
        mdeffd = '%s/' %(mdef)
    op_folder = '%s/nx%s_dx%s/beam%s/%s/%samcutouts/%s' %(results_folder, nx, dx, beamval, tmpnoiseval, cutout_size_am, mdeffd)
    if nclustersorrandoms is not None:
        op_folder = '%s/%sclusters' %(op_folder, nclustersorrandoms)
    if fg_str is not None:
        op_folder = '%s/%s/' %(op_folder, fg_str)
    if pol:
        op_folder = '%s/TQU/' %(op_folder)
    else:
        op_folder = '%s/T/' %(op_folder)
    if models:
        op_folder = '%s/models/' %(op_folder)
    if not os.path.exists(op_folder): os.system('mkdir -p %s' %(op_folder))
    return op_folder

################################################################################################################
################################################################################################################

def get_op_fname(op_folder, sim_type, nclustersorrandoms, total_sims, start = -1, end = -1, extrastr = '', random_seed_for_sims = -1):
    if start != -1 and end != -1:
        fname = '%s/%s_%sobjects_%ssims%sto%s%s_rsval%s.npy' %(op_folder, sim_type, nclustersorrandoms, total_sims, start, end, extrastr, random_seed_for_sims)
    else:
        fname = '%s/%s_%sobjects_%ssims%s_rsval%s.npy' %(op_folder, sim_type, nclustersorrandoms, total_sims, extrastr, random_seed_for_sims)
    return fname

################################################################################################################
################################################################################################################

################################################################################################################

def get_nl(noiseval, el, beamval=None, use_beam_window=False, uk_to_K=False, elknee_t=-1, alpha_knee=0):

    """
    get noise power spectrum (supports both white and 1/f atmospheric noise)
    """

    if uk_to_K: noiseval=noiseval/1e6

    if use_beam_window:
        bl=get_bl(beamval, el)

    delta_T_radians=noiseval * np.radians(1./60.)
    nl=np.tile(delta_T_radians**2., int(max(el)) + 1 )

    nl=np.asarray( [nl[int(l)] for l in el] )

    if use_beam_window: nl=nl/bl**2.

    if elknee_t != -1.:
        nl=np.copy(nl) * (1. + (elknee_t * 1./el)**alpha_knee )

    return nl
