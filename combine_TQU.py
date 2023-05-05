import numpy as np

#print sims data structure that is to be mimiced
print_old=False
if print_old:
	old_data=np.load('clusters_5474sims0to25-1.npy', allow_pickle= True).item()
	print(old_data.keys())
	print(old_data['clusters'].keys())
	print(np.asarray(old_data['clusters']['cutouts_rotated'][0]).shape)
	print(np.asarray(old_data['clusters']['grad_mag'][0]).shape)
	print(np.asarray(old_data['clusters']['stack'][0]).shape)

	old_rands=np.load('randoms_54740sims0to1-1.npy', allow_pickle= True).item()
	print(old_rands.keys())
	print(old_rands['randoms'].keys())
	print(np.asarray(old_rands['randoms']['stack'][0]).shape)

#import new T data and rands and removed un-needed keys
mod_T=True#False
if mod_T:
	T_data=np.load('clusters_m200cut_T_ILC_healpix_nw_apod_zeros_masked_inpaint_planckgrad_WFv3_COMMANDER_v2.pkl', allow_pickle= True)
	print(T_data['clusters'].keys())
	T_data['clusters'].pop('simsp')
	T_data['clusters'].pop('grad_orien')
	print(T_data['clusters'].keys())
	np.save('./clusters_T_v2.npy',T_data)

	T_rands=np.load('randoms_m200cut_T_ILC_healpix_nw_apod_zeros_masked_inpaint_planckgrad_WFv3_COMMANDER.pkl', allow_pickle= True)
	print(T_rands['randoms'].keys())
	T_rands['randoms'].pop('simsp')
	T_rands['randoms'].pop('grad_mag')
	T_rands['randoms'].pop('grad_orien')
	print(T_rands['randoms'].keys())
	np.save('./randoms_T.npy',T_rands)

# read and combine TQU data - for planck
T_data=np.load('clusters_T_v2.npy', allow_pickle= True).item()
T_rands=np.load('randoms_T.npy', allow_pickle= True).item()

Q_data=np.load('clusters_m200cut_Q_ILC_healpix_nw_apod_zeros_masked_inpaint_v3.npy', allow_pickle= True).item()
Q_rands=np.load('randoms_m200cut_Q_ILC_healpix_nw_apod_zeros_masked_inpaint_v3.npy', allow_pickle= True).item()

U_data=np.load('clusters_m200cut_U_ILC_healpix_nw_apod_zeros_masked_inpaint_v3.npy', allow_pickle= True).item()
U_rands=np.load('randoms_m200cut_U_ILC_healpix_nw_apod_zeros_masked_inpaint_v3.npy', allow_pickle= True).item()

data={}
data['clusters'] = {}
for key in T_data['clusters'].keys():
    if key == 'cutouts_rotated':
        data['clusters'][key] = np.stack([T_data['clusters'][key], Q_data['clusters'][key], U_data['clusters'][key]], axis=1)
    elif key == 'grad_mag':
        data['clusters'][key] = np.stack([T_data['clusters'][key], Q_data['clusters'][key], U_data['clusters'][key]], axis=1)
    elif key == 'stack':
        data['clusters'][key] = np.stack([T_data['clusters'][key], Q_data['clusters'][key], U_data['clusters'][key]])

randoms={}
randoms['randoms'] = {}
for key in T_rands['randoms'].keys():
    if key == 'stack':
        randoms['randoms'][key] = np.stack([T_rands['randoms'][key], Q_rands['randoms'][key], U_rands['randoms'][key]])

# Print shapes of combined dictionary keys
print(data['clusters']['cutouts_rotated'].shape)  # (5474, 3, 20, 20)
print(data['clusters']['grad_mag'].shape)  # (5474, 3)
print(data['clusters']['stack'].shape)  # (3, 20, 20)
print(randoms['randoms']['stack'].shape)  # (3, 20, 20)

np.save('./clusters_TQU.npy',data)
np.save('./randoms_TQU.npy',randoms)
# data={}
# data['clusters'] = {}
# for key in T_data['clusters'].keys():
#     data['clusters'][key] = [T_data['clusters'][key], Q_data['clusters'][key], U_data['clusters'][key]]

# randoms={}
# randoms['randoms'] = {}
# for key in T_rands['randoms'].keys():
#     randoms['randoms'][key] = [T_rands['randoms'][key], Q_rands['randoms'][key], U_rands['randoms'][key]]

# print(data['clusters'].keys())

# print(randoms['randoms'].keys())