import numpy as np
import matplotlib.pyplot as plt

planck_data = np.load('./planck/Planck_cutouts_T_nw_masked_WFv3_COMMANDER.npy', allow_pickle=True)
lindsey_data = np.load('./lindsey_maps/LM_cutouts_T_nw_masked_WFv3.npy', allow_pickle=True)

# Create a new figure and two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first slice from the Planck data
im1 = ax1.imshow(planck_data[0], cmap='viridis')
ax1.set_title('Planck CMB Cutout (First Slice)')
fig.colorbar(im1, ax=ax1)

# Plot the first slice from the Lindsey data
im2 = ax2.imshow(lindsey_data[0], cmap='viridis')
ax2.set_title('Lindsey CMB Cutout (First Slice)')
fig.colorbar(im2, ax=ax2)

# Display the plots
plt.show()
