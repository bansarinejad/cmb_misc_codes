import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pylab import *
from scipy.ndimage import rotate

# Read in npy file
data = np.load('./stacks/clusters_m200cut_Q_ILC_healpix_nw_apod_zeros_masked_inpaint_v3_gradtest_saturo_fullorient.npy',allow_pickle=True).item()

data=data['clusters']['grad_orien_full']

# Squeeze the array to remove any extra dimensions
data = np.squeeze(data)

# Take the first element of the array which is a 12x12 pixel image
image = data[1]
if (0):
    x_temp = np.arange(12)
    y_temp = np.arange(12)
    xdata_temp, ydata_temp = np.meshgrid(x_temp, y_temp)

    # Generate an image with a gradient smoothly increasing from left to right
    image = xdata_temp
    image = rotate(image, angle=45, reshape=True)

# Define the function to fit to the 2D image plane
def func(xy, a, b):
    x, y = xy.T
    return (a * x + b * y).ravel()

# Define the x and y coordinates of the image pixels
x = np.arange(image.shape[0])
y = np.arange(image.shape[1])
xdata, ydata = np.meshgrid(x, y)

# Reshape the image into a 1D array for curve_fit
zdata = image.ravel()

# Perform the curve_fit to calculate the gradient
popt, pcov = curve_fit(func, np.column_stack((xdata.flatten(), ydata.flatten())), zdata)

# The gradient is the first coefficient of the fit
gradient = popt[0]

# Plot the original image and the fitted function
fig, ax = plt.subplots()
im = ax.imshow(image, cmap='gray', origin='lower')
ax.set_title('Original Image')
xgrid, ygrid = np.meshgrid(np.linspace(0, 11, 100), np.linspace(0, 11, 100))
xygrid = np.column_stack((xgrid.flatten(), ygrid.flatten()))
zfit = func(xygrid, *popt).reshape(xgrid.shape)
ax.contour(xgrid, ygrid, zfit, colors='r', linewidths=2)
ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
imshow(zfit);show()

# Print the gradient orientation
amp=np.hypot(popt[0],popt[1])
print(amp)
angle = np.arctan2(popt[0], popt[1]) * 180 / np.pi
print(f"The gradient orientation is {angle:.1f} degrees.")