import healpy as hp
#import matplotlib.pyplot as plt

# Read the HEALPix map from the FITS file
t_95 = hp.fitsfunc.read_map('./winter/no_signflip_bundle_000_90GHz_nw_cut.fits',field=0)
t_150 = hp.fitsfunc.read_map('./winter/no_signflip_bundle_000_150GHz_nw_cut.fits',field=0)
t_220 = hp.fitsfunc.read_map('./winter/no_signflip_bundle_000_220GHz_nw_cut.fits',field=0)

# Plot the map
hp.mollview(t_95, norm=None, min=None, max=None);show()
hp.mollview(t_150,norm=None, min=None, max=None);show()
hp.mollview(t_220, norm=None, min=None, max=None);show()

