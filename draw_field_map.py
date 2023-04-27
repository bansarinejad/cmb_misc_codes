#! /usr/bin/env python

import numpy as np
import sys
import os
from spt3g import core, maps, mapspectra
from spt3g.maps.maputils import healpix_to_flatsky
from spt3g.std_processing import CreateFieldMapStub


def plot_healpix_maps(frame, outname, res_amin=0.5, do_weight=False, plot_dir=''):
    """
    Plot a map frame containing HealpixSkyMaps. This projects down to the usual proj5 flat sky
    and then calls plot_flat_sky_maps(). See that method for more details.

    Arguments:
    ----------
    frame : core.G3Frame
        Map frame containing HealpixSkyMaps. Note that this assumes this corresponds to the usual 1500d winter field.
    outname : string
        String to prepend to the file names, usually the file name.
    res_amin : float
        Resolution of the flat sky map to project the Healpix map into. 2 arcminutes is reasonable, turn it up if you want higher resolution
        or down to speed up the method call.
    do_weight : bool
        Plot the map weight as well as the maps themselves.
    plot_dir : str
        Directory in which to save plots
    """

    # Turn a frame consisting of a bunch of healpix maps into a bunch of flat sky maps, then run these through the normal plotting call

    maps.RemoveWeights(frame)

    frame_flat = core.G3Frame(core.G3FrameType.Map)
    frame_flat['Id'] = frame['Id']

    for stoke in ['T', 'Q', 'U']:

        frame_flat[stoke] = healpix_to_flatsky(
            frame[stoke],
            map_stub=CreateFieldMapStub(
                res=res_amin * core.G3Units.arcmin, pol_type=frame[stoke].pol_type, proj = spt3g.maps.MapProjection(0)
            ),
            interp=False,
            rebin=1,
        )

    # For the apodization mask
    weights_flat = maps.G3SkyMapWeights()
    weights_flat.TT = healpix_to_flatsky(
        frame['Wpol'].TT,
        map_stub=CreateFieldMapStub(
            res=res_amin * core.G3Units.arcmin, pol_type=frame['Wpol'].TT.pol_type, proj = spt3g.maps.MapProjection(0)
        ),
        interp=False,
        rebin=1,
    )
    weights_flat.QQ = healpix_to_flatsky(
        frame['Wpol'].QQ,
        map_stub=CreateFieldMapStub(
            res=res_amin * core.G3Units.arcmin, pol_type=frame['Wpol'].QQ.pol_type, proj = spt3g.maps.MapProjection(0)
        ),
        interp=False,
        rebin=1,
    )
    weights_flat.UU = healpix_to_flatsky(
        frame['Wpol'].UU,
        map_stub=CreateFieldMapStub(
            res=res_amin * core.G3Units.arcmin, pol_type=frame['Wpol'].UU.pol_type, proj = spt3g.maps.MapProjection(0)
        ),
        interp=False,
        rebin=1,
    )

    frame_flat['Wpol'] = weights_flat

    # Run the output maps through the usual flat sky plotting call
    plot_flat_sky_maps(frame_flat, outname, do_weight=do_weight, plot_dir=plot_dir)


def plot_flat_sky_maps(frame, outname, do_weight=False, plot_dir=''):
    """
    Plot a map frame and save to files. The files are named by concatening the outname field and the frame Ids.
    Clips the color scale based on the middle 90% of nonzero pixel values and draws in a black and white colorscale.
    Note that this assumes maps are polarized, which we may want to generalize eventually.

    Arguments:
    ----------
    frame : core.G3Frame
        Map frame containing HealpixSkyMaps. Note that this assumes this corresponds to the usual 1500d winter field.
    outname : string
        String to prepend to the file names, usually the file name.
    do_weight : bool
        Plot the map weight as well as the maps themselves.
    plot_dir : str
        Directory in which to save plots
    """

    import matplotlib

    matplotlib.use('agg')
    matplotlib.rcParams['figure.figsize'] = [16, 9]
    matplotlib.rcParams['axes.titlesize'] = 20
    matplotlib.rcParams['axes.labelsize'] = 20

    from matplotlib.pyplot import (
        figure,
        imshow,
        xlabel,
        ylabel,
        gca,
        colorbar,
        tight_layout,
        cm,
        savefig,
        close,
    )

    res_amin = frame['T'].res / core.G3Units.arcmin
    x_extent = frame['T'].shape[1] * (res_amin / 60.0)
    y_extent = frame['T'].shape[0] * (res_amin / 60.0)

    apod_mask = mapspectra.map_analysis.apodmask.make_border_apodization(
        frame, radius_arcmin=30.0
    )

    maps.RemoveWeights(frame)

    for stoke, weight in zip(
        'T Q U'.split(),
        [
            np.asarray(frame['Wpol'].TT),
            np.asarray(frame['Wpol'].UU),
            np.asarray(frame['Wpol'].QQ),
        ],
    ):

        # Map divided by weight
        map_deweight = np.array(frame[stoke] * apod_mask)
        map_range = (
            np.nanpercentile(map_deweight, 95) - np.nanpercentile(map_deweight, 5)
        ) / 2.0

        # Cut low weight pixels
        map_deweight[np.logical_not(weight > 1e-3 * np.nanmax(weight))] = np.nan
        weight[np.logical_not(weight > 1e-3 * np.nanmax(weight))] = np.nan

        figure()
        imshow(
            map_deweight,
            extent=(x_extent * -0.5, x_extent * 0.5, y_extent * -0.5, y_extent * 0.5),
            interpolation='nearest',
            origin='lower',
            vmin=-1.0 * map_range,
            vmax=map_range,
            aspect='auto',
            cmap=cm.gray,
        )
        xlabel('dRa, $^\circ$')
        ylabel('dDec, $^\circ$')
        gca().grid(True)
        colorbar()
        tight_layout()
        savefig(
            os.path.join(
                plot_dir,
                '%s_%s_%s_map.png' % (outname, frame['Id'].replace('*', ''), stoke),
            )
        )

        if do_weight:
            figure()
            imshow(
                weight,
                extent=(
                    x_extent * -0.5,
                    x_extent * 0.5,
                    y_extent * -0.5,
                    y_extent * 0.5,
                ),
                interpolation='nearest',
                origin='lower',
                vmin=None,
                vmax=None,
                aspect='auto',
                cmap=cm.Blues,
            )
            xlabel('dRa, $^\circ$')
            ylabel('dDec, $^\circ$')
            tight_layout()
            gca().grid(True)
            colorbar()
            savefig(
                os.path.join(
                    plot_dir,
                    '%s_%s_%s_weight.png'
                    % (outname, frame['Id'].replace('*', ''), stoke),
                )
            )

        close('all')


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('maps_in', nargs='+')
    parser.add_argument(
        '--weight', action='store_true', help='Also plot the map weights'
    )
    parser.add_argument('--plot-dir', default='', help='Directory to save plots')
    args = parser.parse_args()
    os.makedirs(args.plot_dir, exist_ok=True)

    for infile in args.maps_in:

        outname = os.path.split(infile)[-1].split('.')[0]

        for frame in core.G3File(infile):

            if frame.type != core.G3FrameType.Map:
                continue

            if 'Id' not in frame.keys():
                frame['Id'] = 'unknown'

            if isinstance(frame['T'], maps.FlatSkyMap):
                plot_flat_sky_maps(
                    frame, outname, do_weight=args.weight, plot_dir=args.plot_dir
                )
            elif isinstance(frame['T'], maps.HealpixSkyMap):
                plot_healpix_maps(
                    frame, outname, do_weight=args.weight, plot_dir=args.plot_dir
                )
            else:
                core.log_warn('Unrecognized map type')