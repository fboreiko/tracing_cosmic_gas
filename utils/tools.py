import numpy as np
from pixell import enmap, enplot, utils
from . import rotfuncs
import os
import jax
import jax.numpy as jnp
from jax import vmap

def cutoutGeometry(projCutout='cea', rApMaxArcmin=6., resCutoutArcmin=0.25, test=False):
    '''
    Create enmap for the cutouts to be extracted.
    Returns a null enmap object with the right shape and wcs.
    projCutout: Projection type for the cutout (e.g. 'cea', 'car')
    '''

    # choose postage stamp size to fit the largest ring
    dArcmin = np.ceil(2. * rApMaxArcmin * np.sqrt(2.))        # compute side length in arcmin, rounds up to the nearest integer to be safe.
   
    nx = np.floor((dArcmin / resCutoutArcmin - 1.) / 2.) + 1. # number of pixels from center to edge
    dxDeg = (2. * nx + 1.) * resCutoutArcmin / 60.            # full side length in degrees
    ny = np.floor((dArcmin / resCutoutArcmin - 1.) / 2.) + 1. # number of pixels from center to edge
    dyDeg = (2. * ny + 1.) * resCutoutArcmin / 60.            # full side length in degrees

    # define geometry of small square maps to be extracted
    shape, wcs = enmap.geometry(np.array([[-0.5*dxDeg,-0.5*dyDeg],[0.5*dxDeg,0.5*dyDeg]])*utils.degree, res=resCutoutArcmin*utils.arcmin, proj=projCutout)
    cutoutMap = enmap.zeros(shape, wcs)
    
    if test:
        print("cutout sides are dx, dy =", dxDeg*60., ",", dyDeg*60. , "arcmin")
        print("cutout pixel dimensions are", shape)
        print("hence a cutout resolution of", dxDeg*60./shape[0], ",", dyDeg*60./shape[1], "arcmin per pixel")
        print("(requested", resCutoutArcmin, "arcmin per pixel)")
        
    return cutoutMap

def extractStamp(cmbMap, ra, dec, rApMaxArcmin, resCutoutArcmin, projCutout, Lbox_rad, pathTestFig='plots/', test=False, stampnum=1, cmbMask=None, order=1):
    """
    Extracts a small CEA or CAR map around the given position, with the given angular size and resolution.
    ra, dec in degrees.
    Does it for the map, the mask and the hit count.
    order > 1 is too slow

    cmbMap:             This is your large, primary map. In my case, this would be the
                        custom τ field (loaded as an enmap object).
    ra, dec:            Coordinates of the center of the cutout (in degrees).
    rApMaxArcmin:       The maximum aperture radius (in arcminutes) you plan to measure. 
                        This is used only to determine how big the cutout stamp needs to be.    
                        cutoutGeometry ensures the stamp is large enough to fully contain this 
                        largest circle.
    resCutoutArcmin:    The pixel resolution (in arcminutes per pixel) that you want your new small 
                        stamp to have. This can be different from the resolution of your big map.
    projCutout:         Projection type for the cutout (e.g. 'cea', 'car')
    Lbox_rad:           The size of the full map in radians (used for periodic wrapping).
    cmbMask (Optional): This is a large mask map (e.g., for point sources or bad pixels) that has 
                        the same geometry as cmbMap. If you provide this, the function will also 
                        extract a stamp from the mask.
    order (Optional):   The order of interpolation for sampling. The default is 1 (bilinear), which 
                        is fast and usually good enough.
    """
    # enmap 
    stampMap = cutoutGeometry(rApMaxArcmin=rApMaxArcmin, resCutoutArcmin=resCutoutArcmin, projCutout=projCutout, test=test)
    stampMask = stampMap.copy()
    
    # A posmap is an array where the value of each pixel is its own coordinate.
    opos = stampMap.posmap()

    # coordinate of the center of the square map we want to extract
    sourcecoord = np.array([ra, dec])*utils.degree  # convert from degrees to radians

    # corresponding true coordinates on the big healpy map. 
    ipos = rotfuncs.recenter(opos[::-1], [0, 0, sourcecoord[0], sourcecoord[1]])[::-1]
        
    # Here, I use bilinear interpolation
    stampMap[:, :] = cmbMap.at(ipos, order=order)
    
    if cmbMask is not None:
        print("also extracting mask stamp")
        stampMask[:, :] = cmbMask.at(ipos, order=order)    
        # re-threshold the mask map, to keep 0 and 1 only
        stampMask[:, :] = 1.*(stampMask[:, :]>0.5)
    
    if test:
        print("Extracted cutouts around ra=", ra, "dec=", dec)
        print("- min, mean, max =", np.min(stampMap), np.mean(stampMap), np.max(stampMap))
        # don't save if empty
        if np.min(stampMap) + np.mean(stampMap) + np.max(stampMap) == 0.: return opos, stampMap, stampMask

        plots = enplot.plot(enmap.upgrade(stampMap, 5), grid=True)
        enplot.write(pathTestFig+"/stampmap_num"+str(stampnum)+"_ra"+str(np.round(ra, 2))+"_dec"+str(np.round(dec, 2)), plots)

    return opos, stampMap, stampMask

# ========================== JAX ACCELERATED FUNCTIONS ==========================

@jax.jit
def bilinear_interpolate_jax(image, coords):
    """
    JAX-accelerated bilinear interpolation.
    
    image: [H, W] array
    coords: [2, N] array where coords[0] is y-coords and coords[1] is x-coords in pixels
    
    Returns: [N] interpolated values
    """
    y_coords = coords[0]
    x_coords = coords[1]
    
    # Get image dimensions
    height, width = image.shape
    
    # Clamp coordinates to valid range
    y_coords = jnp.clip(y_coords, 0, height - 1)
    x_coords = jnp.clip(x_coords, 0, width - 1)
    
    # Get integer parts
    y0 = jnp.floor(y_coords).astype(jnp.int32)
    x0 = jnp.floor(x_coords).astype(jnp.int32)
    y1 = jnp.clip(y0 + 1, 0, height - 1)
    x1 = jnp.clip(x0 + 1, 0, width - 1)
    
    # Get fractional parts
    wy = y_coords - y0
    wx = x_coords - x0
    
    # Get corner values
    Ia = image[y0, x0]
    Ib = image[y1, x0]
    Ic = image[y0, x1]
    Id = image[y1, x1]
    
    # Bilinear interpolation
    wa = (1 - wx) * (1 - wy)
    wb = (1 - wx) * wy
    wc = wx * (1 - wy)
    wd = wx * wy
    
    return wa * Ia + wb * Ib + wc * Ic + wd * Id


@jax.jit
def bilinear_interpolate_batch(image, coords_batch):
    """
    Batch bilinear interpolation.
    
    image: [H, W] input map
    coords_batch: [batch, 2, ny, nx] pixel coordinates for each cutout
    
    Returns: [batch, ny, nx] interpolated stamps
    """
    def interpolate_single(coords):
        # coords: [2, ny, nx]
        y_coords = coords[0]
        x_coords = coords[1]
        
        height, width = image.shape
        
        # Clamp to valid range
        y_coords = jnp.clip(y_coords, 0, height - 1.001)
        x_coords = jnp.clip(x_coords, 0, width - 1.001)
        
        # Integer parts
        y0 = jnp.floor(y_coords).astype(jnp.int32)
        x0 = jnp.floor(x_coords).astype(jnp.int32)
        y1 = jnp.clip(y0 + 1, 0, height - 1)
        x1 = jnp.clip(x0 + 1, 0, width - 1)
        
        # Fractional parts
        wy = y_coords - y0
        wx = x_coords - x0
        
        # Corner values
        Ia = image[y0, x0]
        Ib = image[y1, x0]
        Ic = image[y0, x1]
        Id = image[y1, x1]
        
        # Bilinear weights
        wa = (1 - wx) * (1 - wy)
        wb = (1 - wx) * wy
        wc = wx * (1 - wy)
        wd = wx * wy
        
        return wa * Ia + wb * Ib + wc * Ic + wd * Id
    
    # Vectorize over batch
    return vmap(interpolate_single)(coords_batch)


def extractStamp_jax(cmbMap, ra, dec, rApMaxArcmin, resCutoutArcmin, projCutout, Lbox_rad, 
                     pathTestFig='plots/', test=False, stampnum=1, cmbMask=None, order=1):
    """
    JAX-accelerated version of extractStamp.
    
    Extracts a small CEA or CAR map around the given position, with the given angular size and resolution.
    ra, dec in degrees.
    """
    # Create output stamp geometry
    stampMap = cutoutGeometry(rApMaxArcmin=rApMaxArcmin, resCutoutArcmin=resCutoutArcmin, projCutout=projCutout, test=test)
    stampMask = stampMap.copy() if cmbMask is not None else None
    
    # A posmap is an array where the value of each pixel is its own coordinate.
    opos = stampMap.posmap()

    # coordinate of the center of the square map we want to extract
    sourcecoord = np.array([ra, dec]) * utils.degree  # convert from degrees to radians

    # corresponding true coordinates on the big map
    ipos = rotfuncs.recenter(opos[::-1], [0, 0, sourcecoord[0], sourcecoord[1]])[::-1]
    
    # Convert ipos (in radians) to pixel coordinates for the large map
    # ipos shape: [2, ny, nx] where ipos[0] is dec/y and ipos[1] is ra/x
    # We need to convert from world coordinates to pixel coordinates
    pix_coords = cmbMap.sky2pix(ipos, safe=False)  # [2, ny, nx]
    
    # Flatten for interpolation
    pix_coords_flat = pix_coords.reshape(2, -1)  # [2, N]
    
    # Convert to JAX arrays
    image_jax = jnp.array(np.array(cmbMap))
    pix_coords_jax = jnp.array(pix_coords_flat)
    
    # Perform interpolation using JAX
    stampMap_flat = bilinear_interpolate_jax(image_jax, pix_coords_jax)
    
    # Reshape back to stamp shape
    stampMap[:, :] = np.array(stampMap_flat).reshape(stampMap.shape)

    # Here, I use bilinear interpolation
    #stampMap[:, :] = cmbMap.at(ipos, order=order)
    
    # Handle mask if provided
    if cmbMask is not None:
        mask_jax = jnp.array(np.array(cmbMask))
        stampMask_flat = bilinear_interpolate_jax(mask_jax, pix_coords_jax)
        stampMask[:, :] = np.array(stampMask_flat).reshape(stampMask.shape)
        # re-threshold the mask map, to keep 0 and 1 only
        stampMask[:, :] = 1. * (stampMask[:, :] > 0.5)
    
    return opos, stampMap, stampMask


@jax.jit
def compute_all_apertures_jax(stamp_map_jax, inners_jax, outers_jax, r_comoving_mpc_jax):
    """
    Compute all apertures for a single stamp using JAX.
    
    stamp_map_jax: [H, W] stamp
    inners_jax: [n_radii, H, W] boolean masks
    outers_jax: [n_radii, H, W] boolean masks
    r_comoving_mpc_jax: [n_radii] comoving radii
    
    Returns: a_inns [n_radii], a_outs [n_radii]
    """
    def compute_single_aperture(inner, outer, r_com):
        # Inner aperture - use boolean indexing like the original
        # jnp.mean with where parameter is equivalent to np.mean(array[mask])
        mean_inner = jnp.mean(stamp_map_jax, where=inner)
        area_inner = jnp.pi * r_com ** 2
        a_inn = mean_inner * area_inner
        
        # Outer aperture
        mean_outer = jnp.mean(stamp_map_jax, where=outer)
        area_outer = jnp.pi * r_com ** 2
        a_out = mean_outer * area_outer
        
        return a_inn, a_out
    
    # Vectorize over all apertures
    a_inns, a_outs = vmap(compute_single_aperture)(inners_jax, outers_jax, r_comoving_mpc_jax)
    
    return a_inns, a_outs


def build_aperture_templates_jax(stamp_map, theta_arcmins):
    """
    Generate boolean masks for inner/outer disks for all radii.
    Returns JAX arrays for fast computation.
    """
    inners = np.zeros((len(theta_arcmins), stamp_map.shape[0], stamp_map.shape[1]), dtype=bool)
    outers = np.zeros_like(inners)
    modrmap = stamp_map.modrmap()
    
    for i, th in enumerate(theta_arcmins):
        radius = th * utils.arcmin
        inner = modrmap < radius
        outer = (modrmap >= radius) & (modrmap < radius * np.sqrt(2.0))
        inners[i] = inner
        outers[i] = outer
    
    # Convert to JAX arrays
    inners_jax = jnp.array(inners)
    outers_jax = jnp.array(outers)
    
    return inners_jax, outers_jax


def calc_T_AP(imap, rad_arcmin, test=False, mask=None, divmap=None, measurement="mean"):
    modrmap = imap.modrmap()
    radius = rad_arcmin*utils.arcmin
    inner = modrmap < radius
    outer = (modrmap >= radius) & (modrmap < np.sqrt(2.)*radius)
    #res_arcmin = 0.05
    #print("pixelized, true = ", np.sum(inner), np.pi*(rad_arcmin/res_arcmin)**2)
    #ratio = np.sum(inner)/(np.pi*(rad_arcmin/res_arcmin)**2)
    
    if mask is None:
        print("no")
        if measurement == "mean":
            flux_inner = imap[inner].mean()
            flux_outer = imap[outer].mean()
        elif measurement == "integrated":
            flux_inner = imap[inner].sum()
            flux_outer = imap[outer].sum()
        flux_inner_std = imap[inner].std()
        flux_outer_std = imap[outer].std()
        if divmap is not None:
            divs = divmap[inner].mean()
    else:
        if (np.sum(mask[inner]) == 0.) or (np.sum(mask[outer]) == 0.):
            return 0., 0., 0., 0.
        else:
            if measurement == "mean":
                flux_inner = np.sum(imap[inner]*mask[inner])/np.sum(mask[inner])
                flux_outer = np.sum(imap[outer]*mask[outer])/np.sum(mask[outer])
            elif measurement == "integrated":
                # multiplication by this factor accounts for no boundary conditions in simulation (could implement)
                flux_inner = np.sum(imap[inner]*mask[inner])*(np.sum(inner)/np.sum(mask[inner]))
                flux_outer = np.sum(imap[outer]*mask[outer])*(np.sum(outer)/np.sum(mask[outer]))
            flux_inner_std = np.sqrt(np.sum((imap[inner]*mask[inner]-flux_inner)**2)/np.sum(mask[inner]))
            flux_outer_std = np.sqrt(np.sum((imap[outer]*mask[outer]-flux_outer)**2)/np.sum(mask[outer]))
            if divmap is not None:
                divs = np.sum(divmap[inner]*mask[inner])/np.sum(mask[inner])
    
    if divmap is not None:
        return flux_inner, flux_outer, flux_inner_std, flux_outer_std, divs
    return flux_inner, flux_outer, flux_inner_std, flux_outer_std 


def calc_T_AP_fast(imap, rad_arcmin, inner, mask, measurement="mean"):
    
    if (np.sum(mask[inner]) == 0.):
        print("no")
        return 0., 0., 0., 0.
    else:
        if measurement == "mean":
            flux_inner = np.sum(imap[inner]*mask[inner])/np.sum(mask[inner])
        elif measurement == "integrated":
            # multiplication by this factor accounts for no boundary conditions in simulation (could implement)
            flux_inner = np.sum(imap[inner]*mask[inner])*(np.sum(inner)/np.sum(mask[inner]))

    return flux_inner

def extractStamps(cmbMap1, cmbMap2, cmbMap3, ra, dec, rApMaxArcmin, resCutoutArcmin, projCutout, pathTestFig='figs/', test=False, cmbMask=None, order=1):
    """Extracts a small CEA or CAR map around the given position, with the given angular size and resolution.
    ra, dec in degrees.
    Does it for the map, the mask and the hit count.
    order > 1 is too slow
    """
    # enmap 
    stampMap1 = cutoutGeometry(rApMaxArcmin=rApMaxArcmin, resCutoutArcmin=resCutoutArcmin, projCutout=projCutout)
    stampMap2 = stampMap1.copy()
    stampMap3 = stampMap1.copy()
    stampMask = stampMap1.copy()
    
    # coordinates of the square map (between -1 and 1 deg); output map position [{dec,ra},ny,nx]
    opos = stampMap1.posmap()

    # coordinate of the center of the square map we want to extract
    sourcecoord = np.array([ra, dec])*utils.degree  # convert from degrees to radians

    # corresponding true coordinates on the big healpy map
    ipos = rotfuncs.recenter(opos[::-1], [0, 0, sourcecoord[0], sourcecoord[1]])[::-1]

    # Here, I use bilinear interpolation
    stampMap1[:, :] = cmbMap1.at(ipos, prefilter=True, mask_nan=False, order=order)
    stampMap2[:, :] = cmbMap2.at(ipos, prefilter=True, mask_nan=False, order=order)
    stampMap3[:, :] = cmbMap3.at(ipos, prefilter=True, mask_nan=False, order=order)
    if cmbMask is not None:
        stampMask[:, :] = cmbMask.at(ipos, prefilter=True, mask_nan=False, order=order)

        # re-threshold the mask map, to keep 0 and 1 only
        stampMask[:, :] = 1.*(stampMask[:, :]>0.5)
    
        return opos, stampMap1, stampMap2, stampMap3, stampMask
    
    return opos, stampMap1, stampMap2, stampMap3
