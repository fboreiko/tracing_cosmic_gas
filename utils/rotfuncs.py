import numpy as np
import jax
import jax.numpy as jnp

def moveaxis(a, o, n):
    """Move axis - works with both numpy and JAX arrays"""
    if o < 0: o = o + a.ndim
    if n < 0: n = n + a.ndim
    if n <= o: 
        return jnp.moveaxis(a, o, n) if isinstance(a, jnp.ndarray) else np.rollaxis(a, o, n)
    else: 
        return jnp.moveaxis(a, o, n+1) if isinstance(a, jnp.ndarray) else np.rollaxis(a, o, n+1)


def rotmatrix(ang, raxis, axis=0):
	"""Construct a 3d rotation matrix representing a rotation of
	ang degrees around the specified rotation axis raxis, which can be "x", "y", "z"
	or 0, 1, 2. If ang is a scalar, the result will be [3,3]. Otherwise,
	it will be ang.shape + (3,3)."""
	ang  = np.asarray(ang)
	raxis = raxis.lower()
	c, s = np.cos(ang), np.sin(ang)
	R = np.zeros(ang.shape + (3,3))
	if   raxis == 0 or raxis == "x": R[...,0,0]=1;R[...,1,1]= c;R[...,1,2]=-s;R[...,2,1]= s;R[...,2,2]=c
	elif raxis == 1 or raxis == "y": R[...,0,0]=c;R[...,0,2]= s;R[...,1,1]= 1;R[...,2,0]=-s;R[...,2,2]=c
	elif raxis == 2 or raxis == "z": R[...,0,0]=c;R[...,0,1]=-s;R[...,1,0]= s;R[...,1,1]= c;R[...,2,2]=1
	else: raise ValueError("Rotation axis %s not recognized" % raxis)
	return moveaxis(R, 0, axis)

@jax.jit
def rotmatrix_jax(ang, raxis_idx):
    """
    JAX-compiled rotation matrix.
    
    Construct a 3d rotation matrix representing a rotation of
    ang radians around the specified rotation axis.
    
    ang: rotation angle in radians (scalar or array)
    raxis_idx: 0 for x, 1 for y, 2 for z
    
    Returns: rotation matrix of shape ang.shape + (3,3)
    """
    ang = jnp.asarray(ang)
    c, s = jnp.cos(ang), jnp.sin(ang)
    
    # Initialize identity-like matrix
    R = jnp.zeros(ang.shape + (3,3))
    
    # Use jax.lax.switch for conditional logic
    def make_x_rotation(c, s):
        R = jnp.zeros((3, 3))
        R = R.at[0, 0].set(1.0)
        R = R.at[1, 1].set(c)
        R = R.at[1, 2].set(-s)
        R = R.at[2, 1].set(s)
        R = R.at[2, 2].set(c)
        return R
    
    def make_y_rotation(c, s):
        R = jnp.zeros((3, 3))
        R = R.at[0, 0].set(c)
        R = R.at[0, 2].set(s)
        R = R.at[1, 1].set(1.0)
        R = R.at[2, 0].set(-s)
        R = R.at[2, 2].set(c)
        return R
    
    def make_z_rotation(c, s):
        R = jnp.zeros((3, 3))
        R = R.at[0, 0].set(c)
        R = R.at[0, 1].set(-s)
        R = R.at[1, 0].set(s)
        R = R.at[1, 1].set(c)
        R = R.at[2, 2].set(1.0)
        return R
    
    # Vectorize over the angle dimensions if needed
    if ang.ndim == 0:
        # Scalar case
        if raxis_idx == 0:
            return make_x_rotation(c, s)
        elif raxis_idx == 1:
            return make_y_rotation(c, s)
        else:  # raxis_idx == 2
            return make_z_rotation(c, s)
    else:
        # Array case - use vmap
        if raxis_idx == 0:
            return jax.vmap(make_x_rotation)(c, s)
        elif raxis_idx == 1:
            return jax.vmap(make_y_rotation)(c, s)
        else:  # raxis_idx == 2
            return jax.vmap(make_z_rotation)(c, s)

def ang2rect(angs, zenith=True, axis=0):
	"""Convert a set of angles [{phi,theta},...] to cartesian
	coordinates [{x,y,z},...]. If zenith is True (the default),
	the theta angle will be taken to go from 0 to pi, and measure
	the angle from the z axis. If zenith is False, then theta
	goes from -pi/2 to pi/2, and measures the angle up from the xy plane."""
	phi, theta = moveaxis(angs, axis, 0)
	ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
	if zenith: res = np.array([st*cp,st*sp,ct])
	else:      res = np.array([ct*cp,ct*sp,st])
	return moveaxis(res, 0, axis)

@jax.jit
def ang2rect_jax(angs, zenith=True):
    """
    JAX-compiled conversion from angles to cartesian coordinates.
    
    angs: [2, ...] array with [phi, theta]
    zenith: If True, theta measures angle from z-axis (0 to pi)
            If False, theta measures angle from xy-plane (-pi/2 to pi/2)
    
    Returns: [3, ...] array with [x, y, z]
    """
    phi = angs[0]
    theta = angs[1]
    
    ct, st = jnp.cos(theta), jnp.sin(theta)
    cp, sp = jnp.cos(phi), jnp.sin(phi)
    
    if zenith:
        res = jnp.stack([st*cp, st*sp, ct], axis=0)
    else:
        res = jnp.stack([ct*cp, ct*sp, st], axis=0)
    
    return res

def rect2ang(rect, zenith=True, axis=0):
	"""The inverse of ang2rect."""
	x,y,z = moveaxis(rect, axis, 0)
	r     = (x**2+y**2)**0.5
	phi   = np.arctan2(y,x)
	if zenith: theta = np.arctan2(r,z)
	else:      theta = np.arctan2(z,r)
	return moveaxis(np.array([phi,theta]), 0, axis)

@jax.jit
def rect2ang_jax(rect, zenith=True):
    """
    JAX-compiled conversion from cartesian to angles.
    
    rect: [3, ...] array with [x, y, z]
    zenith: If True, theta measures angle from z-axis
            If False, theta measures angle from xy-plane
    
    Returns: [2, ...] array with [phi, theta]
    """
    x, y, z = rect[0], rect[1], rect[2]
    r = jnp.sqrt(x**2 + y**2)
    phi = jnp.arctan2(y, x)
    
    if zenith:
        theta = jnp.arctan2(r, z)
    else:
        theta = jnp.arctan2(z, r)
    
    return jnp.stack([phi, theta], axis=0)

def euler_mat(euler_angles, kind="zyz"):
	"""Defines the rotation matrix M for a ABC euler rotation,
	such that M = A(alpha)B(beta)C(gamma), where euler_angles =
	[alpha,beta,gamma]. The default kind is ABC=ZYZ."""
	alpha, beta, gamma = euler_angles
	R1 = rotmatrix(gamma, kind[2])
	R2 = rotmatrix(beta,  kind[1])
	R3 = rotmatrix(alpha, kind[0])
	return np.einsum("...ij,...jk->...ik",np.einsum("...ij,...jk->...ik",R3,R2),R1)

@jax.jit
def euler_mat_jax(alpha, beta, gamma):
    """
    JAX-compiled Euler rotation matrix for ZYZ convention.
    
    Returns: M = R_z(alpha) @ R_y(beta) @ R_z(gamma)
    """
    # Z rotation by gamma
    c1, s1 = jnp.cos(gamma), jnp.sin(gamma)
    R1 = jnp.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
    
    # Y rotation by beta
    c2, s2 = jnp.cos(beta), jnp.sin(beta)
    R2 = jnp.array([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]])
    
    # Z rotation by alpha
    c3, s3 = jnp.cos(alpha), jnp.sin(alpha)
    R3 = jnp.array([[c3, -s3, 0], [s3, c3, 0], [0, 0, 1]])
    
    return R3 @ R2 @ R1

def euler_rot(euler_angles, coords, kind="zyz"):
	coords = np.asarray(coords)
	co     = coords.reshape(2,-1)
	M      = euler_mat(euler_angles, kind)
	rect   = ang2rect(co, False)
	rect   = np.einsum("...ij,j...->i...",M,rect)
	co     = rect2ang(rect, False)
	return co.reshape(coords.shape)

@jax.jit
def euler_rot_jax(alpha, beta, gamma, coords):
    """
    JAX-compiled Euler rotation of coordinates.
    
    coords: [2, ...] array of angles [phi, theta]
    
    Returns: rotated coordinates [2, ...]
    """
    # Get rotation matrix
    M = euler_mat_jax(alpha, beta, gamma)
    
    # Convert to cartesian
    rect = ang2rect_jax(coords, zenith=False)
    
    # Rotate
    # rect is [3, ...], M is [3, 3]
    # We need to do matrix multiplication on the first axis
    original_shape = rect.shape[1:]
    rect_flat = rect.reshape(3, -1)  # [3, N]
    rect_rot = M @ rect_flat  # [3, N]
    rect_rot = rect_rot.reshape(3, *original_shape)
    
    # Convert back to angles
    coords_rot = rect2ang_jax(rect_rot, zenith=False)
    
    return coords_rot

def recenter(angs, center):
	"""recenter(angs, [from_ra, from_dec]) or recenter(angs, [from_ra, from_dec, to_ra, to_dec]).
	In the first form, performs a coordinate rotation such that a point at (form_ra, from_to)
	ends up at the north pole. In the second form, that point is instead put at (to_ra, to_dec)."""
	# Performs the rotation E(0,-theta,-phi). Originally did
	# E(phi,-theta,-phi), but that is wrong (at least for our
	# purposes), as it does not preserve the relative orientation
	# between the boresight and the sun. For example, if the boresight
	# is at the same elevation as the sun but 10 degrees higher in az,
	# then it shouldn't matter what az actually is, but with the previous
	# method it would.
	#
	# Now supports specifying where to recenter by specifying center as
	# lon_from,lat_from,lon_to,lat_to
	if len(center) == 4: ra0, dec0, ra1, dec1 = center
	elif len(center) == 2: ra0, dec0, ra1, dec1 = center[0], center[1], 0, np.pi/2
	return euler_rot([ra1,dec0-dec1,-ra0], angs, kind="zyz")

def recenter_jax(angs, ra0, dec0, ra1, dec1):
    """
    JAX-compiled version of recenter for batch processing.
    
    angs: [2, ...] coordinates to recenter
    """
    alpha = ra1
    beta = dec0 - dec1
    gamma = -ra0
    
    return euler_rot_jax(alpha, beta, gamma, angs)

def decenter(angs, center):
	"""Inverse operation of recenter."""
	if len(center) == 4: ra0, dec0, ra1, dec1 = center
	elif len(center) == 2: ra0, dec0, ra1, dec1 = center[0], center[1], 0, np.pi/2
	return euler_rot([ra0,dec1-dec0,-ra1],  angs, kind="zyz")