import jax.numpy as jnp
import numpy as np
def resize_flm(flm, target_L):
    """
    resize the flm from L to target_L
    
    Args:
        flm: the original flm, shape: (L, 2L-1, d)
        target_L: the target bandlimit
        
    Returns:
        the resized flm, shape: (target_L, 2*target_L-1, d)
    """
    L = flm.shape[0]
    d = flm.shape[2] if len(flm.shape) > 2 else 1
    
    # create the zero array with the target size
    resized_flm = jnp.zeros((target_L, 2*target_L-1, d), dtype=flm.dtype)
    
    # calculate the center position of m in the new array
    center_new = target_L - 1
    center_old = L - 1
    if L < target_L:
    # copy the original coefficients
        for ell in range(L):
            # the range of m is [-ell, ell]
            m_start = center_old - ell  # the start index in the original array
            m_end = center_old + ell + 1  # the end index in the original array
            
            # the corresponding position in the new array
            new_m_start = center_new - ell
            new_m_end = center_new + ell + 1
            
            resized_flm = resized_flm.at[ell, new_m_start:new_m_end].set(flm[ell, m_start:m_end])
    else:
        # resize the flm
        for ell in range(target_L):
            # the range of m is [-ell, ell]
            m_start = center_old - ell  # the start index in the original array
            m_end = center_old + ell + 1  # the end index in the original array
            
            # the corresponding position in the new array
            new_m_start = center_new - ell
            new_m_end = center_new + ell + 1
            
            resized_flm = resized_flm.at[ell, new_m_start:new_m_end].set(flm[ell, m_start:m_end])
    
    return resized_flm



def Legendre_Polynomial(x, L):
    """
    calculate the Legendre polynomial
    
    Args:
        x: the input value
        L: the polynomial order
        
    Returns:
        the result
    """

    if L == 0:
        return jnp.ones_like(x)
    elif L == 1:
        return x
    else:
        return ((2*L-1)*x*Legendre_Polynomial(x, L-1) - (L-1)*Legendre_Polynomial(x, L-2)) / L

def Legendre_Polynomial_Derivative(x, L):
    """
    calculate the derivative of the Legendre polynomial
    
    Args:
        x: the input value
        L: the polynomial order
        
    Returns:
        the result
    """

    if L == 0:
        return jnp.zeros_like(x)
    elif L == 1:
        return jnp.ones_like(x)
    else:
        return (L/x)*Legendre_Polynomial(x, L-1) - ((L-1)/x)*Legendre_Polynomial(x, L-2)

def infer_L_from_shape(x, sampling):
    """
    infer the bandlimit from the shape of the input x ()
    
    Args:
        x: the input
        sampling: the sampling method
    """
    if sampling == "dh":
        return x.shape[0] // 2
    else:  # mw, mwss
        return x.shape[0]

def get_phi_dim(L, sampling):
    return 2 * L-1 if sampling == "dh" else 2 * L - 1

def pad_inverse_output(x, sampling):
    # for the dh sampling, the output of the inverse is (2L, 2L-1, C), pad the last row to (2L, 2L, C)
    if sampling != "dh":
        return x
    # x.shape = (2L, 2L-1, C)
    last_col = x[:, -1:, :]  # copy the last column
    x_padded = jnp.concatenate([x, last_col], axis=1)
    return x_padded
def get_sampling_grid(L: int, sampling: str = "mw"):
    """
    Generate theta and phi sampling grids for spherical coordinates.
    
    Args:
        L: bandlimit
        sampling: "mw" or "dh"
    
    Returns:
        theta: (L, 2L-1)
        phi: (L, 2L-1)
    """
    if sampling == "mw":
        n_theta = L
        n_phi = 2 * L - 1
        theta = jnp.linspace(0, jnp.pi, n_theta)
        phi = jnp.linspace(0, 2 * jnp.pi, n_phi, endpoint=False)
    elif sampling == "dh":
        n_theta = 2 * L
        n_phi = 2 * L
        theta = jnp.linspace(0, jnp.pi, n_theta, endpoint=False) + jnp.pi / (2 * n_theta)
        phi = jnp.linspace(0, 2 * jnp.pi, n_phi, endpoint=False)
    elif sampling == "gl":
        n_theta = L
        n_phi = 2 * L - 1
        nodes, weights = np.polynomial.legendre.leggauss(L)
        theta = jnp.flip(jnp.arccos(nodes))
        phi = jnp.linspace(0, 2 * jnp.pi, n_phi, endpoint=False)
    else:
        raise ValueError(f"Unsupported sampling: {sampling}")

    phi_grid, theta_grid = jnp.meshgrid(phi, theta)
    return theta_grid, phi_grid  # shape: (n_theta, n_phi)


def get_theta_from_x(x):
    """
    Infer theta values from input x of shape (L, 2L-1, 3)
    Works only if input is sampled on a sphere.
    """
    z = x[..., 2]  # z = cos(theta)
    z = jnp.clip(z, -1.0, 1.0)
    theta = jnp.arccos(z)
    return theta