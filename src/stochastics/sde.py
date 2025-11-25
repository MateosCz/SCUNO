from abc import ABC, abstractmethod
from typing import Tuple, Callable, Optional

import jax.numpy as jnp
import jax
from jax.typing import DTypeLike

class SDE(ABC):

    @abstractmethod
    def drift_fn(self):
        pass

    @abstractmethod
    def diffusion_fn(self):
        pass

class Time_Reversed_SDE_2Dmanifold_Yang(SDE):
    def __init__(self, original_sde: SDE, score_fn: Callable[[jnp.ndarray, float], jnp.ndarray], total_time: float, dt: float, noise_size: Optional[int] = None):
        super().__init__()
        self.original_sde = original_sde
        self.score_fn = score_fn
        self.total_time = total_time
        self.dt = dt
        self.epsilon = 1e-5
        self.noise_size = noise_size if noise_size is not None else original_sde.noise_size
    def compute_div_sigma(self, x: jnp.ndarray, t: float) -> jnp.ndarray:
        def div_sigma_single(x_i):
            def sigma_comp(i):
                sigma_i = lambda x: self.original_sde.diffusion_fn(x, t)[i]
                return jnp.trace(jax.jacfwd(sigma_i)(x_i))
                # return jnp.trace(jax.jacrev(sigma_i)(x_i))
            return jax.vmap(sigma_comp)(jnp.arange(x_i.shape[0]))
        return jax.vmap(div_sigma_single)(x)

    def drift_fn(self, x, t, x0):
        jax.debug.print("score_fn: {0}", self.score_fn(x, self.total_time - t + self.dt, x0))
        def drift_fn_impl(x,t, x0):
            score_cond = self.score_fn(x, self.total_time - t + self.dt, x0)
            drift = -self.original_sde.drift_fn(x, self.total_time - t + self.dt) + score_cond
            # div_sigma = self.compute_div_sigma(x, self.total_time - t + self.dt)
            # drift -= div_sigma
            return drift
 
        return drift_fn_impl(x, t, x0)
    
    def diffusion_fn(self, x, t):
        return self.original_sde.diffusion_fn(x, self.total_time - t + self.dt)
    def Sigma(self, x, t):
        return jnp.einsum('ijk,klm->ijlm', self.diffusion_fn(x, t), self.diffusion_fn(x, t).T)

class Kunita_Flow_SDE_3D_Eulerian_2Dmanifold_distance(SDE):
    '''
    Kunita flow SDE on 2D manifold with distance metric, dx = sigma(x, t) * dW
    X: S1 x S1 -> R^3
    x dimension : (S1, S1, 3) (R^d landmark position, d=3)
    t dimension : (num_particles, 1) (time)
    '''
    def __init__(self, k_alpha: DTypeLike, k_sigma: DTypeLike, grid_num: int, grid_range: Tuple[float, float], x0: jnp.ndarray):
        super().__init__()
        self.k_alpha = k_alpha
        self.k_sigma = k_sigma
        self.grid_dim = 3
        self.grid_num = grid_num
        self.grid_range = grid_range
        self.noise_size = grid_num ** 3
        self.d_grid = ((grid_range[1]-grid_range[0]) / grid_num) ** 3 # small square grid size
        self.x0 = x0

    @property
    def grid(self):
        grid_x = jnp.linspace(*self.grid_range, self.grid_num)
        grid_y = jnp.linspace(*self.grid_range, self.grid_num)
        grid_z = jnp.linspace(*self.grid_range, self.grid_num)
        grid_x, grid_y, grid_z = jnp.meshgrid(grid_x, grid_y, grid_z, indexing='xy')
        grid = jnp.stack([grid_x, grid_y, grid_z], axis=-1)
        grid = grid.reshape(-1, 3)
        return grid

    def drift_fn(self, x, t):
        return jnp.zeros_like(x)    

    def diffusion_fn(self, x, t):
        x = x + self.x0
        def Q_half(x, t):

            # define the kernel function
            kernel_fn = lambda x, y: self.k_alpha * jnp.exp(-0.5 * jnp.linalg.norm(x - y, axis=-1) ** 2 / self.k_sigma ** 2)
            # compute the kernel matrix
            print(self.grid.shape)
            print(x.shape)
            Q_half = jax.vmap(jax.vmap(jax.vmap(kernel_fn, in_axes=(0, None)), in_axes=(None, 0)), in_axes=(None, 0))(self.grid, x) * self.d_grid

            # the integral(simulated) happens when we do the matrix multiplication in the sde solver, so here we just return the kernel matrix
            return Q_half 
        return Q_half(x, t)
    
    def Sigma(self, x, t):
        sigma = self.diffusion_fn(x, t)
        return jnp.einsum('ijk,klm->ijlm', sigma, sigma.T)
