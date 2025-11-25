import jax
import jax.numpy as jnp
import jax.random as jrandom
import abc
import numpy as np
import open3d as o3d
class DataGenerator(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def generate_data(self, key: jnp.ndarray, batch_size: int):
        pass


# spherical coordinate sampled data generator 
# all the data with shape (L, 2L-1, N) is sampled on the unit sphere with the feature channels N
# in this case, the feature channels are decart coordinates
class S2ManifoldDataGenerator(DataGenerator):
    """
    Generate data on various manifolds surfaces with s2fft-compatible sampling. 
    In this case, the sampling scheme is Gauss-Legendre sampling. with the shape of (L, 2L-1).
    This class can generate data on different manifolds surfaces
    with sampling schemes compatible with s2fft for spherical harmonic transforms.
    """
    
    def __init__(self, sampling: str = "gl", manifold_type: str = "sphere", 
                 radius: float = 1.0, height: float = 2.0, minor_radius: float = 0.5, 
                 major_radius: float = 2.0, width: float = 0.5, center: jnp.ndarray = jnp.array([0.0, 0.0, 0.0]), flatten: bool = False, seed=0, randomization: bool = False, epsilon1: float = 1.0, epsilon2: float = 1.0, a: float = 1.0, b: float = 1.0, c: float = 1.0, 
                 A: float = 0.3, n: int = 4, m: int = 5):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.key = jax.random.PRNGKey(seed)
        self.manifold_type = manifold_type
        self.sampling = sampling
        self.radius = radius
        self.height = height
        self.minor_radius = minor_radius
        self.major_radius = major_radius
        self.width = width
        self.center = center
        self.flatten = flatten
        self.randomization = randomization
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.a = a
        self.b = b
        self.c = c  
        self.A = A
        self.n = n
        self.m = m
    def generate_sampling_grid(self, L, sampling='mw'):
        """
        Generate angular sampling grid based on the requested scheme.
        
        Args:
            L: Bandwidth/resolution parameter
            sampling: Sampling scheme ('mw', 'mwss', 'dh', or 'gl') 
            now only support gauss-legendre sampling, other sampling schemes will be supported in the future.
            
        Returns:
            theta_grid: 2D grid of theta values
            phi_grid: 2D grid of phi values
            ntheta: Number of theta samples
            nphi: Number of phi samples
        """
        if self.randomization:
            self.key, key = jrandom.split(self.key)
        
        if sampling == 'mw':
            # McEwen & Wiaux sampling
            ntheta = L
            nphi = 2*L-1
            
            if self.randomization:
                theta = jrandom.uniform(key, (ntheta,), minval=0, maxval=jnp.pi)
                phi = jrandom.uniform(key, (nphi,), minval=0, maxval=2*jnp.pi)
            else:
                theta = jnp.linspace(0, jnp.pi, ntheta, endpoint=True)
                phi = jnp.linspace(0, 2*jnp.pi, nphi, endpoint=False)
            
        elif sampling == 'mwss':
            # McEwen & Wiaux Symmetric Sampling
            ntheta = L + 1
            nphi = 2*L
            if self.randomization:
                theta = jrandom.uniform(key, (ntheta,), minval=0, maxval=jnp.pi)
                phi = jrandom.uniform(key, (nphi,), minval=0, maxval=2*jnp.pi)
            else:
                theta = jnp.linspace(0, jnp.pi, ntheta, endpoint=True)
                phi = jnp.linspace(0, 2*jnp.pi, nphi, endpoint=False)
            
        elif sampling == 'dh':
            # Driscoll & Healy sampling
            ntheta = 2*L
            nphi = 2*L
            if self.randomization:
                theta = jrandom.uniform(key, (ntheta,), minval=0, maxval=jnp.pi) + jnp.pi/(2 *ntheta)
                phi = jrandom.uniform(key, (nphi,), minval=0, maxval=2*jnp.pi)
            else:
                theta = jnp.linspace(0, jnp.pi, ntheta, endpoint=False) + jnp.pi/(2 *ntheta)
                phi = jnp.linspace(0, 2*jnp.pi, nphi, endpoint=False)
        elif sampling == 'gl':
            # Gauss-Legendre sampling
            ntheta = L
            nphi = 2 * L - 1
            nodes, weights = np.polynomial.legendre.leggauss(L)
            theta = jnp.flip(jnp.arccos(nodes))
            phi = jnp.linspace(0, 2 * jnp.pi, nphi, endpoint=False)
            
        else:
            raise ValueError(f"Unsupported sampling scheme: {sampling}. Use 'mw', 'mwss', 'dh', or 'gl'")
        
        # Create meshgrid
        phi_grid, theta_grid = jnp.meshgrid(phi, theta)
        
        return theta_grid, phi_grid, ntheta, nphi
    
    def sphere(self, theta_grid, phi_grid, radius=1.0, center=None):
        """
        Generate points on a sphere.
        
        Args:
            theta_grid: Grid of theta values
            phi_grid: Grid of phi values
            radius: Sphere radius
            center: Sphere center coordinates, default is origin [0,0,0]
            
        Returns:
            points: 3D Cartesian coordinates on the sphere
        """
        if center is None:
            center = jnp.array([0.0, 0.0, 0.0])
        
        # Convert spherical to Cartesian coordinates
        x = radius * jnp.sin(theta_grid) * jnp.cos(phi_grid)
        y = radius * jnp.sin(theta_grid) * jnp.sin(phi_grid)
        z = radius * jnp.cos(theta_grid)
        
        # Apply center offset
        x = x + center[0]
        y = y + center[1]
        z = z + center[2]
        
        # Combine coordinates
        points = jnp.stack([x, y, z], axis=-1)
        
        return points
    
    
    
    def superquadric_sphere(self, theta_grid, phi_grid, epsilon1=1.0, epsilon2=1.0, a=1.0, center=None):
        """
        generate superquadric sphere point cloud.
        theta in [0, π], phi in [0, 2π]
        """
        if center is None:
            center = jnp.array([0.0, 0.0, 0.0])

        # helper: handle the sign of the negative exponent
        def sgnpow(x, p):
            return jnp.sign(x) * jnp.abs(x) ** p

        # calculate the each dimension
        x = a * sgnpow(jnp.sin(theta_grid), epsilon1) * sgnpow(jnp.cos(phi_grid), epsilon2)
        y = a * sgnpow(jnp.sin(theta_grid), epsilon1) * sgnpow(jnp.sin(phi_grid), epsilon2)
        z = a * sgnpow(jnp.cos(theta_grid), epsilon1)

        pts = jnp.stack([x, y, z], axis=-1)
        pts = pts + center[None, None, :]
        return pts
    
    def superellipsoid(self, theta_grid, phi_grid, 
                    a=1.0, b=1.0, c=1.0, 
                    epsilon1=1.0, epsilon2=1.0, 
                    center=None):
        """
        generate super ellipsoid point cloud.
        theta in [0, π], phi in [0, 2π]
        """
        if center is None:
            center = jnp.array([0.0, 0.0, 0.0])
        
        def sgnpow(x, p):
            return jnp.sign(x) * jnp.abs(x) ** p

        cos_theta = jnp.cos(theta_grid)
        sin_theta = jnp.sin(theta_grid)
        cos_phi = jnp.cos(phi_grid)
        sin_phi = jnp.sin(phi_grid)

        x = a * sgnpow(cos_theta, epsilon1) * sgnpow(cos_phi, epsilon2)
        y = b * sgnpow(cos_theta, epsilon1) * sgnpow(sin_phi, epsilon2)
        z = c * sgnpow(sin_theta, epsilon1)

        pts = jnp.stack([x, y, z], axis=-1)
        pts = pts + center[None, None, :]
        return pts
    
    def bump_sphere(self, theta_grid, phi_grid, radius=1.0, A=0.3, n=4, m=5, center=None):
        if center is None:
            center = jnp.array([0.0, 0.0, 0.0])

        r = radius + A * jnp.sin(n * theta_grid) * jnp.cos(m * phi_grid)

        x = r * jnp.sin(theta_grid) * jnp.cos(phi_grid)
        y = r * jnp.sin(theta_grid) * jnp.sin(phi_grid)
        z = r * jnp.cos(theta_grid)

        pts = jnp.stack([x, y, z], axis=-1)
        pts = pts + center[None, None, :]
        return pts

    def real_data(self, theta_grid, phi_grid, center=None, file_path=None, scale=1.0, src_type='pcd', normalize=True, healpix=False):
            """
            Map real world surface sampling points to spherical coordinate system using JAX
            
            Args:
                theta_grid: Polar angle grid (n_lat, n_lon)
                phi_grid: Azimuthal angle grid (n_lat, n_lon) 
                center: Center point coordinates
                file_path: File path
                scale: Scaling factor
                src_type: Source file type ('pcd' or 'mesh')
                normalize: Whether to normalize
            
            Returns:
                mapped_points: Points mapped to spherical grid (n_lat, n_lon, 3)
            """
            if healpix == False:
                n_lat, n_lon = theta_grid.shape
            else:
                n_lat = len(theta_grid)
                n_lon = len(phi_grid)
            
            if center is None:
                center = jnp.array([0.0, 0.0, 0.0])
            if file_path is None:
                raise ValueError("file_path is required")
            
            # 1. Load and sample point cloud/mesh
            if src_type == 'pcd':
                pcd = o3d.io.read_point_cloud(file_path)
                sampled_pcd = pcd.sample_points_poisson_disk(number_of_points=n_lat * n_lon * 4)
                surface_points = jnp.asarray(sampled_pcd.points)
            elif src_type == 'mesh':
                mesh = o3d.io.read_triangle_mesh(file_path)
                if healpix:
                    sampled_pcd = mesh.sample_points_poisson_disk(number_of_points=n_lat * 4)
                else:
                    sampled_pcd = mesh.sample_points_poisson_disk(number_of_points=n_lat * n_lon * 4)
                surface_points = jnp.asarray(sampled_pcd.points)
            else:
                raise ValueError(f"Unsupported source type: {src_type}")
            
            # 2. Preprocess surface points
            if normalize:
                surface_points = surface_points - jnp.mean(surface_points, axis=0)
                surface_norms = jnp.linalg.norm(surface_points, axis=1, keepdims=True)
                surface_points = surface_points / jnp.mean(surface_norms)
            
            surface_points = surface_points * scale + center
            
            # 3. Project surface points to unit sphere and compute spherical coordinates
            surface_points_centered = surface_points - center
            surface_radii = jnp.linalg.norm(surface_points_centered, axis=1, keepdims=True)
            surface_points_normalized = surface_points_centered / (surface_radii + 1e-8)
            
            # Compute spherical coordinates of surface points
            x, y, z = surface_points_normalized[:, 0], surface_points_normalized[:, 1], surface_points_normalized[:, 2]
            surface_theta = jnp.arccos(jnp.clip(z, -1.0, 1.0))  # Polar angle [0, π]
            surface_phi = jnp.arctan2(y, x)  # Azimuthal angle [-π, π]
            
            # 4. Generate target spherical grid coordinates
            if healpix:
                theta_flat = theta_grid.flatten()
                phi_flat = phi_grid.flatten()
            else:
                theta_flat = theta_grid.flatten()
                phi_flat = phi_grid.flatten()
            
            # 5. Perform nearest neighbor search using JAX (spherical distance)
            def spherical_distance(theta1, phi1, theta2, phi2):
                """Calculate distance between two points on sphere"""
                # Using variant of haversine formula
                dtheta = theta2 - theta1
                dphi = phi2 - phi1
                a = jnp.sin(dtheta/2)**2 + jnp.sin(theta1) * jnp.sin(theta2) * jnp.sin(dphi/2)**2
                return 2 * jnp.arcsin(jnp.sqrt(jnp.clip(a, 0, 1)))
            
            def find_nearest_surface_point(target_theta, target_phi):
                """Find nearest surface point for target point"""
                distances = jax.vmap(lambda st, sp: spherical_distance(target_theta, target_phi, st, sp))(
                    surface_theta, surface_phi
                )
                nearest_idx = jnp.argmin(distances)
                return nearest_idx, surface_radii[nearest_idx, 0]
            
            # Perform nearest neighbor search for all target points
            indices_and_radii = jax.vmap(find_nearest_surface_point)(theta_flat, phi_flat)
            nearest_indices, mapped_radii = indices_and_radii
            
            # 6. Generate final mapped points
            target_x = jnp.sin(theta_flat) * jnp.cos(phi_flat)
            target_y = jnp.sin(theta_flat) * jnp.sin(phi_flat)
            target_z = jnp.cos(theta_flat)
            
            target_points = jnp.stack([target_x, target_y, target_z], axis=1)
            mapped_points = target_points * mapped_radii[:, None] + center[None, :]
            if healpix:
                mapped_points = mapped_points
            else:
                mapped_points = mapped_points.reshape(n_lat, n_lon, 3)
            
            return mapped_points

    
    def _generate_data_single(self, L, sampling, key):
        theta_grid, phi_grid, ntheta, nphi = self.generate_sampling_grid(L, sampling)
        if self.manifold_type == 'sphere':
            points = self.sphere(theta_grid, phi_grid, self.radius, self.center)
        elif self.manifold_type == 'superquadric_sphere':
            points = self.superquadric_sphere(theta_grid, phi_grid, self.epsilon1, self.epsilon2, self.a, self.center)
        elif self.manifold_type == 'superellipsoid':
            points = self.superellipsoid(theta_grid, phi_grid, self.a, self.b, self.c, self.epsilon1, self.epsilon2, self.center)
        elif self.manifold_type == 'bump_sphere':
            points = self.bump_sphere(theta_grid, phi_grid, self.radius, self.A, self.n, self.m, self.center)
        else:
            raise ValueError(f"Unsupported manifold type: {self.manifold_type}")
            
        return points

        
    
    def generate_data(self, L, batch_size=1, **kwargs):
        """
        Generate data on the specified manifold with s2fft-compatible sampling.
        
        Args:
            L: Bandwidth/resolution parameter
            batch_size: Batch size
            **kwargs: Additional parameters for the specific manifold
                - radius, center for sphere
            
        Returns:
            points: Tensor of shape (batch_size, ntheta, nphi, 3) with 3D coordinates
        """
        if batch_size > 1:
            self.key, key_new = jrandom.split(self.key)
            key_new = jrandom.split(key_new, batch_size)
            points = jax.vmap(self._generate_data_single, in_axes=(None, None, 0))(L, self.sampling, key_new)
        else:
            points = self._generate_data_single(L, self.sampling, self.key)
            points = points[None, ...]
        if self.flatten:
            points = jnp.reshape(points, (points.shape[0], points.shape[1] * points.shape[2], points.shape[3]))
            
        return points