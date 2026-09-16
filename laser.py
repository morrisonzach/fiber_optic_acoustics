import os
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
import matplotlib.pyplot as plt
from matplotlib import cm
from jax import jit
import jax.numpy as jnp
import jax
from jax import tree_util
jax.config.update("jax_enable_x64", True)


@jit
def jax_hermite(n, x):
    def body_fun(i, val):
        h_prev, h_curr = val
        return h_curr, 2 * x * h_curr - 2 * i * h_prev
    _, hn = jax.lax.fori_loop(0, n, body_fun, (jnp.zeros_like(x), jnp.ones_like(x)))
    return hn

def adaptive_hermite_linspace(waist, start, stop, res, mode, density = 0.05, w = 1):
    lin = jnp.linspace(start, stop, res)

    Ex = jax_hermite(mode[0], jnp.sqrt(2) * lin / w) * jnp.exp(-lin ** 2 / w ** 2)
    Ix = Ex ** 2
    Ix = Ix + density * jnp.max(Ix)

    d = 2 * waist * (3 + jnp.sqrt(jnp.max(jnp.array(mode))) * 1.5) / (res - 1)
    cdfx = jnp.cumsum(Ix) * d
    cdf_normx = cdfx / cdfx[-1]
    
    Ey = jax_hermite(mode[1], jnp.sqrt(2) * lin / w) * jnp.exp(-lin ** 2 / w ** 2)
    Iy = Ey ** 2
    Iy = Iy + density * jnp.max(Iy)
    
    cdfy = jnp.cumsum(Iy) * d
    cdf_normy = cdfy / cdfy[-1]

    targ = jnp.linspace(0, 1, res)
    x = jnp.interp(targ, cdf_normx, lin)
    y = jnp.interp(targ, cdf_normy, lin)

    return x, y
# @jit
# def adaptive_laguerre_linspace(start, stop, res, mode, w = 1):
#     fine_res = res * 10
#     x_fine = jnp.linspace(start, stop, fine_res)
#     (jax_genlaguerre(mode[0], mode[1], 2 * r ** 2 / w ** 2)) ** 2 * jnp.cos(mode[1] * jnp.arctan(y / x)) ** 2 * jnp.exp(-2 * r ** 2 / w ** 2)
#     E = jax_genlaguerre(mode[0], mode[1], 
#     return

@jit
def jax_genlaguerre(n, alpha, x):
    init_state = (0, jnp.zeros_like(x), jnp.ones_like(x))
    
    def body_fun(state):
        i, l_prev, l_curr = state
        l_next = ((2 * i + 1 + alpha - x) * l_curr - (i + alpha) * l_prev) / (i + 1)
        return (i + 1, l_curr, l_next)
    
    _, _, ln = jax.lax.while_loop(
        lambda state: state[0] < n,
        body_fun,
        init_state
    )
    return ln

class laser:
    def __init__(self, mu_x = 0, mu_y = 0, z = 0, mode = [0, 0], mode_type = 0, res = 5000, waist = 1, power = 100, chop = True, chop_threshold = 1e-3, wavelen = 1064e-9, density = 0.05):
        self.power = power
        self.waist = waist
        self.resolution = res
        self.chop = chop
        self.chop_threshold = chop_threshold
        self.wl = wavelen
        self.mode = mode
        self.mode_type = mode_type
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.z = z
        w = self.w(z)
        self.x, self.y, self.P = self.make_rays(tuple(mode), waist, res, mu_x, mu_y, mode_type, power, chop, chop_threshold, w, density)
        return
    @jit
    def dy(self, dy = 0):
        y = self.y.at[:].add(dy)
        children, aux = self._tree_flatten()
        new_children = (self.x, y, self.P)
        return self._tree_unflatten(aux, new_children)
    def w(self, z, n = 1):
        return self.waist * jnp.sqrt(1 + jnp.power(z / (jnp.pi * self.waist * 2 * n / self.wl), 2))
    @jit
    def chop_laser(self, p, chop_threshold):
        return jnp.where(jnp.abs(p) < chop_threshold, jnp.nan, p) 
    @staticmethod
    @jit(static_argnames=["res", "mt", "chop", "mode"])
    def make_rays(mode, waist, res, mu_x, mu_y, mt, power, chop, chop_threshold, w, density):
        if mt == 0:
            x, y = adaptive_hermite_linspace(waist, -waist * (3 + jnp.sqrt(jnp.max(jnp.array(mode))) * 1.5), waist * (3 + jnp.sqrt(jnp.max(jnp.array(mode))) * 1.5), res, mode, density, w)
        x = (x - mu_x)[jnp.newaxis, :]
        y = (y - mu_y)[:, jnp.newaxis]
        if mt == 0:
            x_part = (jax_hermite(mode[0], jnp.sqrt(2) * x / w) * jnp.exp(-x ** 2 / w ** 2)) ** 2
            y_part = (jax_hermite(mode[1], jnp.sqrt(2) * y / w) * jnp.exp(-y ** 2 / w ** 2)) ** 2
            p = power * (waist / w) ** 2
            P =  p * x_part * y_part
            P = power * P / jnp.sum(P)
        if mt == 1:
            r = jnp.sqrt(x ** 2 + y ** 2)
            p = power * (2 * r ** 2 / self.w() ** 2) ** mode[1]
            P = p * (jax_genlaguerre(mode[0], mode[1], 2 * r ** 2 / w ** 2)) ** 2 * jnp.cos(mode[1] * jnp.arctan(y / x)) ** 2 * jnp.exp(-2 * r ** 2 / w ** 2)
            P = power * P / jnp.sum(P)
        if chop == True:
            P = jnp.where(jnp.abs(P) < chop_threshold, jnp.nan, P)
        return x[0, :], y[:, 0], P
    def plot_surface(self, ax = None, figsize = (10, 10), alpha = 1, cmap = cm.coolwarm, antialiased = False, show = False, save = False, name = "", *args, **kwargs):
        if ax == None:
            fig = plt.figure(figsize = figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = plt.gcf()
        ax.plot_surface(self.x[jnp.newaxis, :], self.y[:, jnp.newaxis], self.P, alpha = alpha, cmap = cmap, antialiased = antialiased, *args, **kwargs)
        if save and (name != ""):
            fig.savefig("")
        if show:
            plt.show()
        return fig, ax
    def imshow(self, ax = None, dimensions = (1, 1), figsize = (10, 10), cmap = cm.coolwarm, save = False, origin = "lower", extent = [],  name = "", show = False):
        if ax == None:
            fig, ax = plt.subplots(*dimensions, figsize = figsize)
        else:
            fig = plt.gcf()
        if extent != []:
            ax.pcolormesh(self.x, self.y, self.P, cmap = cmap, origin = origin, extent = extent)
        else:
            mask = ~jnp.isnan(self.P)
            val_x = jnp.any(mask, axis = 0)
            val_y = jnp.any(mask, axis = 1)

            xmin = float(jnp.where(jnp.any(val_x), jnp.min(self.x[val_x]), jnp.min(self.x)))
            xmax = float(jnp.where(jnp.any(val_x), jnp.max(self.x[val_x]), jnp.max(self.x)))
            ymin = float(jnp.where(jnp.any(val_y), jnp.min(self.y[val_y]), jnp.min(self.y)))
            ymax = float(jnp.where(jnp.any(val_y), jnp.max(self.y[val_y]), jnp.max(self.y)))
            ax.imshow(self.P, cmap = cmap, origin = origin, extent = [xmin, xmax, ymin, ymax])
        if save and (name != ""):
            fig.savefig("")
        if show:
            plt.show()
        return fig, ax
    def __len__(self):
        return len(self.x)
    def _tree_flatten(self):
        children = (self.x, self.y, self.P)
        aux = {
            "power": self.power, "waist": self.waist, "res": self.resolution, 
            "chop": self.chop, "chop_threshold": self.chop_threshold, 
            "wavelen": self.wl, "mu_x": self.mu_x, "mu_y": self.mu_y, 
            "z": self.z, "mode": self.mode, "mode_type": self.mode_type
        }
        return (children, aux)
    @classmethod
    def _tree_unflatten(cls, aux, children):
        obj = cls.__new__(cls)
        obj.power = aux["power"]
        obj.waist = aux["waist"]
        obj.resolution = aux["res"]
        obj.chop = aux["chop"]
        obj.chop_threshold = aux["chop_threshold"]
        obj.wl = aux["wavelen"]
        obj.mode = aux["mode"]
        obj.mode_type = aux["mode_type"]
        obj.mu_x = aux["mu_x"]
        obj.mu_y = aux["mu_y"]
        obj.z = aux["z"]
        obj.x, obj.y, obj.P = children
        return obj

tree_util.register_pytree_node(laser, laser._tree_flatten, laser._tree_unflatten)

### BENCHMARKING FUNCTIONS

def laser_generation_bench(n, x, y, z, m, mt, r, w, p, c, ct, wl):
    jax.clear_caches()
    import time
    import old_laser
    print("warming up by generating an initial laser...")
    warmup = jax.block_until_ready(laser(x, y, z, m, mt, r, w, p, c, ct, wl))
    print("warmed up. starting benchmark...")
    tot_time = 0
    for i in range(n):
        s = time.perf_counter()
        l = jax.block_until_ready(laser(x, y, z, m, mt, r, w, p, c, ct, wl))
        e = time.perf_counter()
        t = e - s
        tot_time += t
        # print(f"\rIteration {i + 1}: {t:.2f} seconds")
    print("\n" + "=" * 30)
    print(f"\nTotal loop execution time: {tot_time:.4f} seconds")
    print(f"Average time per profile:  {tot_time / n:.4f} seconds")
    print("\n" + "=" * 30)

    # print("\nwarming up by generating an initial Laser...")
    # warmup = old_laser.Laser(x, y, z, m, "h", r, w, p, c, ct, wl)
    # print("warmed up. starting benchmark...")
    # for i in range(n):
    #     s = time.perf_counter()
    #     l = old_laser.Laser(x, y, z, m, "h", r, w, p, c, ct, wl)
    #     e = time.perf_counter()
    #     t = e - s
    #     tot_time += t
    #     # print(f"\rIteration {i + 1}: {t:.2f} seconds")
    # print("\n" + "=" * 30)
    # print(f"\nTotal loop execution time: {tot_time:.4f} seconds")
    # print(f"Average time per profile:  {tot_time / n:.4f} seconds")
    # print("\n" + "=" * 30)

def laser_dy_bench(n, x, y, z, m, mt, r, w, p, c, ct, wl):
    jax.clear_caches()
    import time
    import old_laser
    print("warming up by generating an initial laser...")
    warmup = jax.block_until_ready(laser(x, y, z, m, mt, r, w, p, c, ct, wl))
    print("warmed up. starting benchmark...")
    tot_time = 0
    l = jax.block_until_ready(laser(x, y, z, m, mt, r, w, p, c, ct, wl))
    for i in range(n):
        s = time.perf_counter()
        jax.block_until_ready(l.dy(1e-6))
        e = time.perf_counter()
        t = e - s
        tot_time += t
    print("\n" + "=" * 30)
    print(f"\nlaser total loop execution time: {tot_time:.4f} seconds")
    print(f"Average time per profile:  {tot_time / n:.4f} seconds")
    print("\n" + "=" * 30)

    # print("warming up by generating an initial Laser...")
    # warmup = jax.block_until_ready(laser(x, y, z, m, mt, r, w, p, c, ct, wl))
    # print("warmed up. starting benchmark...")
    # L = old_laser.Laser(x, y, z, m, "h", r, w, p, c, ct, wl)
    # tot_time = 0
    # for i in range(n):
    #     s = time.perf_counter()
    #     L.dy(1e-6)
    #     e = time.perf_counter()
    #     t = e - s
    #     tot_time += t
    # print("\n" + "=" * 30)
    # print(f"\nLaser total loop execution time: {tot_time:.4f} seconds")
    # print(f"Average time per profile:  {tot_time / n:.4f} seconds")
    # print("\n" + "=" * 30)

if __name__ == "__main__":
    jax.clear_caches()
    laser_generation_bench(1000, 0, 0, 0, [0, 0], 0, 10000, 1e-3, 100, True, 1e-5, 1064e-9)
    laser_dy_bench(1000, 0, 0, 0, [0, 0], 0, 10000, 1e-3, 100, True, 1e-5, 1064e-9)