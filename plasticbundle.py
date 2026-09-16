import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import laser
from jax import jit
jax.config.update("jax_enable_x64", True)

class plasticbundle:
    def __init__(self, r, gap, center = [0, 0], res = 1000, orientation = "v"):
        self.r = r
        self.gap = gap
        self.res = res
        self.center = center
        self.orientation = orientation
        self.halves = self.draw_bundle(r, gap, res, center, orientation)
        return
    
    @staticmethod
    @jit(static_argnames=["res", "orientation"])
    def draw_bundle(r, gap, res, center, orientation):
        if orientation.upper() == "v".upper():
            f1 = jnp.array([jnp.append(jnp.full(100, center[0] + gap / 2), r * jnp.sin(jnp.linspace(0, 1, res - 100) * jnp.pi) + gap / 2 + center[0]), jnp.append(jnp.linspace(-(r - center[1]), (r + center[1]), 100), r * jnp.cos(jnp.linspace(0, 1, res - 100) * jnp.pi) + center[1])])
            f2 = jnp.array([jnp.append(jnp.full(100, center[0] - gap / 2), r * jnp.sin(-jnp.linspace(0, 1, res - 100) * jnp.pi) - gap / 2 + center[0]), jnp.append(jnp.linspace(-(r - center[1]), (r + center[1]), 100), r * jnp.cos(jnp.linspace(0, 1, res - 100) * jnp.pi) + center[1])])
        else:
            f1 = jnp.array([jnp.append(jnp.linspace(-(r - center[0]), (r + center[0]), 100), r * jnp.cos(jnp.linspace(0, 1, res - 100) * jnp.pi) + center[0]), jnp.append(jnp.full(100, center[1] + gap / 2), r * jnp.sin(jnp.linspace(0, 1, res - 100) * jnp.pi) + gap / 2 + center[1])])
            f2 = jnp.array([jnp.append(jnp.linspace(-(r - center[0]), (r + center[0]), 100), r * jnp.cos(jnp.linspace(0, 1, res - 100) * jnp.pi) + center[0]), jnp.append(jnp.full(100, center[1] - gap / 2), r * jnp.sin(-jnp.linspace(0, 1, res - 100) * jnp.pi) - gap / 2 + center[1])])
        return jnp.array([f1, f2])
    
    def plot(self, ax = None, figsize = (8, 6), im = [], c = "k"):
        import matplotlib.pyplot as plt
        plt.ioff()
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
        else:
            fig = plt.gcf()
        for h in self.halves:
            ax.plot(h[0], h[1], c = c)
        if len(im) > 0:
            r, c = jnp.where(~jnp.isnan(im.P))
            ax.scatter(im.x[c], im.y[r], c = "b", s = 0.1)
        plt.ion()
        return fig, ax
    
    @jit(static_argnums=(0,2))
    def sum_power(self, l, half = -1):
        P = jnp.where(jnp.isnan(l.P), 0, l.P)
        if self.orientation.upper() == "v".upper():
            half1 = ((l.x - self.center[0] - self.gap / 2) ** 2 + (l.y - self.center[1]) ** 2 < self.r ** 2) & (l.x > self.center[0] + self.gap / 2)
            half2 = ((l.x - self.center[0] + self.gap / 2) ** 2 + (l.y - self.center[1]) ** 2 < self.r ** 2) & (l.x < self.center[0] - self.gap / 2)
        else:
            half1 = ((l.x - self.center[0]) ** 2 + (l.y - self.center[1] - self.gap / 2) ** 2 < self.r ** 2) & (l.y > self.center[1] + self.gap / 2)
            half2 = ((l.x - self.center[0]) ** 2 + (l.y - self.center[1] + self.gap / 2) ** 2 < self.r ** 2) & (l.y < self.center[1] - self.gap / 2)
            
        half1_P = jnp.sum(P * half1)
        half2_P = jnp.sum(P * half2)
        if half == 0:
            return half1_P
        elif half == 1:
            return half2_P
        else:
            return half1_P + half2_P
    @jit(static_argnums=0)
    def diff_power(self, l):
        return self.sum_power(l, 0) - self.sum_power(l, 1)

    def _tree_flatten(self):
        children = (self.halves,)
        aux = {
            "r": self.r, "gap": self.gap, "center": self.center, "res": self.res, "orientation": self.orientation
        }
        return (children, aux)
    @classmethod
    def _tree_unflatten(cls, aux, children):
        obj = cls.__new__(cls)
        obj.r = aux["r"]
        obj.gap = aux["gap"]
        obj.center = aux["center"]
        obj.res = aux["res"]
        obj.orientation = aux["orientation"]
        obj.halves = children
        return obj

if __name__ == "__main__":
    jax.clear_caches()
    l = jax.block_until_ready(laser.laser(0, 0, 0, waist = 1e-4, power = 102, res = 10000, mode = [0, 1], mode_type = 0, chop = True, chop_threshold = 1e-5))
    p = plasticbundle(1e-3, 1e-4, orientation = "h")
    p1 = p.diff_power(l)
    l = l.dy(1e-9)
    p2 = p.diff_power(l)
    print(p1 - p2)