import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import laser
from jax import jit
import plasticbundle as pbndl
import numpy as np
import os
import matplotlib.animation as animation
from quadax import quadgk
from laser import jax_hermite
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"


def I_x(x, mode, waist, w, P_total, mux=0.0):
    n = mode[0]
    norm_n = 1.0 / jnp.sqrt((2.0**n) * jax.scipy.special.factorial(n))
    
    norm_factor = (2.0 / (jnp.pi * w**2))**0.25
    field_x = norm_factor * norm_n * jax_hermite(n, jnp.sqrt(2) * (x - mux) / w) * jnp.exp(-(x - mux)**2 / w**2)
    
    return jnp.sqrt(P_total) * (field_x**2)

def I_y(y, mode, waist, w, P_total, muy=0.0):
    m = mode[1]
    norm_m = 1.0 / jnp.sqrt((2.0 ** m) * jax.scipy.special.factorial(m))
    
    norm_factor = (2.0 / (jnp.pi * w ** 2)) ** 0.25
    field_y = norm_factor * norm_m * jax_hermite(m, jnp.sqrt(2) * (y - muy) / w) * jnp.exp(-(y - muy)**2 / w**2)
    
    return jnp.sqrt(P_total) * (field_y ** 2)

def x_integ(bounds, mode, waist, w, P_total, mux = 0.0):
    integ, info = quadgk(I_x, bounds, args=(mode, waist, w, P_total, mux))
    return integ

def y_integ(bounds, mode, waist, w, P_total, muy = 0.0):
    integ, info = quadgk(I_y, bounds, args=(mode, waist, w, P_total, muy))
    return integ


def main():
    # modes = jnp.array([jnp.array([0, 0], dtype = jnp.int32), jnp.array([0, 1], dtype = jnp.int32)])
    # targ_power = 5.0
    # mux = 0
    # amp = [0.44e-9, 0.46e-9, 0.40e-9, 0.35e-9, 0.30e-9, 0.24e-9, 0.19e-9, 0.14e-9, 0.09e-9, 0.05e-9]
    # volts = jnp.linspace(1, 10, 10)[::-1]
    # waists = jnp.linspace(2e-4, 5e-3, 100)
    # gaps = jnp.logspace(0, 1e-3, 100)
    # t = jnp.linspace(0, 5 / 108000, 1000)
    # z = 0
    # d = 0
    # for mode in modes:
    #     for i in range(len(amp)):
    #         muy = amp[i] * jnp.sin((2 * jnp.pi * 108000) * t)
    #         for j in range(len(gaps)):
    #             for k in range(len(waists)):
    #                 d += 1
    #                 w = waists[k] * jnp.sqrt(1 + jnp.power(z / (jnp.pi * waists[k] * 2 * 1 / 1064e-9), 2))
    #                 r = 5 * waist
    #                 def diff(mu):
    #                     return x_integ(jnp.array([-r, r]), mode, waists[k], w, targ_power, mux = mux) * y_integ(jnp.array([gaps[j] / 2, r]), mode, waists[k], w, targ_power, muy = mu) - x_integ(jnp.array([-r, r]), mode, waists[k], w, targ_power, mux = mux) * y_integ(jnp.array([-r, -gaps[j] / 2]), mode, waists[k], w, targ_power, muy = mu)
    #                 df = pd.DataFrame({"Mode": f"HG_{mode[0]}{mode[1]}", "Osc. Ampl.": amp[i], "Gap Size": gaps[j], "Beam Waist": waists[k], "Time": np.array(t), "Power Diff.": np.array(jax.vmap(diff)(muy))}).to_csv("data/20260817/data" + str(d) + ".csv", index = False)


    mode = [0, 0]
    targ_power = 8
    mux = 0
    amp = 0.46e-9 * np.sin(np.pi / 2 - 10 * np.pi  / 180)
    gap = 0.94e-3
    z = 0
    waist = 1e-3
    t = np.arange(0, 1048576) * (1 / 25e6)
    muy = amp * jnp.sin((2 * jnp.pi * 109000) * t)
    w = waist * jnp.sqrt(1 + jnp.power(z / (jnp.pi * waist * 2 * 1 / 1064e-9), 2))
    r = 5 * w
    # def diff(mu):
    #     return x_integ(jnp.array([-r, r]), mode, waist, w, targ_power, mux = mux) * y_integ(jnp.array([gap / 2, r]), mode, waist, w, targ_power, muy = mu) - x_integ(jnp.array([-r, r]), mode, waist, w, targ_power, mux = mux) * y_integ(jnp.array([-r, -gap / 2]), mode, waist, w, targ_power, muy = mu)
    def lsum(mu, r, gap, mode, waist, w, targ_power, mux):
        return x_integ(jnp.array([-r, r]), mode, waist, w, targ_power, mux = mux) * y_integ(jnp.array([gap / 2, r]), mode, waist, w, targ_power, muy = mu) + x_integ(jnp.array([-r, r]), mode, waist, w, targ_power, mux = mux) * y_integ(jnp.array([-r, -gap / 2]), mode, waist, w, targ_power, muy = mu)
    def diffmap(mu, r, gap, mode, waist, w, targ_power, mux):
        def diff(mu):
            p = targ_power / (lsum(0, r, gap, mode, waist, w, targ_power, mux) / targ_power)
            # print(lsum(0, r, gap, mode, waist, w, p, mux))
            return x_integ(jnp.array([-r, r]), mode, waist, w, p, mux = mux) * y_integ(jnp.array([gap / 2, r]), mode, waist, w, p, muy = mu) - x_integ(jnp.array([-r, r]), mode, waist, w, p, mux = mux) * y_integ(jnp.array([-r, -gap / 2]), mode, waist, w, p, muy = mu)
        return jax.vmap(diff)(mu)
    df = pd.DataFrame({"Mode": f"HG_{mode[0]}{mode[1]}", "Osc. Ampl.": amp, "Gap Size": gap, "Beam Waist": waist, "Time": np.array(t), "Power Diff.": np.array(diffmap(muy, r, gap, mode, waist, w, targ_power, mux))}).to_csv("data/20260910/data1.csv", index = False)
                    

if __name__ == "__main__":
    jax.clear_caches()
    main()
# fig = plt.figure(figsize = (12, 6))
# ax = fig.add_subplot(111, projection = "3d")
# mesh1 = ax.plot_surface(x[jnp.newaxis, :], y1[:, jnp.newaxis], I_x(x[jnp.newaxis, :], mode, waist, w, targ_power, mux[0]) * I_y(y1[:, jnp.newaxis], mode, waist, w, targ_power, muy[0]), color = "r")
# mesh2 = ax.plot_surface(x[jnp.newaxis, :], y2[:, jnp.newaxis], I_x(x[jnp.newaxis, :], mode, waist, w, targ_power, mux[0]) * I_y(y2[:, jnp.newaxis], mode, waist, w, targ_power, muy[0]), color = "b")

# def ani(i):
#     ax.clear()
#     mesh1 = ax.plot_surface(x[jnp.newaxis, :], y1[:, jnp.newaxis], I_x(x[jnp.newaxis, :], mode, waist, w, targ_power, mux[i]) * I_y(y1[:, jnp.newaxis], mode, waist, w, targ_power, muy[i]))
#     mesh2 = ax.plot_surface(x[jnp.newaxis, :], y2[:, jnp.newaxis], I_x(x[jnp.newaxis, :], mode, waist, w, targ_power, mux[i]) * I_y(y2[:, jnp.newaxis], mode, waist, w, targ_power, muy[i]))
#     return mesh1, mesh2

# a = animation.FuncAnimation(
#     fig = fig, 
#     func=ani, 
#     frames=range(0, len(mux), 20), 
#     interval=30, 
#     blit=False
# )
# plt.show()