import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import cm
class Laser:
    def __init__(self, mu_x, mu_y, z, mode = [0, 0], mode_type = "h", res = 5000, waist = 1, power = 100, chop = True, chop_threshold = 1e-3, wavelen = 1064e-9):
        self.power = power
        self.waist = waist
        self.resolution = res
        self.chop = chop
        self.chop_threshold = chop_threshold
        self.wl = wavelen
        self.xyz = self.make_rays(self.power, self.resolution, mu_x, mu_y, z, mode, mode_type)
        self.x, self.y, self.P = self.set_coords()
        return
    def set_coords(self):
        if self.chop == True:
            x = self.xyz[np.isnan(self.xyz[:, 2]) == False][:, 0]
            y = self.xyz[np.isnan(self.xyz[:, 2]) == False][:, 1]
            P = self.xyz[np.isnan(self.xyz[:, 2]) == False][:, 2]
        else:
            x = self.xyz[:, 0]
            y = self.xyz[:, 1]
            P = self.xyz[:, 2]
        return x, y, P
    def set_power(self, power):
        self.power = power
        return power
    def get_power(self):
        return self.power
    def dy(self, dy = 0):
        self.xyz = np.column_stack([self.xyz[:, 0].flat, (self.xyz[:, 1] + dy).flat, self.xyz[:, 2].flat])
        self.set_coords()
        return
    def w(self, z, n = 1):
        return self.waist * np.sqrt(1 + np.power(z / (np.pi * self.waist * 2 * n / self.wl), 2))
    def make_rays(self, power, res, mu_x, mu_y, z = 0, mode = [0, 0], mode_type = "H"):
        from scipy.special import hermite, genlaguerre
        X, Y = np.meshgrid(np.linspace(-self.waist * 10, self.waist * 10, res), np.linspace(-self.waist * 10, self.waist * 10, res))
        if mode_type.upper() == "h".upper():
            xyz = np.column_stack([X.flat, Y.flat, power * (self.waist / self.w(z)) ** 2 * (hermite(mode[0])(np.sqrt(2) * (np.array(X.flat) - mu_x) / self.w(z)) * np.exp(- (np.array(X.flat) - mu_x) ** 2 / self.w(z) ** 2)) ** 2 * (hermite(mode[1])(np.sqrt(2) * (np.array(Y.flat) - mu_y) / self.w(z)) * np.exp(- (np.array(Y.flat) - mu_y) ** 2 / self.w(z) ** 2)) ** 2])
        if mode_type.upper() == "l".upper():
            xyz = np.column_stack([X.flat, Y.flat, power * (2 * np.sqrt((np.array(X.flat) - mu_x) ** 2 + (np.array(Y.flat) - mu_y) ** 2) ** 2 / self.w(z) ** 2) ** mode[1] * (genlaguerre(mode[0], mode[1])(2 * np.sqrt((np.array(X.flat) - mu_x) ** 2 + (np.array(Y.flat) - mu_y) ** 2) ** 2 / self.w(z) ** 2)) ** 2 * np.cos(mode[1] * np.arctan((np.array(Y.flat) - mu_y) / (np.array(X.flat) - mu_x))) ** 2 * np.exp(-2 * np.sqrt((np.array(X.flat) - mu_x) ** 2 + (np.array(Y.flat) - mu_y) ** 2) ** 2 / self.w(z) ** 2)])
        xyz[:, 2][np.isnan(xyz[:, 2]) == False] = power * xyz[:, 2][np.isnan(xyz[:, 2]) == False] / np.sum(xyz[:, 2][np.isnan(xyz[:, 2]) == False])
        if self.chop == True:
            xyz[np.ix_(np.where(np.abs(xyz[:, 2]) < self.chop_threshold)[0], [2])] = np.nan
        return xyz
    def plot_surface(self, ax = None, figsize = (10, 10), alpha = 1, cmap = cm.coolwarm, antialiased = False, show = False, save = False, name = "", *args, **kwargs):
        if ax == None:
            fig = plt.figure(figsize = figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = plt.gcf()
        ax.plot_surface(*[self.xyz[:, i].reshape(int(np.sqrt(len(self.xyz[:, i]))), int(np.sqrt(len(self.xyz[:, i])))) for i in range(3)], alpha = alpha, cmap = cmap, antialiased = antialiased, *args, **kwargs)
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
            ax.imshow(self.xyz[:, 2].reshape(int(np.sqrt(len(self.xyz[:, 2]))), int(np.sqrt(len(self.xyz[:, 2])))), cmap = cmap, origin = origin, extent = extent)
        else:
            ax.imshow(self.xyz[:, 2].reshape(int(np.sqrt(len(self.xyz[:, 2]))), int(np.sqrt(len(self.xyz[:, 2])))), cmap = cmap, origin = origin, extent = [np.min(self.xyz[:, 0][np.isnan(self.xyz[:, 2]) == False]), np.max(self.xyz[:, 0][np.isnan(self.xyz[:, 2]) == False]), np.min(self.xyz[:, 1][np.isnan(self.xyz[:, 2]) == False]), np.max(self.xyz[:, 1][np.isnan(self.xyz[:, 2]) == False])])
        if save and (name != ""):
            fig.savefig("")
        if show:
            plt.show()
        return fig, ax
    def __len__(self):
        return len(self.xyz[:, 0].flat)