import time
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from matplotlib.image import BboxImage
from matplotlib import cm

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

class FiberBundle:  
    def __init__(self, r, core = 2.5e-5, cladding = 0):
        self.r = r
        self.core = core
        self.cladding = cladding
        self.fr = (self.core + self.cladding)
        self.fiber_rings = np.r_[self.fr, np.linspace(2 * self.fr, (self.r - self.fr) - (self.r - self.fr) % (2 * self.fr), int((self.r - self.fr) / (2 * self.fr)))]
        # theta position, ring radius, ring number, half
        self.fibers = self.make_bundle()
        self.centers = np.array(Parallel(n_jobs = -1, backend = "threading")(delayed(self.set_fiber_pos)(self.fr, f[0], f[1], f[2], f[3]) for i in range(len(self.fibers)) for f in self.fibers[i]))
        self.thetas = np.array([f[0] for i in range(len(self.fibers)) for f in self.fibers[i]])
        self.ring_radius = np.array([f[1] for i in range(len(self.fibers)) for f in self.fibers[i]])
        self.ring_number = np.array([int(f[2]) for i in range(len(self.fibers)) for f in self.fibers[i]])
        self.half = np.array([int(f[3]) for i in range(len(self.fibers)) for f in self.fibers[i]])
        self.count = len(self.centers)
        self.drawings = np.array(Parallel(n_jobs = -1, backend = "threading")(delayed(self.draw_fiber)(self.centers[i]) for i in range(len(self.centers))))
        return
    
    def reset(self):
        self.fiber_rings = np.r_[self.fr, np.linspace(2 * self.fr, (self.r - self.fr) - (self.r - self.fr) % (2 * self.fr), int((self.r - self.fr) / (2 * self.fr)))]
        self.fibers = self.make_bundle()
        self.centers = np.array(Parallel(n_jobs = -1, backend = "threading")(delayed(self.set_fiber_pos)(self.fr, f[0], f[1], f[2], f[3]) for i in range(len(self.fibers)) for f in self.fibers[i]))
        self.thetas = np.array([f[0] for i in range(len(self.fibers)) for f in self.fibers[i]])
        self.ring_radius = np.array([f[1] for i in range(len(self.fibers)) for f in self.fibers[i]])
        self.ring_number = np.array([int(f[2]) for i in range(len(self.fibers)) for f in self.fibers[i]])
        self.half = np.array([int(f[3]) for i in range(len(self.fibers)) for f in self.fibers[i]])
        self.count = len(self.centers)
        self.drawings = np.array(Parallel(n_jobs = -1, backend = "threading")(delayed(self.draw_fiber)(self.centers[i]) for i in range(len(self.centers))))
        return
    
    def drop_rings(self, i):
        if type(i) == int:
            idx = np.where(self.ring_number == i)[0]
            self.fibers = np.delete(self.fibers, i)
            self.fiber_rings = np.delete(self.fiber_rings, i, axis = 0)
            self.centers = np.delete(self.centers, idx, axis = 0)
            self.thetas = np.delete(self.thetas, idx, axis = 0)
            self.ring_radius = np.delete(self.ring_radius, idx, axis = 0)
            self.ring_number = np.delete(self.ring_number, idx, axis = 0)
            self.half = np.delete(self.half, idx, axis = 0)
            self.drawings = np.delete(self.drawings, idx, axis = 0)
            self.count = len(self.centers)
        if type(i) == tuple:
            for j in reversed(range(*i)):
                self.drop_rings(j)
        return None
    
    def set_fiber_pos(self, r, theta, ring, i, half):
        if half == 0:
            if i == 0:
                return np.array([r * np.cos(theta - np.pi / 2), r * np.sin(theta - np.pi / 2)])
            else:
                return np.array([ring * np.cos(theta - np.pi), ring * np.sin(theta - np.pi) - r])
        else:
            if i == 0:
                return np.array([r * np.cos(theta - np.pi / 2), r * np.sin(theta - np.pi / 2)])
            else:
                return np.array([ring * np.cos(theta - np.pi), ring * np.sin(theta - np.pi) + r])

    def draw_fiber(self, center = [0, 0]): 
        """
        
        draw_fiber --- draws the x, y points for the fiber to be that it can be plotted.
        
        :return: x and y arrays for the circumference of the fiber being plotted.
        
        """
        ts = np.linspace(0, 2 * np.pi, 100) # thetas for drawing the individual fiber
        return np.array([center[0] + self.fr * np.cos(ts), center[1] + self.fr * np.sin(ts)])

    def make_bundle(self):
        return np.array(Parallel(n_jobs = -1)(delayed(self.draw_ring)(ring, i) for i, ring in enumerate(self.fiber_rings)), dtype = object)

    def draw_ring(self, ring, i):
        ts = np.append(np.linspace(0, np.pi, int(np.pi / ((2 * ring * np.arccos(1 - (self.fr ** 2) / (2 * ring ** 2))) / ring))), np.linspace(np.pi, 2 * np.pi, int(np.pi / ((2 * ring * np.arccos(1 - (self.fr ** 2) / (2 * ring ** 2))) / ring))))
        f1 = []
        Parallel(n_jobs = -1)(delayed(f1.append)([t, ring, int(i), 0]) for t in ts[:int(len(ts) / 2)])
        f2 = []
        Parallel(n_jobs = -1)(delayed(f2.append)([t, ring, int(i), 1]) for t in ts[int(len(ts) / 2):])
        return np.array(f1 + f2)

    def plot(self, fibers = True, centers = False, ax = None, figsize = (10, 10), scatter_size = 1, im = [], lc = "b", show = False, save = False, name = "", *args, **kwargs):
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
        else:
            fig = plt.gcf()
        plt.ioff()
        for i in range(len(self.drawings)):
            if fibers:
                ax.plot(self.drawings[i][0], self.drawings[i][1], zorder = 100, *args, **kwargs)
            if centers:
                ax.scatter(*self.centers[i], s = scatter_size, zorder = 100, *args, **kwargs)
        if len(im) > 0:
            ax.scatter(im.x, im.y, c = lc)
        plt.ion()
        ax.set_xlim((-self.r - self.r * .1), (self.r + self.r * .1))
        ax.set_ylim((-self.r - self.r * .1), (self.r + self.r * .1))
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Fiber Bundle")
        if save and name != "":
            plt.savefig(name)
        if show:
            plt.show()
        return fig, ax
    
    def sum_power(self, l, half = -1):
        if half == -1:
            return sum(Parallel(n_jobs = -1, backend = "threading")(delayed(sum)(l.P[i]) for i in Parallel(n_jobs = -1, backend = "threading")(delayed(np.where)((l.x - c[0]) ** 2 + (l.y - c[1]) ** 2 < ((self.fr - self.cladding) ** 2)) for c in self.centers)))
        else:
            return sum(Parallel(n_jobs = -1, backend = "threading")(delayed(sum)(l.P[i]) for i in Parallel(n_jobs = -1, backend = "threading")(delayed(np.where)((l.x - c[0]) ** 2 + (l.y - c[1]) ** 2 < ((self.fr - self.cladding) ** 2)) for c in self.centers[np.where(self.half == half)[0]])))
    
    def diff_power(self, l):
        return sum(Parallel(n_jobs = -1, backend = "threading")(delayed(sum)(l.P[i]) for i in Parallel(n_jobs = -1, backend = "threading")(delayed(np.where)((l.x - c[0]) ** 2 + (l.y - c[1]) ** 2 < ((self.fr - self.cladding) ** 2)) for c in self.centers[np.where(self.half == 0)[0]]))) - sum(Parallel(n_jobs = -1, backend = "threading")(delayed(sum)(l.P[i]) for i in Parallel(n_jobs = -1, backend = "threading")(delayed(np.where)((l.x - c[0]) ** 2 + (l.y - c[1]) ** 2 < ((self.fr - self.cladding) ** 2)) for c in self.centers[np.where(self.half == 1)[0]])))