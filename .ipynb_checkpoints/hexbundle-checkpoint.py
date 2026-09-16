import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from jax import jit

class HexagonalBundle:
    def __init__(self, R_max, r_core, r_clad):
        """
        R_max: Radius of the main circular enclosure boundary.
        r_hex: Outer radius of an individual hexagon (center to vertex distance).
        """
        self.R_max = R_max
        self.r_hex = r_core + r_clad
        self.r_core = r_core
        self.r_clad = r_clad
        self.centers = self.generate_packed_centers(self.r_hex, self.R_max)

    @staticmethod
    @jit(static_argnames=["r_hex", "R_max"])
    def generate_packed_centers(r_hex, R_max):
        row_spacing = r_hex * jnp.sqrt(3)

        max_rings = jnp.ceil(R_max / (r_hex * 0.75)) + 2
        
        centers = []
        
        for q in range(-max_rings.astype(jnp.int32), max_rings.astype(jnp.int32)):
            for r in range(-max_rings.astype(jnp.int32), max_rings.astype(jnp.int32)):
                x_c = r_hex * (3.0 / 2.0 * q)
                y_c = r_hex * (jnp.sqrt(3.0) * r + jnp.sqrt(3.0) / 2.0 * q)
                
                center_dist = jnp.sqrt(x_c ** 2 + y_c ** 2)
                
                if center_dist + r_hex <= R_max:
                    centers.append([x_c, y_c])
                    
        return jnp.array(centers)

    def plot(self, figsize=(8, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        
        boundary_circle = plt.Circle((0, 0), self.R_max, color='r', fill=False, linestyle='--', linewidth=2, label="Boundary")
        ax.add_patch(boundary_circle)
        
        for center in self.centers:
            hex_patch = RegularPolygon(
                (center[0], center[1]), 
                numVertices=6, 
                radius=self.r_hex, 
                orientation=np.pi / 6, 
                facecolor='lightgray', 
                edgecolor='black', 
                linewidth=1.2,
                alpha=0.7
            )
            ax.add_patch(hex_patch)
            hex_patch = RegularPolygon(
                (center[0], center[1]), 
                numVertices=6, 
                radius=self.r_core, 
                orientation=np.pi / 6, 
                facecolor='red', 
                edgecolor='black', 
                linewidth=1.2,
                alpha=0.7
            )
            ax.add_patch(hex_patch)
            
        padding = self.r_hex * 2
        ax.set_xlim(-self.R_max - padding, self.R_max + padding)
        ax.set_ylim(-self.R_max - padding, self.R_max + padding)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.3)
        plt.title(f"Total Hexagons: {len(self.centers)}")
        
        return fig, ax

    def _tree_flatten(self):
        children = (self.centers, self.r_hex)
        aux = {"R_max": self.R_max, "r_core": self.r_core, "r_clad": self.r_clad}
        return (children, aux)
    @classmethod
    def _tree_unflatten(cls, aux, children):
        obj = cls.__new__(cls)
        obj.R_max = aux["R_max"]
        obj.r_core = aux["r_core"]
        obj.r_clad = aux["r_clad"]
        obj.centers, obj.r_hex = children
        return obj

if __name__ == "__main__":
    bundle = HexagonalBundle(R_max=2e-3, r_core = 220e-6, r_clad = 20e-6)
    fig, ax = bundle.plot()
    plt.show()