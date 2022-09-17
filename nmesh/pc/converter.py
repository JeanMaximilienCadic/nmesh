from nmesh import NMesh
import numpy as np
import os
from gnutools.fs import name


class Converter:
    def __init__(self, folder, resolution=64, crop_size=30):
        self._vertices = {}
        self._resolution = resolution
        self._folder_name=name(folder)
        hcs = crop_size//2
        crop_array = np.array([hcs, hcs, hcs])
        for id in ["x", "y"]:
            m = NMesh(f"{folder}/{id}.ply")
            m.crop_bounding_box([-crop_array, crop_array])
            vertices = m.vertices
            vertices += crop_array
            vertices /= crop_size
            vertices *= resolution
            vertices = np.array(vertices, dtype=int)
            vertices = np.unique(vertices, axis=0)
            self._vertices[id] = vertices

    def export(self, root_output):
        output_dir = f"{root_output}/{self._resolution}/{self._folder_name}"
        os.makedirs(output_dir, exist_ok= True)
        [np.savetxt(f"{output_dir}/{id}.xyz", vertices) for (id, vertices) in self._vertices.items()]


