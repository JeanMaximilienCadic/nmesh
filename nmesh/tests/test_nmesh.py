import unittest
from nmesh import NMesh, cfg
import numpy as np


class TestNMeshMethods(unittest.TestCase):
    def test_download_gdrive(self):
        m = NMesh(cfg.gdrive.bull)

    def test_ranges(self):
        m = NMesh(cfg.gdrive.bull)
        bbox = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
        r = m.crop_bounding_box(bbox).ranges()
        self.assertTrue(np.min(r[0]) > np.min(bbox))
        self.assertTrue(np.max(r[1]) < np.max(bbox))


if __name__ == "__main__":
    unittest.main()