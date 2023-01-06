import unittest

import numpy as np

from nmesh import NMesh, cfg


class TestNMeshMethods(unittest.TestCase):
    def test_download_gdrive(self):
        NMesh(cfg.gdrive.bull)

    def test_ranges(self):
        m = NMesh(cfg.gdrive.bull)
        bbox = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
        r = m.crop_bounding_box(bbox).ranges()
        self.assertTrue(np.min(r[0]) > np.min(bbox))
        self.assertTrue(np.max(r[1]) < np.max(bbox))

    def test_components(self):
        m = NMesh(cfg.gdrive.bull)
        cmpts = m.components()
        self.assertTrue(len(cmpts) >= 0)


if __name__ == "__main__":
    unittest.main()
