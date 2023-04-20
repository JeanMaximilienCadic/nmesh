from nmesh import NMesh
from miniofs import Object
import trimesh
# m = NMesh("/srv/lake/landing/AI3D/disks/2020/USB6/xv_105552_xf_209393_yv_14624_yf_29244/preparationscan.stl")
m = NMesh(trimesh.load("/srv/lake/landing/AI3D/disks/2020/USB6/xv_105552_xf_209393_yv_14624_yf_29244/preparationscan.stl"))
# m = NMesh("zfs://kiron/landing/disks/2020/USB6/xv_105552_xf_209393_yv_14624_yf_29244/preparationscan.stl")
# m = NMesh(Object("zfs://kiron/landing/disks/2020/USB6/xv_105552_xf_209393_yv_14624_yf_29244/preparationscan.stl").collect())
# m = NMesh("/zfs/kiron/landing/disks/2020/USB6/xv_105552_xf_209393_yv_14624_yf_29244/preparationscan.stl")
# m = NMesh(Object("/zfs/kiron/landing/disks/2020/USB6/xv_105552_xf_209393_yv_14624_yf_29244/preparationscan.stl").collect())
# m = NMesh(Object("/zfs/kiron/landing/disks/2020/USB6/xv_105552_xf_209393_yv_14624_yf_29244/preparationscan.stl").collect())

m.show()