![nmesh Logo](imgs/nmesh.jpg)

--------------------------------------------------------------------------------

NMesh is a Python package that provides two high-level features:
- A simple Mesh processor
- A list of tool to convert mesh files into point cloud

You can reuse your favorite Python packages such as NumPy, SciPy and Cython to extend ZakuroCache integration.


## NMesh modules

At a granular level, NMesh is a library that consists of the following components:

| Component | Description |
| ---- | --- |
| **nmesh** | Contains the implementation of NMesh |
| **nmesh.core** | Contain the functions executed by the library. |
| **nmesh.cp** | Processor for the point cloud|

## Installation

### Docker
To build the image with docker-compose
```
sh docker.sh
```

### Local
```
python setup.py install
```
