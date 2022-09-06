<h1 align="center">
  <br>
  <a href=https://drive.google.com/uc?id=1lp-TSsLdTPBppssMm_f0YORERVSbDf9J"><img src="https://drive.google.com/uc?id=1lp-TSsLdTPBppssMm_f0YORERVSbDf9J"></a>
  <br>
  NMesh
  <br>
</h1>


<p align="center">
  <a href="#modules">Modules</a> •
  <a href="#code-design">Code Design</a> •
  <a href="#code-structure">Code Structure</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#docker">Docker</a> •
  <a href="#pythonenv">PythonEnv</a> •
  <a href="#resssource">Ressources</a> •
</p>


NMesh is a Python package that provides two high-level features:
- A simple Mesh processor
- A list of tool to convert mesh files into point cloud

You can reuse your favorite Python packages such as NumPy, SciPy and Cython to extend ZakuroCache integration.


## Modules

At a granular level, NMesh is a library that consists of the following components:

| Component | Description |
| ---- | --- |
| **nmesh** | Contains the implementation of NMesh |
| **nmesh.core** | Contain the functions executed by the library. |
| **nmesh.cp** | Processor for the point cloud|


## Code design
* We recommend using Docker for dev and production. Therefore we encourage its usage all other the repo.
* We have `vanilla` and `sandbox` environment. 
  * `Vanilla` refers to a prebuilt docker image that already contains system dependencies.
  * `Sandbox` referes a predbuilt docker image that contains the code of this repo.
* Semantic versioning https://semver.org/ . We commit fix to `a.b.x`, features to `a.x.c` and stable release (master) to `x.b.c`. 
* PR are done to `dev` and reviewed for additional features. This should only be reviewed by the engineers in the team.
* PR are done to `master` for official (internal) release of the codes. This should be reviewed by the maximum number of engineers.   
* The ETL jobs are scatter accross sequential refinement of the data `landing/bronze/silver/gold` 
* Modules and scripts: Any piece of code that can of use for parent module modules should be moved at a higher level. 
  * eg: `functional.py` contains common funtions for `etl.bronze` and `etl.silver`
```
...
├── etl
│   ├── bronze
│   │   ├── __init__.py
│   │   └── __main__.py
│   ├── functional.py
│   ├── __init__.py
│   └── landing
│       ├── __init__.py
│       └── __main__.py
├── functional.py
├── __init__.py
...
```
* Modules should ideally contain a `__main__.py` that demo an exeution of the module
  * `etl/bronze/__main__.py` describes an etl job for the creation of the bronze parition
  * `trainer/__main__.py` describes the training pipeline



# Code structure
```python
from setuptools import setup
from nmesh import __version__
setup(
    name='nmesh',
    version=__version__,
    packages=[
        "nmesh",
        "nmesh.core",
        "nmesh.cp"
    ],
    url='https://github.com/JeanMaximilienCadic/nmesh',
    include_package_data=True,
    package_data={"": ["*.yml"]},
    long_description="".join(open("README.md", "r").readlines()),
    long_description_content_type='text/markdown',
    license='MIT',
    author='Jean Maximilien Cadic',
    python_requires='>=3.6',
    install_requires=[r.rsplit()[0] for r in open("requirements.txt")],
    author_email='support@cadic.jp',
    description='GNU Tools for python',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
    ]
)
```

## How to use
To clone and run this application, you'll need [Git](https://git-scm.com) and [ https://docs.docker.com/docker-for-mac/install/]( https://docs.docker.com/docker-for-mac/install/) and Python installed on your computer. 
From your command line:

Install the package:
```bash
# Clone this repository and install the code
git clone https://github.com/JeanMaximilienCadic/nmesh

# Go into the repository
cd nmesh
```

## Makefile
Exhaustive list of make commands:
```
install_wheels
sandbox_cpu
sandbox_gpu
build_sandbox
push_environment
push_container_sandbox
push_container_vanilla
pull_container_vanilla
pull_container_sandbox
build_vanilla
clean
build_wheels
auto_branch 
```
## Docker
(\* recommended)

To build and run the docker image
```
make build
make docker_run_sandbox_cpu
```

## PythonEnv
(\* not recommended)
```
make install_wheels
```


## Ressources
* Vanilla:  https://en.wikipedia.org/wiki/Vanilla_software
* Sandbox: https://en.wikipedia.org/wiki/Sandbox_(software_development)
* All you need is docker: https://www.theregister.com/2014/05/23/google_containerization_two_billion/
* Dev in containers : https://code.visualstudio.com/docs/remote/containers
* Delta lake partitions: https://k21academy.com/microsoft-azure/data-engineer/delta-lake/




