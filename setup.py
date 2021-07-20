from setuptools import setup
from nmesh import __version__
setup(
    name='nmesh',
    version=__version__,
    long_description="",
    packages=[
        "nmesh",
        "nmesh.core",
        "nmesh.cp"
    ],
    url='https://github.com/JeanMaximilienCadic/nmesh',
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

