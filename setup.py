from setuptools import setup

from nmesh import __version__

setup(
    name='nmesh',
    version=__version__,
    packages=[
        "nmesh",
        "nmesh.core",
        "nmesh.pc",
        "nmesh.tests"
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
    author_email='git@cadic.jp',
    description='GNU Tools for python',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
    ]
)
