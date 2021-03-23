from setuptools import setup

setup(
    name='nmesh',
    version="0.0.1",
    long_description="",
    packages=[
        "nmesh"
    ],
    include_package_data=True,
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

