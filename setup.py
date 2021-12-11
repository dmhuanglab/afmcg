import setuptools

setuptools.setup(
    name="afmcg",
    version="0.0.1",
    license="CC-BY 4.0",
    author="Huong TL Nguyen",
    author_email="huong.nguyen@adelaide.edu.au",
    description="Anisotropic coarse-graining of atomistic dynamical simulations using force-matching",
    url="https://github.com/dmhuanglab/afmcg",
    download_url = "https://github.com/dmhuanglab/afmcg/archive/refs/tags/v0.0.1.tar.gz",
    keywords = ['coarse-grained','parametrization','force-matching'],
    packages=setuptools.find_packages(),
    install_requires=(
        'matplotlib',
        'numpy',
    ),
    classifiers=[
        "License :: CC-BY 4.0",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
)
