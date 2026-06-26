from setuptools import find_packages, setup

setup(
    name="harl",
    version="1.0.0",
    author="Satellite-MPE MAE contributors",
    description="Satellite-MPE MAE benchmark and on-policy MARL baselines",
    url="https://github.com/Hidenshadow/HARL_MPE_Satellite",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.8",
    install_requires=[
        "gymnasium",
        "numpy",
        "pygame",
        "supersuit",
        "torch>=1.9.0",
        "pyyaml>=5.3.1",
        "tensorboard>=2.2.1",
        "tensorboardX",
        "setproctitle",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
