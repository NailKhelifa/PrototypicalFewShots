from setuptools import setup, find_packages

setup(
    name="CPD_L0_Penalty",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["pandas", "torch", "numpy",
                      "matplotlib", "scikit-learn", "h5py",
                      "tqdm"],
)