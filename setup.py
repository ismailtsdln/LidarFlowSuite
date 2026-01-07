from setuptools import setup, find_packages

setup(
    name="lidarflowsuite",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "open3d",
        "fastapi",
        "uvicorn",
        "typer",
        "pyyaml",
        "scipy",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "lidarflowsuite=lidarflowsuite.cli.main:app",
        ],
    },
)
