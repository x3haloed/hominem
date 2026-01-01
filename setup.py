from __future__ import annotations

from setuptools import find_packages, setup


setup(
    name="hominem",
    version="0.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
    extras_require={
        "training_factory": [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "pydantic>=2.0.0",
            "mlx",
            "mlx-vlm @ git+https://github.com/x3haloed/mlx-vlm.git@weighted-loss",
        ],
    },
)
