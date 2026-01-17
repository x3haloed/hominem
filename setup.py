from __future__ import annotations

from setuptools import find_packages, setup


setup(
    name="hominem",
    version="0.0.0",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    include_package_data=True,
    install_requires=[
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "agent": [
            "qwen-agent>=0.0.31",
            "openai>=1.0.0",
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
        ],
        "tools": [
            "tabstack>=2.0.0",
        ],
        "infer": [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
        ],
        "open_webui": [
            "open-webui==0.7.2",
        ],
        "training_factory": [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "pydantic>=2.0.0",
            "mlx",
            "mlx-vlm @ git+https://github.com/x3haloed/mlx-vlm.git@weighted-loss",
            "torchvision"
        ],
    },
)
