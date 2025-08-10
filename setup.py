from setuptools import find_packages, setup

setup(
    name="instinct_onboard",
    version="0.1.0",
    packages=find_packages(exclude=["scripts", "tests"]),
    install_requires=[
        "numpy",
        "numpy-quaternion",
        "pyyaml",
        # 'rclpy',
        "transformations",
        "onnxruntime",
        "empy==3.3.2",  # codespell:ignore
    ],
    zip_safe=True,
)
