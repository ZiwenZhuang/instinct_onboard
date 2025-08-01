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
    ],
    zip_safe=True,
)
