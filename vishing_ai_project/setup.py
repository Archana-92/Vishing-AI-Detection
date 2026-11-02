from setuptools import setup, find_packages

setup(
    name="vishing_ai",
    version="1.0.0",
    description="Multimodal voice phishing (vishing) detection via a Convolutional "
                 "Transformer Fusion Network (CTFN) with attention-based explainability.",
    packages=find_packages(include=["vishing_ai", "vishing_ai.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
        "librosa>=0.10",
        "soundfile>=0.12",
        "openai-whisper>=20231117",
        "numpy>=1.24",
        "scipy>=1.11",
    ],
    extras_require={
        "dev": ["pytest>=7.4"],
    },
    entry_points={
        "console_scripts": [
            "vishing-train=vishing_ai.train:main",
            "vishing-infer=vishing_ai.infer:main",
            "vishing-generate-dataset=vishing_ai.data.generate_dataset:main",
            "vishing-build-cache=vishing_ai.scripts.build_feature_cache:main",
        ],
    },
)
