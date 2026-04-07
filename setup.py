from setuptools import find_packages, setup

setup(
    name="agent_enhanced_graphrag",
    version="0.1.0",
    description="Agent-Enhanced GraphRAG for multi-hop QA",
    packages=find_packages(where="."),
    include_package_data=True,
    install_requires=[
        line.strip()
        for line in open("requirements.txt", encoding="utf-8")
        if line.strip() and not line.startswith("#")
    ],
)
