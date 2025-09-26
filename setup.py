"""Setup script for Agent Forge."""

from setuptools import setup, find_packages
import os

# Read README for long description
README_PATH = os.path.join(os.path.dirname(__file__), 'README.md')
if os.path.exists(README_PATH):
    with open(README_PATH, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "Agent Forge - Multi-agent system for intelligent task execution"

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    requirements = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove version constraints for base requirements
                    if '>=' in line:
                        package = line.split('>=')[0]
                    elif '==' in line:
                        package = line.split('==')[0]
                    elif '<' in line:
                        package = line.split('<')[0]
                    else:
                        package = line

                    # Handle extras (e.g., package[extra])
                    if '[' in package and ']' in package:
                        requirements.append(line)  # Keep full line for extras
                    else:
                        requirements.append(package)
    return requirements

# Core requirements (essential for basic functionality)
install_requires = [
    'pydantic>=2.0.0',
    'pydantic-settings>=2.0.0',
    'fastapi>=0.104.0',
    'uvicorn[standard]>=0.24.0',
    'aiohttp>=3.9.0',
    'numpy>=1.24.0',
    'python-jose[cryptography]>=3.3.0',
    'passlib[bcrypt]>=1.7.4',
    'python-multipart>=0.0.6',
    'click>=8.1.0',
    'python-dotenv>=1.0.0',
    'tqdm>=4.66.0'
]

# Optional extras
extras_require = {
    'ai': [
        'torch>=2.0.0',
        'scikit-learn>=1.3.0',
        'sentence-transformers>=2.2.0',
    ],
    'cache': [
        'aioredis>=2.0.0',
    ],
    'data': [
        'networkx>=3.0',
        'pandas>=2.0.0',
    ],
    'database': [
        'sqlalchemy>=2.0.0',
        'alembic>=1.12.0',
    ],
    'dev': [
        'pytest>=7.4.0',
        'pytest-asyncio>=0.21.0',
        'pytest-cov>=4.1.0',
        'black>=23.0.0',
        'isort>=5.12.0',
        'flake8>=6.0.0',
        'mypy>=1.5.0',
    ],
    'viz': [
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
    ],
    'docs': [
        'mkdocs>=1.5.0',
        'mkdocs-material>=9.4.0',
    ],
    'jupyter': [
        'jupyter>=1.0.0',
        'ipykernel>=6.25.0',
    ]
}

# All optional dependencies
extras_require['all'] = [
    dep for deps in extras_require.values()
    for dep in deps
]

setup(
    name="agent-forge",
    version="2.0.0",
    author="Agent Forge Team",
    author_email="team@agentforge.dev",
    description="Multi-agent system for intelligent task execution and coordination",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DNYoussef/agent-forge",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'agent-forge=agent_forge.cli:main',
        ],
    },
    include_package_data=True,
    package_data={
        'agent_forge': [
            'templates/*.json',
            'config/*.yaml',
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/DNYoussef/agent-forge/issues",
        "Source": "https://github.com/DNYoussef/agent-forge",
        "Documentation": "https://github.com/DNYoussef/agent-forge/wiki",
    },
    keywords=[
        "ai", "artificial-intelligence", "multi-agent", "agents",
        "coordination", "distributed-systems", "machine-learning",
        "automation", "intelligent-systems", "swarm-intelligence"
    ],
    zip_safe=False,
)