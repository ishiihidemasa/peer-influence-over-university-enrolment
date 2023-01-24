# updated on: 2023-01-13
FROM python:3.10.8-bullseye

# Suppress creation of __pycache__
# value is arbitrary
ENV PYTHONDONTWRITEBYTECODE 1

# for Python Interactive window in VS Code
RUN python -m pip install jupyter==1.0.0

# update setuptools and pip
RUN python -m pip install setuptools==65.7.0 \
    && python -m pip install pip==22.3.1

# formatter
RUN python -m pip install black==22.12.0

# libraries
RUN python -m pip install numpy==1.23.5
RUN python -m pip install matplotlib==3.6.3
RUN python -m pip install scipy==1.10.0
RUN python -m pip install networkx==3.0
RUN python -m pip install pandas==1.5.2
RUN python -m pip install numba==0.56.4

