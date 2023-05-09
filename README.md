# Causal Inference and Discovery in Python
Causal Inference and Discovery in Python (2023) by Aleksander Molak (Packt Publishing)

## Environment installation
To install the basic environment run:
`conda env create -f causal_book_py39_cuda117.yml`

To install the environment for notebook `Chapter_11.2.ipynb` run:
`conda create -f causal-pymc.yml`

## Selecting the kernel
After a successful installation of the environment, open your notebook and select the kernel `causal_book_py39_cuda117`

For notebook `Chapter_11.2.ipynb` change kernel to `causal-pymc`

## Using `graphviz` and GPU

**Note**: Depending on your system settings, you might need to install `graphviz` manually in order to recreate the graph plots in the code. 
Check https://pypi.org/project/graphviz/ for specific instructions 
specific to your operating system.

**Note 2**: To use GPU you'll need to install CUDA 11.7 drivers.
This can be done here: https://developer.nvidia.com/cuda-11-7-0-download-archive
