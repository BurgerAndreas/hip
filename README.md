# HIP: Hessian Interatomic Potentials Without Derivatives


<p align="center">
<a href="https://arxiv.org/abs/2509.21624"><img src="https://img.shields.io/badge/arXiv-b31b1b?style=for-the-badge&logo=arxiv" alt="arXiv"/></a>
<!-- <a href="https://github.com/plainerman/variational-doob"><img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python"/></a> -->
<a href="https://colab.research.google.com/drive/1H_e9eABIutVTT6Q6LfV6ku0Mqp_Uzm8M?usp=sharing"><img src="https://img.shields.io/badge/Colab-e37e3d.svg?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Jupyter"/></a>
<a href="https://github.com/BurgerAndreas/hip"><img src="https://img.shields.io/badge/library-PyTorch-5f0964?style=for-the-badge" alt="PyTorch"/></a>
</p>

A machine learning interatomic potential to directly predict the Hessian.
Trained on the [HORM Hessian dataset](https://github.com/deepprinciple/HORM), which consists of off-equilibrium geometries of small, neutral organic molecules, contained H, C, N, O, based on the T1x and RGD1 datasets, at the $\omega$B97X/6-31G(d) level of theory.

Compared to autograd Hessians:
- 10-70x faster for a single molecule of 5-30 atoms
- 70x faster for a typical T1x batch in batched prediction
- 3x memory reduction
- Better accuracy (Hessian, Hessian eigenvalues and eigenvectors)
- Better downstream accuracy (relaxation, transition state search, frequency analysis)

![Speed and memory comparison](static/combined_speed_memory_batchsize.png)

## Installation

### Setting up the environment
Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

```bash
git clone git@github.com:BurgerAndreas/hip.git
cd hip

# Create virtual environment and install base dependencies
uv venv .venv --python 3.11
source .venv/bin/activate
uv sync

# Install PyTorch with CUDA support
uv pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126

# Install PyTorch Geometric packages with CUDA support
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
uv pip install torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
uv pip install torch-geometric

# Install the package in development mode
uv pip install -e .
```

## Use our model

Download the checkpoint from HuggingFace
```bash
wget https://huggingface.co/andreasburger/heigen/resolve/main/ckpt/hip_v2.ckpt -O ckpt/hip_v2.ckpt
```

```bash
uv run example.py
```

## Setting up the HORM dataset for training
Kaggle automatically downloads to the `~/.cache` folder. 
I highly recommend to set up a symbolic link to a local folder to avoid running out of space on your home directory:
```bash
PROJECT = <folder where you want to store the dataset>
mkdir -p ${PROJECT}/.cache
ln -s ${PROJECT}/.cache ${HOME}/.cache
```

Get the HORM dataset: 
```bash
uv run scripts/download_horm_data_kaggle.py
```

## Citation

If you found this code useful, please consider citing:
```bibtex
@misc{burger2025hiphessian,
      title={Shoot from the HIP: Hessian Interatomic Potentials without derivatives}, 
      author={Andreas Burger and Luca Thiede and Nikolaj Rønne and Varinia Bernales and Nandita Vijaykumar and Tejs Vegge and Arghya Bhowmik and Alan Aspuru-Guzik},
      year={2025},
      eprint={2509.21624},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.21624}, 
}
```

The training code and the dataset are based on the HORM [paper](https://arxiv.org/abs/2505.12447), [dataset](https://www.kaggle.com/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/data), and [code](https://github.com/deepprinciple/HORM)
We thank the authors of from DeepPrinciple for making their code and data openly available. 
