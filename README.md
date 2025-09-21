# Molecular Hessians Without Derivatives

A machine learning force field (interactomic potential) to directly predict the Hessian.
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

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install --upgrade pip

uv pip install torch==2.7.0  --index-url https://download.pytorch.org/whl/cu126
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
uv pip install torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
uv pip install torch-geometric

uv pip install git+https://github.com/KellerJordan/Muon
uv pip install -r requirements.txt

uv pip install -e .
```


### Setting up the HORM dataset
Kaggle automatically downloads to the `~/.cache` folder. 
I highly recommend to set up a symbolic link to a local folder to avoid running out of space on your home directory:
```bash
PROJECT = <folder where you want to store the dataset>
mkdir -p ${PROJECT}/.cache
ln -s ${PROJECT}/.cache ${HOME}/.cache
```

Get the HORM dataset: # TODO: upload preprocessed data
```bash
python scripts/download_horm_data_kaggle.py
```

Preprocess the Hessian dataset (takes ~48 hours) 
```bash
python scripts/preprocess_hessian_dataset.py --dataset-file data/sample_100.lmdb

python scripts/preprocess_hessian_dataset.py --dataset-file ts1x-val.lmdb
python scripts/preprocess_hessian_dataset.py --dataset-file RGD1.lmdb
python scripts/preprocess_hessian_dataset.py --dataset-file ts1x_hess_train_big.lmdb
```

### Coming soon: RGD1 dataset

```bash
uv run scripts/process_rgd1_minimal_to_lmdb.py
```


## Use our model

Download the checkpoint from HuggingFace
```bash
wget https://huggingface.co/andreasburger/heigen/resolve/main/ckpt/hesspred_v1.ckpt -O ckpt/hesspred_v1.ckpt
```

See [example_inference.py](example_inference.py) for a full example how to use our model.

```python
import os
import torch
from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.equiformer_ase_calculator import EquiformerASECalculator # also try this
from hip.inference_utils import get_dataloader
from hip.frequency_analysis import analyze_frequencies_torch


device = "cuda" if torch.cuda.is_available() else "cpu"

# you might need to change this
project_root = os.path.dirname(os.path.dirname(__file__))
checkpoint_path = os.path.join(project_root, "ckpt/hesspred_v1.ckpt")
calculator = EquiformerTorchCalculator(
    checkpoint_path=checkpoint_path,
    hessian_method="predict",
)

# Example 1: load a dataset file and predict the first batch
dataset_path = os.path.join(project_root, "data/sample_100.lmdb")
dataloader = get_dataloader(
    dataset_path, calculator.potential, batch_size=1, shuffle=False
)
batch = next(iter(dataloader))
results = calculator.predict(batch)
print("\nExample 1:")
print(f"  Energy: {results['energy'].shape}")
print(f"  Forces: {results['forces'].shape}")
print(f"  Hessian: {results['hessian'].shape}")

print("\nGAD:")
gad = calculator.get_gad(batch)
print(f"  GAD: {gad['gad'].shape}")

# Example 2: create a random data object with random positions and predict
n_atoms = 10
elements = torch.tensor([1, 6, 7, 8])  # H, C, N, O
pos = torch.randn(n_atoms, 3)  # (N, 3)
atomic_nums = elements[torch.randint(0, 4, (n_atoms,))]  # (N,)
results = calculator.predict(coords=pos, atomic_nums=atomic_nums)
print("\nExample 2:")
print(f"  Energy: {results['energy'].shape}")
print(f"  Forces: {results['forces'].shape}")
print(f"  Hessian: {results['hessian'].shape}")

print("\nFrequency analysis:")
hessian = results["hessian"]
frequency_analysis = analyze_frequencies_torch(hessian, pos, atomic_nums)
print(f"eigvals: {frequency_analysis['eigvals'].shape}")
print(f"eigvecs: {frequency_analysis['eigvecs'].shape}")
print(f"neg_num: {frequency_analysis['neg_num']}")
print(f"natoms: {frequency_analysis['natoms']}")
```


## Reproduce results from our paper

Training run we used: 
```bash
uv run scripts/train.py trgt=hessian experiment=hesspred_alldata preset=luca8w10only training.bz=128 model.num_layers_hessian=3
```

Evaluation: 

DFT Hessians for reactant geometries in T1x validation set, which we use to evaluate geometry optimization (relaxations)
```bash
uv run scripts/compute_dft_hessian_t1x.py --noiserms 0.05
```

## Evaluations

### Evluation setup
To run the transition state workflow and the relaxation evaluations you need the sister repository as well:
```bash
cd ..
git clone git@github.com:BurgerAndreas/ReactBench.git
cd ReactBench/dependencies 
git clone git@github.com:BurgerAndreas/pysisyphus.git 
git clone git@github.com:BurgerAndreas/pyGSM.git 
cd ..

uv pip install -e . # install ReactBench

# install leftnet env
uv pip install -e ReactBench/MLIP/leftnet/

# Mace requires e3nn<5.*, but pytorch 2.7.0 only supports e3nn>=5.0.0
# install mace env
# uv pip install -e ReactBench/MLIP/mace

# Get the recomputed Transition1x subset for validation, 960 datapoints
mkdir -p data 
tar -xzf ts1x.tar.gz -C data
find data/ts1x -type f | wc -l # 960

cd ../hip
uv pip install -r requirements.txt
```

For the relaxations:
```bash
uv pip install gpu4pyscf-cuda12x cutensor-cu12

wget https://huggingface.co/andreasburger/heigen/resolve/main/data/t1x_val_reactant_hessian_100_noiserms0.03.h5 -O data/t1x_val_reactant_hessian_100_noiserms0.03.h5

wget https://huggingface.co/andreasburger/heigen/resolve/main/data/t1x_val_reactant_hessian_100_noiserms0.05.h5 -O data/t1x_val_reactant_hessian_100_noiserms0.05.h5
```

Get the baseline model checkpoints:
- `ckpt/eqv2.ckpt`: HORM EquiformerV2 finetuned on the HORM Hessian dataset. Can be used to get the Hessian via autograd. Used as starting point for training our HessianLearning model as well as baseline for evaluation.

```bash
# Download HORM EquiformerV2 with Energy-Force-Hessian Training
mkdir -p ckpt
wget https://huggingface.co/yhong55/HORM/resolve/main/eqv2.ckpt -O ckpt/eqv2.ckpt
# Other models from the HORM paper
wget https://huggingface.co/yhong55/HORM/resolve/main/left-df.ckpt -O ckpt/left-df.ckpt
wget https://huggingface.co/yhong55/HORM/resolve/main/left.ckpt -O ckpt/left.ckpt
wget https://huggingface.co/yhong55/HORM/resolve/main/alpha.ckpt -O ckpt/alpha.ckpt
```

### Run evaluations

```bash
export REACTBENCHDIR=/ssd/Code/ReactBench
export hipDIR=/ssd/Code/hip
export HPCKPT="${hipDIR}/ckpt/hesspred_v1.ckpt"

cd $hipDIR
source .venv/bin/activate

# Table 1: MAE, cosine similarity, ...
# other HORM autograd models
uv run scripts/eval_horm.py --max_samples=1000 --ckpt_path=ckpt/alpha.ckpt 
uv run scripts/eval_horm.py --max_samples=1000 --ckpt_path=ckpt/left.ckpt  
uv run scripts/eval_horm.py --max_samples=1000 --ckpt_path=ckpt/left-df.ckpt 
# autograd EquiformerV2
uv run scripts/eval_horm.py --max_samples=1000 --ckpt_path=ckpt/eqv2.ckpt 
# Learned EquiformerV2
uv run scripts/eval_horm.py --max_samples=1000 --ckpt_path=$HPCKPT --hessian_method=predict 

# Plot results in ../hip/results/eqv2_ts1x-val_autograd_metrics.csv / wandb export
uv run scripts/plot_frequency_analysis.py

# Speed and memory comparison (plot included)
cd $hipDIR
uv run scripts/speed_comparison.py --dataset RGD1.lmdb --max_samples_per_n 100 --ckpt_path $hipDIR/ckpt/eqv2.ckpt

# Transition state workflow
cd $REACTBENCHDIR
uv run ReactBench/main.py config.yaml --calc=equiformer --hessian_method=autograd --redo_all=True --config_path=null
uv run ReactBench/main.py config.yaml --calc=equiformer --ckpt_path=$HPCKPT --hessian_method=predict --redo_all=True

cd $REACTBENCHDIR
# # /ssd/Code/ReactBench/runs/leftnet-d_hormleft-df_ts1x_autograd/ts_proposal_geoms
# uv run verify_ts_with_dft.py leftnet-d_hormleft-df --hessian_method autograd --max_samples 100
# # /ssd/Code/ReactBench/runs/leftnet_hormleft_ts1x_autograd/ts_proposal_geoms
# uv run verify_ts_with_dft.py leftnet_hormleft --hessian_method autograd --max_samples 100
uv run verify_ts_with_dft.py equiformer_hesspred_v1 --max_samples 100
uv run verify_ts_with_dft.py $REACTBENCHDIR/runs/equiformer_ts1x_autograd --hessian_method autograd --max_samples 100
# plot
uv run verify_ts_with_dft.py plot

# Lollipop plots for TS workflow
cd $hipDIR
uv run scripts/plot_reactbench.py

# Relaxations (2nd order geometry optimization)
cd $hipDIR
# uv run scripts/compute_dft_hessian_t1x.py --noiserms 0.03
# uv run scripts/compute_dft_hessian_t1x.py --noiserms 0.05
# --redo True --coord cart
# uv run scripts/second_order_relaxation_pysiyphus.py --max_samples 80 --thresh gau --max_cycles 150 --xyz $hipDIR/data/t1x_val_reactant_hessian_100_noiserms0.03.h5
uv run scripts/second_order_relaxation_pysiyphus.py --max_samples 80 --thresh gau --max_cycles 150 --xyz $hipDIR/data/t1x_val_reactant_hessian_100_noiserms0.05.h5

# Zero-point energy
uv run scripts/zero_point_energy_at_dft_reactant_product.py --thresh gau_tight --max_samples 80

# Optional: equivariance test
# uv run scripts/test_hessian_prediction.py
```

### Hyperparameter search

Create the sweep and note the SWEEP_ID:
```bash
source .venv/bin/activate
wandb sweep sweeps/hessian_uv.yaml
```

Start the background relaunch loop (on the login node or screen/tmux):
```bash
export SWEEP_ID=<YOUR_SWEEP_ID>
bash scripts/launch_sweep_loop.sh

# stop later
pkill -f scripts/launch_sweep_loop.sh
```

### Sella: work in progress
```bash
uv pip install -U "jax[cuda12]"==0.6.2
uv pip install -e sella
uv pip install git+https://github.com/virtualzx-nad/geodesic-interpolate.git
```


## Citation

If you found this code useful, please consider citing:
```bibtex
@inproceedings{
burger2025hessians,
title={Molecular Hessians Without Derivatives},
author={Andreas Burger and Luca Thiede and  Nikolaj Rønne and Nandita Vijaykumar and Tejs Vegge and Arghya Bhowmik and Alan Aspuru-Guzik},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=CNLC4ZkLmW}
}
```

The training code and the dataset are based on the HORM [paper](https://arxiv.org/abs/2505.12447), [dataset](https://www.kaggle.com/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/data), and [code](https://github.com/deepprinciple/HORM)
We thank the authors of from DeepPrinciple for making their code and data openly available. 
```bibtex
@misc{cui2025hormlargescalemolecular,
      title={HORM: A Large Scale Molecular Hessian Database for Optimizing Reactive Machine Learning Interatomic Potentials}, 
      author={Taoyong Cui and Yunhong Han and Haojun Jia and Chenru Duan and Qiyuan Zhao},
      year={2025},
      eprint={2505.12447},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2505.12447}, 
}
```