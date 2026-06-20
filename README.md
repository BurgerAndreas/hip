# HIP: Hessian Interatomic Potentials

Paper: [https://arxiv.org/abs/2509.21624](https://arxiv.org/abs/2509.21624)   

Official repo: [https://github.com/BurgerAndreas/hip](https://github.com/BurgerAndreas/hip)   

MACE implementation (work in progress): [https://github.com/BurgerAndreas/hip-mace](https://github.com/BurgerAndreas/hip-mace)   

HIPs are machine learning interatomic potentials (MLIPs) that directly predict the Hessian, in addition to the usual energy and forces.
This repo primarily trains HIP-EquiformerV2 on the [HORM Hessian dataset](https://github.com/deepprinciple/HORM), which consists of off-equilibrium geometries of small, neutral organic molecules, contained H, C, N, O, based on Transition1x, at the $\omega$B97X/6-31G(d) level of theory.

Compared to autograd Hessians, HIP is:

- 10-70x faster for a single molecule of 5-30 atoms
- 70x faster for a typical T1x batch in batched prediction
- 3x memory reduction
- Better accuracy (Hessian, Hessian eigenvalues and eigenvectors)
- Better downstream accuracy (relaxation, transition state search, frequency analysis)

Speed and memory comparison

## Installation

This should only take 5-10 minutes depending on your internet connection.

### Setting up the environment

First install the uv package manager (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

```bash
git clone git@github.com:BurgerAndreas/hip.git
cd hip

# --extra cuda126
uv sync --python 3.12 --extra cuda121
```

## Use our model

Download the latest checkpoints from HuggingFace:

```bash
mkdir -p ckpt
wget https://huggingface.co/andreasburger/hip/resolve/main/ckpt/hip_v3.ckpt -O ckpt/hip_v3.ckpt
wget https://huggingface.co/andreasburger/hip/resolve/main/ckpt/hip_v3.yaml -O ckpt/hip_v3.yaml
wget https://huggingface.co/andreasburger/hip/resolve/main/ckpt/hip_v3_cf.ckpt -O ckpt/hip_v3_cf.ckpt
wget https://huggingface.co/andreasburger/hip/resolve/main/ckpt/hip_v3_cf.yaml -O ckpt/hip_v3_cf.yaml
```

Available checkpoints:

- `hip_v3.ckpt`: latest HIP checkpoint with direct force prediction.
- `hip_v3_cf.ckpt`: latest HIP checkpoint trained with conservative forces (`model.direct_forces=False`).
- `hip_v3.yaml` and `hip_v3_cf.yaml`: saved model, optimizer, and training configs for the matching checkpoints.

Run a few forward passes (should take 30s)

```bash
uv run example.py
```

## Setting up the HORM dataset for training

Our models are trained on the Hessian dataset for Optimizing Reactive MLIP (HORM).

The HORM dataset is hosted on Kaggle.
Kaggle automatically downloads to the `~/.cache` folder. 
If you want to use another location for the files, I recommend to set up a symbolic link to a another folder:

```bash
PROJECT = <folder where you want to store the dataset>
mkdir -p ${PROJECT}/.cache
ln -s ${PROJECT}/.cache ${HOME}/.cache
```

Now download the HORM dataset (25GB): 

```bash
uv run scripts/download_horm_data_kaggle.py
```

Train HIP (around two to three days on a H100 GPU)

```bash
uv run scripts/train.py

# conservative forces
uv run scripts/train.py model.direct_forces=False

# reduce the batch size if you are running on a L40s or A100 with 40GB GPU RAM
# uv run scripts/train.py +extra=bz64
```

## Transition state search

For the transition state search we followed the HORM paper and used `ReactBench`

- [https://github.com/deepprinciple/ReactBench](https://github.com/deepprinciple/ReactBench)
- [https://github.com/deepprinciple/pysisyphus](https://github.com/deepprinciple/pysisyphus)
- [https://github.com/deepprinciple/pyGSM](https://github.com/deepprinciple/pyGSM)

Unfortunetly, the `ReactBench` code is a bit of a mess.

If I were to do this project again, I would use `geodesic interpolation + Sella TS search + Sella IRC` instead of `ReactBench + pysisyphus + pyGSM` as done in this paper:  
[https://www.nature.com/articles/s41467-024-52481-5](https://www.nature.com/articles/s41467-024-52481-5)  
For that you would need to install:  
[https://github.com/virtualzx-nad/geodesic-interpolate](https://github.com/virtualzx-nad/geodesic-interpolate)  
[https://github.com/zadorlab/sella](https://github.com/zadorlab/sella)  
and follow their workflow from here:  
[https://github.com/Quantum-Accelerators/quacc/blob/main/src/quacc/recipes/newtonnet/ts.py](https://github.com/Quantum-Accelerators/quacc/blob/main/src/quacc/recipes/newtonnet/ts.py)

## Plots

Plotting scripts read generated data from `runs/` and write rendered figures under `plots/` by default.

Regenerate the current plot set:

```bash
uv run python plotting/plot_eqv2_detach_no_detach_fd_convergence.py
uv run python plotting/plot_eqv2_roughness_vs_dft.py
uv run python plotting/plot_eqv2_force_smoothness.py
uv run python plotting/plot_t1x_val_force_spectra.py --scan-dir runs/t1x_val_force_spectra_100x2x51
uv run python plotting/plot_leftnet_joint_force_spectra.py
uv run python plotting/plot_eqv2_hip_joint_force_spectra.py
uv run python plotting/plot_glycine_pt_energy_surfaces.py --scan-dir runs/glycine_pt_scan --orca-dir orca_wb97x_631gd_glycine_pt_nh_oh_scan_80
uv run python plotting/plot_glycine_pt_hessian_scan.py --scan-dir runs/glycine_pt_scan --orca-dir orca_wb97x_631gd_glycine_pt_nh_oh_scan_80
uv run python plotting/plot_glycine_pt_dft_cv_diagnostics.py --scan-dir runs/glycine_pt_scan_n36
uv run python plotting/plot_glycine_pt_path_forces.py --path-dir runs/glycine_pt_path
uv run python plotting/plot_glycine_pt_path_mechanism.py --path-dir runs/glycine_pt_path
uv run python plotting/plot_glycine_pt_path_ad_failure.py --path-dir runs/glycine_pt_path_dft
uv run python plotting/plot_glycine_pt_path_hessian_diag.py --path-dir runs/glycine_pt_path_dft
uv run python plotting/plot_glycine_ad_hessian_failure.py
uv run python plotting/plot_glycine_pt_mep_73_diagnostics.py --mep-dir runs/glycine_pt_mep_73
uv run python plotting/plot_eval_horm_error_distributions.py
uv run python plotting/visualize_glycine_pt_xyzrender.py
```

Regenerate the 150-point glycine proton-slide DFT/AD/HIP mechanism figures. The ORCA cache is reused from `runs/glycine_pt_path_n150/orca_vib_cache.npz`; rerun the MLIP path only when `path_arrays.npz` needs the full Cartesian AD/HIP forces and Hessians:

```bash
sbatch -p polar --export=ALL,OUTPUT_DIR=runs/glycine_pt_path_n150,N_DENSE=150,N_DFT=150,WRITE_DFT=0,OVERWRITE_MLIP=1 scripts/run_glycine_pt_path.sbatch
```

```bash
uv run python scripts/adapt_glycine_pt_path_arrays.py --path-dir runs/glycine_pt_path_n150
```

```bash
uv run python plotting/plot_glycine_pt_mep_mechanism.py && uv run python plotting/plot_glycine_pt_mep_73_diagnostics.py
```

### Glycine MEP Hessian eigenvalues (73- vs 150-point)

The `mep_lowest_hessian_eigenvalues.png` figure (six lowest mass-weighted Hessian eigenvalues along the proton-transfer path, DFT vs HIP vs AD) used in the paper is produced by `plot_glycine_pt_mep_73_diagnostics.py`. We keep two versions so they can be compared: the 73-point geodesic-interpolated NEB path and the 150-point dense scan. Each run reads `runs/<mep-dir>/{orca_vib_cache.npz,hip_v2_arrays.npz,eqv2_autograd_arrays.npz}` and writes to `plots/<mep-dir>/mep_diagnostics/`.

73-point path (geodesic-interpolated NEB):

```bash
uv run python plotting/plot_glycine_pt_mep_73_diagnostics.py --mep-dir runs/glycine_pt_mep_73 --output-dir plots/glycine_pt_mep_73/mep_diagnostics
cp plots/glycine_pt_mep_73/mep_diagnostics/mep_lowest_hessian_eigenvalues.png plots/mep_lowest_hessian_eigenvalues_v2.png
```

150-point path (dense proton slide):

```bash
uv run python plotting/plot_glycine_pt_mep_73_diagnostics.py --mep-dir runs/glycine_pt_path_n150 --output-dir plots/glycine_pt_path_n150/mep_diagnostics
cp plots/glycine_pt_path_n150/mep_diagnostics/mep_lowest_hessian_eigenvalues.png plots/mep_lowest_hessian_eigenvalues_n150.png
```

The paper figure (`hip.tex`) currently references `plots/mep_lowest_hessian_eigenvalues_v2.png` (the 73-point version); swap in `plots/mep_lowest_hessian_eigenvalues_n150.png` to use the 150-point version.

Regenerate the individual force median spectra summaries used by the joint spectra plots:

```bash
uv run python plotting/plot_t1x_val_force_spectra.py --scan-dir runs/t1x_val_force_spectra_100x2x51
uv run python plotting/plot_t1x_val_force_spectra.py --scan-dir runs/t1x_val_force_spectra_100x2x51 --model-force-dir runs/t1x_val_force_spectra_100x2x51/eqv2_orig_force_outputs --model-label 'EqV2 orig' --out-dir plots/t1x_val_force_spectra_100x2x51/force_spectra_analysis_eqv2_orig --no-hessian-metrics
uv run python plotting/plot_t1x_val_force_spectra.py --scan-dir runs/t1x_val_force_spectra_100x2x51 --model-force-dir runs/t1x_val_force_spectra_100x2x51/hip_v2_force_outputs --model-label 'HIP v2' --out-dir plots/t1x_val_force_spectra_100x2x51/force_spectra_analysis_hip_v2 --no-hessian-metrics
uv run python plotting/plot_t1x_val_force_spectra.py --scan-dir runs/t1x_val_force_spectra_100x2x51 --model-force-dir runs/t1x_val_force_spectra_100x2x51/t1x_val_force_spectra_leftnet/leftnet-cf --model-label 'LeftNet CF' --out-dir plots/t1x_val_force_spectra_100x2x51/t1x_val_force_spectra_leftnet/force_spectra_analysis/leftnet-cf --no-hessian-metrics
uv run python plotting/plot_t1x_val_force_spectra.py --scan-dir runs/t1x_val_force_spectra_100x2x51 --model-force-dir runs/t1x_val_force_spectra_100x2x51/t1x_val_force_spectra_leftnet/leftnet-df --model-label 'LeftNet DF' --out-dir plots/t1x_val_force_spectra_100x2x51/t1x_val_force_spectra_leftnet/force_spectra_analysis/leftnet-df --no-hessian-metrics
uv run python plotting/plot_t1x_val_force_spectra.py --scan-dir runs/t1x_val_force_spectra_100x2x51 --model-force-dir runs/t1x_val_force_spectra_100x2x51/t1x_val_force_spectra_leftnet/leftnet-cf-orig --model-label 'LeftNet CF orig' --out-dir plots/t1x_val_force_spectra_100x2x51/t1x_val_force_spectra_leftnet/force_spectra_analysis/leftnet-cf-orig --no-hessian-metrics
uv run python plotting/plot_t1x_val_force_spectra.py --scan-dir runs/t1x_val_force_spectra_100x2x51 --model-force-dir runs/t1x_val_force_spectra_100x2x51/t1x_val_force_spectra_leftnet/leftnet-df-orig --model-label 'LeftNet DF orig' --out-dir plots/t1x_val_force_spectra_100x2x51/t1x_val_force_spectra_leftnet/force_spectra_analysis/leftnet-df-orig --no-hessian-metrics
```

## Citation

If I can help you run the code or setup your own project, please email me at: `<firstname>.<lastname>(at)mail.utoronto.ca`

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

The dataset and parts of the training code are based on the HORM [paper](https://arxiv.org/abs/2505.12447), [dataset](https://www.kaggle.com/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/data), and [code](https://github.com/deepprinciple/HORM)
We thank the authors of from DeepPrinciple for making their code and data openly available. 
