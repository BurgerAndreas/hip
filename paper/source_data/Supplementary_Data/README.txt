Supplementary Data for HIP: Hessian Interatomic Potentials without derivatives

These files are Supplementary Data for SI figures and tables, not Source Data.
Source Data for main-text Figures 2-5 is in Figure_2.zip through Figure_5.zip.
Column names follow the same style as those Source Data files.

Supplementary_Figure_batching.csv
  Replicates behind the batch-size timing figure. method is HIP or
  EquiformerV2 AD (direct force). time_milliseconds is wall time for the
  batch; time_per_sample_milliseconds is wall time divided by batch_size
  (the plotted quantity). n=10 replicates per method and batch size.
  AD stops at batch size 7.

Supplementary_Figure_loss_relaxation.csv
  Per-geometry RFO runs for HIP Hessians trained with MSE, MAE or MAE+Sub.
  Columns match Figure 4a (molecule_index, optimization_method, converged,
  optimization_step_count, wall_time_seconds). n=80 geometries.

Supplementary_Figure_glycine_energy_surface.csv
  DFT energies on the 579-point glycine surface. Coordinate names match
  Figure 2a. Full geometries, forces and Hessians are on Hugging Face
  (orca_wb97x_d3_631gd_glycine_pt_dft_relaxed_579) and Zenodo
  10.5281/zenodo.22003643.

Supplementary_Table_loss_accuracy.csv
  Validation Hessian metrics for the three loss functions, n=1000
  HORM-T1x geometries. Cosine similarities are unitless; MAE columns
  include units.

Supplementary_Table_loss_zpe.csv
  Per-geometry zero-point energies (reactant and product) for n=47 pairs.
  The SI table reports reactant ZPE MAE only. Values in parentheses are the
  sample standard deviation of the signed error (model minus DFT), not of
  the absolute error. Product ZPEs are included here so the table can be
  recomputed under other aggregations.

Supplementary_Table_loss_delta_zpe.csv
  Per-reaction Delta ZPE versus DFT for the same 47 pairs. The SI table
  reports MAE of |Delta ZPE_model - Delta ZPE_DFT| and the sample standard
  deviation of the signed Delta ZPE error.

Supplementary_Table_loss_tssearch.csv
  Per-reaction ReactBench outcomes for the three losses, n=960.
  Column names follow Figure 4b (gsm_converged, rfo_converged).

Supplementary_Table_loss_freqanalysis_per_geometry.csv
  Per-geometry negative-mode counts for the frequency-analysis table,
  n=1000 HORM-T1x validation geometries per loss. Column names follow
  Figure 4d. A first-order transition state has exactly one negative
  eigenvalue. Accuracy All in the SI table is
  exact_negative_mode_count_match, not three_class_match.

Supplementary_Table_loss_freqanalysis.csv
  Confusion counts and rates for identifying first-order transition
  states (positive class: negative-mode count = 1). Rates are unitless
  fractions; the SI table reports them as percents. The last two columns
  match transition_state_identification_accuracy_fraction and
  exact_negative_mode_accuracy_fraction in Supplementary_Table_loss_accuracy.csv.
