import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import re
import wandb
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")


def flatten_dict(d, parent_key="", sep="."):
    """Recursively flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def exponential_func(x, a, b, c):
    """Exponential function: y = a * x^b + c"""
    return a * np.power(x, b) + c


def power_law_func(x, a, b):
    """Power law function: y = a * x^b"""
    return a * np.power(x, b)


def compare_models(x_data, y_data, max_poly_degree=4):
    """
    Compare different polynomial and exponential models using statistical tests.

    Parameters:
    -----------
    x_data : array-like
        Input data (dataset sizes)
    y_data : array-like
        Target data (loss values)
    max_poly_degree : int
        Maximum polynomial degree to test

    Returns:
    --------
    dict : Results with best model and comparison statistics
    """
    # Convert to log space for better fitting
    x_log = np.log10(x_data)
    y_log = np.log10(y_data)

    results = {}

    # Test polynomial models (degrees 1 to max_poly_degree)
    for degree in range(1, max_poly_degree + 1):
        try:
            # Fit polynomial
            poly_coeffs = np.polyfit(x_log, y_log, degree)
            poly_func = np.poly1d(poly_coeffs)
            y_pred = poly_func(x_log)

            # Calculate metrics
            mse = mean_squared_error(y_log, y_pred)
            r2 = r2_score(y_log, y_pred)

            # AIC calculation (Akaike Information Criterion)
            n = len(x_log)
            k = degree + 1  # number of parameters
            aic = n * np.log(mse) + 2 * k

            # BIC calculation (Bayesian Information Criterion)
            bic = n * np.log(mse) + k * np.log(n)

            # Cross-validation
            kf = KFold(n_splits=min(5, len(x_log)), shuffle=True, random_state=42)
            cv_scores = []
            for train_idx, val_idx in kf.split(x_log):
                x_train, x_val = x_log[train_idx], x_log[val_idx]
                y_train, y_val = y_log[train_idx], y_log[val_idx]

                poly_coeffs_cv = np.polyfit(x_train, y_train, degree)
                poly_func_cv = np.poly1d(poly_coeffs_cv)
                y_pred_cv = poly_func_cv(x_val)
                cv_scores.append(mean_squared_error(y_val, y_pred_cv))

            cv_mse = np.mean(cv_scores)
            cv_std = np.std(cv_scores)

            results[f"poly_degree_{degree}"] = {
                "type": "polynomial",
                "degree": degree,
                "mse": mse,
                "r2": r2,
                "aic": aic,
                "bic": bic,
                "cv_mse": cv_mse,
                "cv_std": cv_std,
                "coeffs": poly_coeffs,
                "func": poly_func,
            }
        except Exception as e:
            print(f"Error fitting polynomial degree {degree}: {e}")
            continue

    # Test exponential models
    try:
        # Exponential: y = a * x^b + c
        popt_exp, pcov_exp = curve_fit(
            exponential_func, x_data, y_data, p0=[1, -1, 0], maxfev=10000
        )
        y_pred_exp = exponential_func(x_data, *popt_exp)
        y_pred_exp_log = np.log10(y_pred_exp)

        mse_exp = mean_squared_error(y_log, y_pred_exp_log)
        r2_exp = r2_score(y_log, y_pred_exp_log)

        # AIC/BIC for exponential
        n = len(x_data)
        k = 3  # 3 parameters: a, b, c
        aic_exp = n * np.log(mse_exp) + 2 * k
        bic_exp = n * np.log(mse_exp) + k * np.log(n)

        # Cross-validation for exponential
        kf = KFold(n_splits=min(5, len(x_data)), shuffle=True, random_state=42)
        cv_scores_exp = []
        for train_idx, val_idx in kf.split(x_data):
            x_train, x_val = x_data[train_idx], x_data[val_idx]
            y_train, y_val = y_data[train_idx], y_data[val_idx]

            try:
                popt_cv, _ = curve_fit(
                    exponential_func, x_train, y_train, p0=[1, -1, 0], maxfev=10000
                )
                y_pred_cv = exponential_func(x_val, *popt_cv)
                y_pred_cv_log = np.log10(y_pred_cv)
                y_val_log = np.log10(y_val)
                cv_scores_exp.append(mean_squared_error(y_val_log, y_pred_cv_log))
            except Exception:
                continue

        cv_mse_exp = np.mean(cv_scores_exp) if cv_scores_exp else np.inf
        cv_std_exp = np.std(cv_scores_exp) if cv_scores_exp else np.inf

        results["exponential"] = {
            "type": "exponential",
            "mse": mse_exp,
            "r2": r2_exp,
            "aic": aic_exp,
            "bic": bic_exp,
            "cv_mse": cv_mse_exp,
            "cv_std": cv_std_exp,
            "params": popt_exp,
            "func": lambda x: exponential_func(x, *popt_exp),
        }
    except Exception as e:
        print(f"Error fitting exponential model: {e}")

    try:
        # Power law: y = a * x^b
        popt_power, pcov_power = curve_fit(
            power_law_func, x_data, y_data, p0=[1, -1], maxfev=10000
        )
        y_pred_power = power_law_func(x_data, *popt_power)
        y_pred_power_log = np.log10(y_pred_power)

        mse_power = mean_squared_error(y_log, y_pred_power_log)
        r2_power = r2_score(y_log, y_pred_power_log)

        # AIC/BIC for power law
        n = len(x_data)
        k = 2  # 2 parameters: a, b
        aic_power = n * np.log(mse_power) + 2 * k
        bic_power = n * np.log(mse_power) + k * np.log(n)

        # Cross-validation for power law
        kf = KFold(n_splits=min(5, len(x_data)), shuffle=True, random_state=42)
        cv_scores_power = []
        for train_idx, val_idx in kf.split(x_data):
            x_train, x_val = x_data[train_idx], x_data[val_idx]
            y_train, y_val = y_data[train_idx], y_data[val_idx]

            try:
                popt_cv, _ = curve_fit(
                    power_law_func, x_train, y_train, p0=[1, -1], maxfev=10000
                )
                y_pred_cv = power_law_func(x_val, *popt_cv)
                y_pred_cv_log = np.log10(y_pred_cv)
                y_val_log = np.log10(y_val)
                cv_scores_power.append(mean_squared_error(y_val_log, y_pred_cv_log))
            except Exception:
                continue

        cv_mse_power = np.mean(cv_scores_power) if cv_scores_power else np.inf
        cv_std_power = np.std(cv_scores_power) if cv_scores_power else np.inf

        results["power_law"] = {
            "type": "power_law",
            "mse": mse_power,
            "r2": r2_power,
            "aic": aic_power,
            "bic": bic_power,
            "cv_mse": cv_mse_power,
            "cv_std": cv_std_power,
            "params": popt_power,
            "func": lambda x: power_law_func(x, *popt_power),
        }
    except Exception as e:
        print(f"Error fitting power law model: {e}")

    # Find best model based on different criteria
    if not results:
        return None

    # Best by AIC (lower is better)
    best_aic = min(results.items(), key=lambda x: x[1]["aic"])

    # Best by BIC (lower is better)
    best_bic = min(results.items(), key=lambda x: x[1]["bic"])

    # Best by cross-validation MSE (lower is better)
    best_cv = min(results.items(), key=lambda x: x[1]["cv_mse"])

    # Best by R² (higher is better)
    best_r2 = max(results.items(), key=lambda x: x[1]["r2"])

    return {
        "results": results,
        "best_aic": best_aic,
        "best_bic": best_bic,
        "best_cv": best_cv,
        "best_r2": best_r2,
        "summary": {
            "best_aic_name": best_aic[0],
            "best_bic_name": best_bic[0],
            "best_cv_name": best_cv[0],
            "best_r2_name": best_r2[0],
        },
    }


# Set seaborn style for poster
sns.set_style("whitegrid")
# sns.set_palette("husl")
sns.set_context("poster")

base_dir = "plots/datascaling"
os.makedirs(base_dir, exist_ok=True)

# https://wandb.ai/andreas-burger/hip/workspace?nw=fdkkquw19gl

force_download = False
fit_polynomial = False
max_poly_degree = 3
force_fit_exponential = False
plot_marginal_improvement = False
multiple = 2

# Check if parquet file exists
parquet_file = "wandb_datascaling.parquet"

if os.path.exists(parquet_file) and not force_download:
    print("Loading existing parquet file...")
    runs_df = pd.read_parquet(parquet_file)
else:
    print("Downloading data from wandb...")
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs("andreas-burger/hip")

    list_of_runs = []
    for run in tqdm(runs):
        # filter by training.split=random
        if "split" not in run.config.get("training", {}):
            continue
        if run.config["training"]["split"] != "random":
            continue
        # if pltrainer.limit_val_batches == 80, skip
        if run.config["pltrainer"]["limit_val_batches"] == 80:
            continue

        # # Check runtime - skip runs that have been running for less than 24 hours
        # # Calculate runtime in hours
        # if run.state == "running":
        #     # For running jobs, calculate time since start
        #     created_at = datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))
        #     runtime_hours = (datetime.now(created_at.tzinfo) - created_at).total_seconds() / 3600
        #     if runtime_hours < 24:
        #         continue
        # else:
        #     # For finished jobs, check if they ran for at least 24 hours
        #     run.created_at
        #     else:
        #         # If we can't determine runtime, skip the run
        #         continue

        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_dict = run.summary._json_dict
        # Keep only columns containing "val"
        summary_dict = {
            k: v
            for k, v in summary_dict.items()
            if "val" in k or k == "num_train_samples"
        }

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_dict = {k: v for k, v in run.config.items() if not k.startswith("_")}
        # Recursively flatten dict
        config_dict = flatten_dict(config_dict)
        # Keep relevant config columns
        config_dict = {
            k: v
            for k, v in config_dict.items()
            if any(keyword in k for keyword in ["weight", "split", "dataset", "size"])
        }

        if "val-Loss E" not in summary_dict:
            # old runs or runs without required metrics
            continue

        print("Found:", run.name)

        # Get history for minimum values
        keys = ["val-Loss E", "val-Loss F"]
        if config_dict.get("training.hessian_loss_weight", 0) > 0:
            keys += ["val-MAE Hessian"]

        # Get history data
        history = run.history(
            samples=5_000, keys=keys, x_axis="_step", pandas=True, stream="default"
        )
        # get the minimum value for each key using pandas
        min_history = {k: float(history[k].min()) for k in keys if k in history.columns}
        print("Found minimum values:", min_history)

        # Overwrite summary with min values
        summary_dict.update(min_history)

        # .name is the human-readable name of the run.
        list_of_runs.append(
            {
                "name": run.name,
                **config_dict,
                **summary_dict,
            }
        )

    runs_df = pd.DataFrame(list_of_runs)

    # Save the downloaded data
    runs_df.to_parquet(parquet_file)
    print("Data saved to parquet file.")

# List the columns
# print("DataFrame columns:")
# print(runs_df.columns.tolist())

for losstype in ["Loss E", "Loss F", "MAE Hessian"]:
    human_name = {"Loss E": "Energy", "Loss F": "Force", "MAE Hessian": "Hessian"}[
        losstype
    ]

    # Work with wandb data
    df = runs_df.copy()

    # Print basic info about the dataset
    print(f"\nProcessing {losstype} data...")
    # print("Original dataset shape:", df.shape)
    # print("Column names:")
    # print(df.columns.tolist())

    # Filter for the specific loss type
    loss_col = f"val-{losstype}"
    if loss_col not in df.columns:
        print(f"Column {loss_col} not found, skipping {losstype}")
        continue

    # Extract statistics for each method
    stats_data = []
    for idx, row in df.iterrows():
        method_name = row["name"]
        dataset_size = row["num_train_samples"]
        loss_value = row[loss_col]

        if pd.notna(loss_value):
            # Check if method ends with "EF" (Energy-Force only)
            # is_ef = method_name.endswith("EF")
            is_ef = row["training.hessian_loss_weight"] < 0.1

            # # Extract dataset size from the method name
            # # Find the numeric part at the beginning (including scientific notation)
            # match = re.match(r"^([0-9.]+(?:e[+-]?[0-9]+)?)", method_name)
            # if match:
            #     dataset_size = float(match.group(1))
            # else:
            #     dataset_size = None

            stats_data.append(
                {
                    "Method": method_name,
                    "Last_Value": loss_value,
                    "Min_Value": loss_value,  # Already min from history
                    "Max_Value": loss_value,  # Same as min since we took min from history
                    "ef": is_ef,
                    "Dataset size": dataset_size,
                }
            )

    # Create new dataframe with methods as rows and statistics as columns
    df_stats = pd.DataFrame(stats_data)
    df_stats = df_stats.set_index("Method")

    # Save the filtered data
    csvfname = os.path.join(base_dir, f"loss_{human_name.lower()}.csv")
    # df_stats.to_csv(csvfname, index=False)
    # print(f"Filtered data saved to '{csvfname}'")

    # print("\nStatistics for each method:")
    # print(df_stats)

    # Filter out rows where dataset size is not None
    plot_data = df_stats.dropna(subset=["Dataset size"])

    # plot_data = df_stats[df_stats["training.hessian_loss_weight"] > 0.1]
    # ef_data = df_stats[df_stats["training.hessian_loss_weight"] < 0.1]

    # Create log-log plot of min_value vs dataset size
    plt.figure(figsize=(10, 8))

    # Create scatter plot with different colors for EF vs non-EF methods
    ef_data = plot_data[plot_data["ef"]]
    efh_data = plot_data[~plot_data["ef"]]

    # if there is two runs with the x value, pick the one with the lowest y value
    ef_data = (
        ef_data.sort_values("Dataset size")
        .groupby("Dataset size")
        .first()
        .reset_index()
    )
    efh_data = (
        efh_data.sort_values("Dataset size")
        .groupby("Dataset size")
        .first()
        .reset_index()
    )

    # if with_ef:
    plt.scatter(
        ef_data["Dataset size"],
        ef_data["Min_Value"],
        label="Energy-Force",
        marker="o",
        s=100,
        alpha=0.7,
    )
    plt.scatter(
        efh_data["Dataset size"],
        efh_data["Min_Value"],
        label="Energy-Force-Hessian",
        marker="s",
        s=100,
        alpha=0.7,
    )

    ###############################################################33
    # Statistical Model Comparison and Marginal Improvement Threshold
    ###############################################################33

    # Use statistical model comparison to find best fit
    if len(efh_data) and fit_polynomial > 2:  # Need at least 3 points for fitting
        # Sort data by dataset size for fitting
        sorted_data = efh_data.sort_values("Dataset size")
        x_fit = sorted_data["Dataset size"].values
        y_fit = sorted_data["Min_Value"].values

        if force_fit_exponential:
            print(f"\nForcing exponential fit for {losstype}...")
            # Force exponential model
            try:
                from scipy.optimize import curve_fit

                popt_exp, pcov_exp = curve_fit(
                    exponential_func, x_fit, y_fit, p0=[1, -1, 0], maxfev=10000
                )
                y_pred_exp = exponential_func(x_fit, *popt_exp)
                y_pred_exp_log = np.log10(y_pred_exp)
                y_fit_log = np.log10(y_fit)

                mse_exp = mean_squared_error(y_fit_log, y_pred_exp_log)
                r2_exp = r2_score(y_fit_log, y_pred_exp_log)

                # Create a mock model comparison result
                model_comparison = {
                    "results": {
                        "exponential": {
                            "type": "exponential",
                            "mse": mse_exp,
                            "r2": r2_exp,
                            "params": popt_exp,
                            "func": lambda x: exponential_func(x, *popt_exp),
                        }
                    },
                    "summary": {
                        "best_cv_name": "exponential",
                        "best_aic_name": "exponential",
                        "best_bic_name": "exponential",
                        "best_r2_name": "exponential",
                    },
                }
                print(f"Forced exponential fit: R² = {r2_exp:.4f}, MSE = {mse_exp:.6f}")
            except Exception as e:
                print(f"Failed to fit exponential model: {e}")
                model_comparison = None
        else:
            print(f"\nPerforming statistical model comparison for {losstype}...")
            model_comparison = compare_models(
                x_fit, y_fit, max_poly_degree=max_poly_degree
            )

        if model_comparison:
            # # Print comparison results
            # print("\nModel Comparison Results:")
            # print("=" * 50)
            # for model_name, model_info in model_comparison['results'].items():
            #     print(f"{model_name}:")
            #     print(f"  R² = {model_info['r2']:.4f}")
            #     print(f"  AIC = {model_info['aic']:.2f}")
            #     print(f"  BIC = {model_info['bic']:.2f}")
            #     print(f"  CV MSE = {model_info['cv_mse']:.6f} ± {model_info['cv_std']:.6f}")
            #     print()

            print("Best Models by Different Criteria:")
            print(f"  Best AIC: {model_comparison['summary']['best_aic_name']}")
            print(f"  Best BIC: {model_comparison['summary']['best_bic_name']}")
            print(f"  Best CV:  {model_comparison['summary']['best_cv_name']}")
            print(f"  Best R²:  {model_comparison['summary']['best_r2_name']}")

            # Use the best model by cross-validation (most robust)
            best_model_name = model_comparison["summary"]["best_cv_name"]
            best_model = model_comparison["results"][best_model_name]

            # Generate plot data for the best model
            x_plot = np.linspace(x_fit.min(), x_fit.max(), 100)

            if best_model["type"] == "polynomial":
                # For polynomial models, work in log space
                x_log_plot = np.log10(x_plot)
                y_log_plot = best_model["func"](x_log_plot)
                y_plot = 10**y_log_plot
                fit_label = f"Best fit ({best_model['degree']}-degree polynomial)"
            else:
                # For exponential/power law models, work in original space
                y_plot = best_model["func"](x_plot)
                if best_model["type"] == "exponential":
                    if force_fit_exponential:
                        fit_label = "Forced exponential fit"
                    else:
                        fit_label = "Best fit (exponential)"
                else:
                    fit_label = "Best fit (power law)"

            # Plot the best fit
            plt.plot(
                x_plot,
                y_plot,
                color="C1",  # Same color as second scatter plot (EFH)
                alpha=0.7,
                linewidth=2,
                label=fit_label,
            )

            # Calculate slope threshold for marginal improvement
            if plot_marginal_improvement:
                # Calculate slope threshold: less than 1% error improvement for 2x data increase
                # In log space: slope = (log(y2) - log(y1)) / (log(x2) - log(x1))
                # For 2x increase: slope = (log(y2) - log(y1)) / log(2)
                # 1% improvement: y2 = 0.99 * y1, so log(y2) - log(y1) = log(0.99)
                # slope_threshold = log(0.99) / log(2) ≈ -0.0145
                slope_threshold = np.log(0.99) / np.log(multiple)

                # Add marginal improvement threshold line
                if best_model["type"] == "polynomial":
                    # For polynomial models, work in log space
                    x_range = np.linspace(
                        np.log10(x_fit.min()), np.log10(x_fit.max()), 1000
                    )
                    y_range = best_model["func"](x_range)
                    slopes = np.gradient(y_range, x_range)

                    # Find intersection point where slope equals threshold
                    slope_diff = slopes - slope_threshold
                    sign_changes = np.where(np.diff(np.sign(slope_diff)))[0]

                    if len(sign_changes) > 0:
                        # Find the first point where slope becomes less negative than threshold
                        threshold_idx = sign_changes[0]
                        threshold_x = (
                            10 ** x_range[threshold_idx]
                        )  # Convert back to original scale

                        # Plot vertical line at threshold point
                        plt.axvline(
                            x=threshold_x,
                            color="orange",
                            linestyle=":",
                            alpha=0.8,
                            label=rf"Slope < {multiple}x data, 1% MAE",
                        )
                else:
                    # For exponential/power law, calculate slope in log space
                    x_log_range = np.log10(x_plot)
                    y_log_range = np.log10(y_plot)
                    slopes = np.gradient(y_log_range, x_log_range)

                    # Find intersection point where slope equals threshold
                    slope_diff = slopes - slope_threshold
                    sign_changes = np.where(np.diff(np.sign(slope_diff)))[0]

                    if len(sign_changes) > 0:
                        # Find the first point where slope becomes less negative than threshold
                        threshold_idx = sign_changes[0]
                        threshold_x = x_plot[threshold_idx]

                        # Plot vertical line at threshold point
                        plt.axvline(
                            x=threshold_x,
                            color="orange",
                            linestyle=":",
                            alpha=0.8,
                            label=r"Slope < 2x data, 1% MAE",
                        )
        else:
            print(f"Could not fit models for {losstype}")

    # Add horizontal line at 5% above the lowest loss
    min_loss = plot_data["Min_Value"].min()
    max_loss = plot_data["Max_Value"].max()
    threshold_5pct = min_loss * 1.05
    threshold_1pct = (max_loss - min_loss) * 0.01 + min_loss

    # Find first datapoint that crosses below each threshold
    # Sort data by dataset size to find crossing points
    sorted_plot_data = plot_data.sort_values("Dataset size")

    # Find first point below 5% threshold
    first_cross_5pct = None
    for idx, row in sorted_plot_data.iterrows():
        if row["Min_Value"] <= threshold_5pct:
            first_cross_5pct = int(row["Dataset size"])
            break
    # Create labels with crossing points
    label_5pct = r"5% MAE$_{\text{min}}$"
    if first_cross_5pct is not None:
        label_5pct += f" (≤{first_cross_5pct:.0e})"
    plt.axhline(
        y=threshold_5pct,
        color="darkgray",
        linestyle="--",
        alpha=0.7,
        label=label_5pct,
    )

    # # Find first point below 1% threshold
    # first_cross_1pct = None
    # for idx, row in sorted_plot_data.iterrows():
    #     if row["Min_Value"] <= threshold_1pct:
    #         first_cross_1pct = int(row["Dataset size"])
    #         break
    # label_1pct = r"1% MAE$_{\text{min-max}}$"
    # if first_cross_1pct is not None:
    #     label_1pct += f" (≤{first_cross_1pct:.0e})"
    # plt.axhline(
    #     y=threshold_1pct,
    #     color="lightgray",
    #     linestyle="--",
    #     alpha=0.7,
    #     label=label_1pct,
    # )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Training Samples")
    plt.ylabel(f"{human_name} MAE (validation)")
    plt.title(f"{human_name} Error vs Dataset Size")
    plt.legend(
        # title='Training Loss',
        frameon=True,
        edgecolor="none",
        fontsize=12,
    )
    plt.grid(True, alpha=0.3, which="major")
    plt.grid(True, alpha=0.1, which="minor")
    plt.tight_layout()
    fname = os.path.join(base_dir, f"log_log_{human_name.lower()}_mae.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    # plt.show()

    print(f"\nLog-log plot saved as '{fname}'")
