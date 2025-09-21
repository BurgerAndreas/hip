# if hesspred in the name, rename to EquiformerV2
# hessian_method predict -> learned

# metrics:
# time_incltransform -> time
# eigval1_mae -> lambda1
# eigvec1_cos -> CosSim

import csv
from pathlib import Path


CSV_PATH = Path("/ssd/Code/hip/results/eval_horm_time_wandb_export.csv")


def _to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _fmt(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"


def _fmt_time_ms(value: float) -> str:
    return _fmt(value, 1)


def _bold(s: str, do_bold: bool) -> str:
    return f"\\textbf{{ {s} }}" if do_bold else s


def load_rows(csv_path: Path):
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Normalize names where needed
    for r in rows:
        name = (r.get("Name") or "").lower()
        if "hesspred" in name:
            r["model_name"] = "EquiformerV2"
    return rows


def select_by_method(rows, method: str):
    return [r for r in rows if (r.get("hessian_method") or "").lower() == method]


def pick_metrics(r):
    return {
        "model": r.get("model_name", ""),
        "hessian": _to_float(r.get("hessian_mae", "nan")),
        "eigvals": _to_float(r.get("eigval_mae", "nan")),
        "cos1": _to_float(r.get("eigvec1_cos", "nan")),
        "lambda1": _to_float(r.get("eigval1_mae", "nan")),
        "time_ms": _to_float(r.get("time_incltransform", r.get("time", "nan"))),
    }


def main() -> None:
    rows = load_rows(CSV_PATH)

    # Group rows
    auto_rows = select_by_method(rows, "autograd")
    learned_rows = select_by_method(rows, "predict")

    # Order for Autograd block
    desired_order = ["AlphaNet", "LEFTNet", "LEFTNet-df", "EquiformerV2"]
    auto_by_model = {r.get("model_name"): pick_metrics(r) for r in auto_rows}
    auto_ordered = [auto_by_model[m] for m in desired_order if m in auto_by_model]

    # Learned (Ours): expect EquiformerV2
    learned_metrics = None
    if learned_rows:
        # If multiple, pick the first
        learned_metrics = pick_metrics(learned_rows[0])

    # Begin LaTeX table
    lines = []
    lines.append("\\begin{tabular}{llccccc}")
    lines.append("\\hline")
    lines.append(
        "\\multirow{2}{*}{Hessian} & \\multirow{2}{*}{Model} & Hessian $\\downarrow$  & Eigenvalues $\\downarrow$ & CosSim $\\evec_1$ $\\uparrow$ & $\\eval_1$ $\\downarrow$ & Time $\\downarrow$ \\\\"
    )
    lines.append(" & & eV/\\AA$^2$ & eV/\\AA$^2$ & unitless & eV/\\AA$^2$ & ms \\\\")
    lines.append("\\hline")

    # Autograd block
    lines.append("\\multirow{4}{*}{Autograd} ")
    for idx, m in enumerate(auto_ordered):
        prefix = " & " if idx > 0 else " & "
        lines.append(
            f"{prefix}{m['model']} & {_fmt(m['hessian'])} & {_fmt(m['eigvals'])} & {_fmt(m['cos1'])} & {_fmt(m['lambda1'])} & {_fmt_time_ms(m['time_ms'])} \\\\"
        )
    lines.append("\\hline")

    # Learned (Ours) block
    if learned_metrics is not None:
        lm = learned_metrics
        lines.append(
            "\\multirow{1}{*}{Learned (Ours)} & EquiformerV2 "
            + f"& {_bold(_fmt(lm['hessian']), True)} "
            + f"& {_bold(_fmt(lm['eigvals']), True)} "
            + f"& {_bold(_fmt(lm['cos1']), True)} "
            + f"& {_bold(_fmt(lm['lambda1']), True)} "
            + f"& {_bold(_fmt_time_ms(lm['time_ms']), True)} \\\\"
        )

    lines.append("")
    lines.append("\\end{tabular}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
