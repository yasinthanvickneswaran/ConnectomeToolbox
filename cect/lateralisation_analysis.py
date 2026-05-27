"""
Quantify lateralisation for each bilateral neuron class in the C. elegans connectome.

Neurons ending in L and R form a bilateral pair whose class name is the shared prefix.
  e.g. ALNL + ALNR  -> class ALN
       AVKL + AVKR  -> class AVK
       RMDDL + RMDDR -> class RMDD

For each class we compare the outgoing connections of the L and R neurons:
  - A connection L->X is symmetric  if R->X' also exists  (X' = contralateral of X)
  - A connection L->X is left-only  if R->X' does not exist
  - A connection R->Y is right-only if L->Y' does not exist

Lateralisation % = (left_only + right_only) / (symmetric + left_only + right_only) * 100

A score of 0 % means perfectly symmetric; 100 % means every connection is unique to one side.

Usage:
    python -m cect.lateralisation_analysis           # run all datasets, show comparison plot
    python -m cect.lateralisation_analysis --quick   # Cook2019Herm only, single-dataset plot
    python -m cect.lateralisation_analysis -nogui    # text output only
"""

import sys
import importlib
from collections import defaultdict

from cect.Cells import (
    get_contralateral_cell,
    is_bilateral_left,
    get_SIM_class,
)
from cect.ConnectomeDataset import ConnectomeDataset


CLASS_COLORS = {
    "Sensory": "#e07b54",
    "Interneuron": "#6baed6",
    "Motor": "#74c476",
    "Other": "#aaaaaa",
}

MIN_CONNECTIONS = 3

# Dataset registry ordered for developmental trajectory:
# Witvliet L1-early-adult larvae (1-6) -> Yim2024 dauer -> Witvliet late-adult (7-8) -> adult herm/male
# Yim2024 placed before Witvliet7 so the larva->dauer->adult transition is visible on the x-axis.
DATASETS = {
    "Witvliet1":    "cect.WitvlietDataReader1",
    "Witvliet2":    "cect.WitvlietDataReader2",
    "Witvliet3":    "cect.WitvlietDataReader3",
    "Witvliet4":    "cect.WitvlietDataReader4",
    "Witvliet5":    "cect.WitvlietDataReader5",
    "Witvliet6":    "cect.WitvlietDataReader6",
    "Yim2024":      "cect.Yim2024DataReader",
    "Witvliet7":    "cect.WitvlietDataReader7",
    "Witvliet8":    "cect.WitvlietDataReader8",
    "Cook2019Herm": "cect.Cook2019HermReader",
    "Cook2019Male": "cect.Cook2019MaleReader",
}

# Subset for quick runs
DATASETS_QUICK = {
    "Witvliet1":    DATASETS["Witvliet1"],
    "Witvliet6":    DATASETS["Witvliet6"],
    "Yim2024":      DATASETS["Yim2024"],
    "Cook2019Herm": DATASETS["Cook2019Herm"],
}


def get_functional_class(cell: str) -> str:
    """Classify a cell as Sensory / Interneuron / Motor / Other.

    Delegates to cect.Cells.get_SIM_class, which is the canonical source of truth,
    and normalises "Motorneuron" to "Motor" to match CLASS_COLORS keys.
    """
    cls = get_SIM_class(cell)
    return "Motor" if cls == "Motorneuron" else cls


def _safe_contralateral(cell: str) -> str:
    """Return the contralateral cell; fall back to the same cell for unknowns."""
    try:
        return get_contralateral_cell(cell)
    except Exception:
        return cell


def pick_chemical_synclass(cds: ConnectomeDataset) -> str:
    """Return the key that represents chemical synapses in this dataset."""
    for candidate in ("Generic_CS", "Chemical", "Functional"):
        if candidate in cds.connections:
            return candidate
    for key in cds.connections:
        if not any(s in key for s in ("GJ", "Electrical", "Contact", "Gap")):
            return key
    return list(cds.connections.keys())[0]


def compute_pair_lateralisation(cds: ConnectomeDataset, synclass: str) -> dict:
    """
    For each bilateral class (e.g. ALN = ALNL + ALNR), compare outgoing
    connectivity of the two neurons and return a lateralisation score.

    Returns
    -------
    dict mapping class_name (e.g. "ALN") -> {
        'left'             : left neuron name (e.g. "ALNL"),
        'right'            : right neuron name (e.g. "ALNR"),
        'symmetric'        : connections present on both sides,
        'left_only'        : connections present only in the left neuron,
        'right_only'       : connections present only in the right neuron,
        'total'            : symmetric + left_only + right_only,
        'lateralisation_pct': (left_only + right_only) / total * 100,
        'functional_class' : Sensory / Interneuron / Motor / Other,
    }
    """
    if synclass not in cds.connections:
        print(f"  Synclass '{synclass}' not found. Available: {list(cds.connections.keys())}")
        return {}

    node_idx = {node: i for i, node in enumerate(cds.nodes)}
    conn_array = cds.connections[synclass]
    results = {}

    for node in cds.nodes:
        if not is_bilateral_left(node):
            continue

        left = node
        right = node[:-1] + "R"

        if right not in node_idx:
            continue

        left_i = node_idx[left]
        right_i = node_idx[right]

        left_targets = {
            cds.nodes[j]
            for j in range(len(cds.nodes))
            if j != left_i and conn_array[left_i, j] > 0
        }
        right_targets = {
            cds.nodes[j]
            for j in range(len(cds.nodes))
            if j != right_i and conn_array[right_i, j] > 0
        }

        if not left_targets and not right_targets:
            continue

        right_targets_mirrored = {_safe_contralateral(t) for t in right_targets}

        symmetric = 0
        left_only = 0
        right_only = 0

        for t in left_targets:
            if t in right_targets_mirrored:
                symmetric += 1
            else:
                left_only += 1

        for t in right_targets:
            if _safe_contralateral(t) not in left_targets:
                right_only += 1

        total = symmetric + left_only + right_only
        if total < MIN_CONNECTIONS:
            continue

        class_name = left[:-1]
        results[class_name] = {
            "left": left,
            "right": right,
            "symmetric": symmetric,
            "left_only": left_only,
            "right_only": right_only,
            "total": total,
            "lateralisation_pct": 100 * (left_only + right_only) / total,
            "functional_class": get_functional_class(left),
        }

    return results


def aggregate_by_functional_class(pair_results: dict) -> dict:
    """Pool bilateral-pair scores into functional class totals."""
    class_data = defaultdict(lambda: {"symmetric": 0, "left_only": 0, "right_only": 0, "pairs": []})

    for class_name, info in pair_results.items():
        fc = info["functional_class"]
        class_data[fc]["symmetric"] += info["symmetric"]
        class_data[fc]["left_only"] += info["left_only"]
        class_data[fc]["right_only"] += info["right_only"]
        class_data[fc]["pairs"].append((class_name, info["lateralisation_pct"], info["total"]))

    summary = {}
    for fc, data in class_data.items():
        total = data["symmetric"] + data["left_only"] + data["right_only"]
        if total > 0:
            summary[fc] = {
                "lateralisation_pct": 100 * (data["left_only"] + data["right_only"]) / total,
                "symmetric": data["symmetric"],
                "left_only": data["left_only"],
                "right_only": data["right_only"],
                "total": total,
                "pairs_ranked": sorted(data["pairs"], key=lambda x: x[1], reverse=True),
            }
    return summary


def print_results(pair_results: dict, fc_summary: dict, dataset_name: str, synclass: str):
    print(f"\n{'='*66}")
    print(f"  LATERALISATION ANALYSIS  |  {dataset_name}  |  {synclass}")
    print(f"{'='*66}")

    print("\nFUNCTIONAL CLASS SUMMARY")
    print(f"  {'Class':<14} {'Score':>8}   {'Left-only':>10}  {'Right-only':>11}  {'Symmetric':>10}  {'Total':>7}")
    print("  " + "-" * 64)
    ranked_fc = sorted(fc_summary.items(), key=lambda x: x[1]["lateralisation_pct"], reverse=True)
    for fc, data in ranked_fc:
        print(
            f"  {fc:<14} {data['lateralisation_pct']:>7.1f}%"
            f"   {data['left_only']:>10}  {data['right_only']:>11}"
            f"  {data['symmetric']:>10}  {data['total']:>7}"
        )

    print(f"\n\nBILATERAL CLASS RANKING  (>= {MIN_CONNECTIONS} connections, most -> least lateralised)")
    print(f"  {'Class':<8} {'L neuron':<10} {'R neuron':<10} {'Func class':<13}"
          f" {'Score':>8}   {'L-only':>7}  {'R-only':>7}  {'Shared':>7}  {'Total':>6}")
    print("  " + "-" * 84)

    ranked = sorted(pair_results.items(), key=lambda x: x[1]["lateralisation_pct"], reverse=True)
    for class_name, info in ranked:
        print(
            f"  {class_name:<8} {info['left']:<10} {info['right']:<10}"
            f" {info['functional_class']:<13}"
            f" {info['lateralisation_pct']:>7.1f}%"
            f"   {info['left_only']:>7}  {info['right_only']:>7}"
            f"  {info['symmetric']:>7}  {info['total']:>6}"
        )

    print(f"\n  {len(pair_results)} bilateral classes analysed.")


def plot_results(pair_results: dict, fc_summary: dict, dataset_name: str, synclass: str):
    """Single-dataset plot: functional-class bar chart + per-pair scatter."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Functional class lateralisation", "Per bilateral-class lateralisation"),
        horizontal_spacing=0.1,
    )

    ranked_fc = sorted(fc_summary.items(), key=lambda x: x[1]["lateralisation_pct"], reverse=True)
    for fc, data in ranked_fc:
        fig.add_trace(
            go.Bar(
                x=[fc],
                y=[data["lateralisation_pct"]],
                name=fc,
                marker_color=CLASS_COLORS.get(fc, "#aaaaaa"),
                text=[f"n={data['total']}"],
                textposition="outside",
                hovertemplate=(
                    f"<b>{fc}</b><br>"
                    f"Lateralisation: {data['lateralisation_pct']:.1f}%<br>"
                    f"Left-only: {data['left_only']}<br>"
                    f"Right-only: {data['right_only']}<br>"
                    f"Symmetric: {data['symmetric']}<br>"
                    f"Total: {data['total']}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=1,
        )

    fig.add_hline(y=50, line_dash="dash", line_color="grey", line_width=1, opacity=0.5, row=1, col=1)

    top8_names = {
        name for name, _ in
        sorted(pair_results.items(), key=lambda x: x[1]["lateralisation_pct"], reverse=True)[:8]
    }

    for fc in list(CLASS_COLORS.keys()):
        pairs = [(name, info) for name, info in pair_results.items() if info["functional_class"] == fc]
        if not pairs:
            continue

        fig.add_trace(
            go.Scatter(
                x=[info["total"] for _, info in pairs],
                y=[info["lateralisation_pct"] for _, info in pairs],
                mode="markers+text",
                name=fc,
                text=[name if name in top8_names else "" for name, _ in pairs],
                textposition="top right",
                textfont=dict(size=9, color="#222222"),
                marker=dict(
                    color=CLASS_COLORS.get(fc, "#aaaaaa"),
                    size=8,
                    opacity=0.8,
                    line=dict(color="white", width=0.5),
                ),
                customdata=[
                    [name, info["left"], info["right"], info["left_only"],
                     info["right_only"], info["symmetric"], info["total"]]
                    for name, info in pairs
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{customdata[1]} / %{customdata[2]})<br>"
                    "Lateralisation: %{y:.1f}%<br>"
                    "Left-only: %{customdata[3]}<br>"
                    "Right-only: %{customdata[4]}<br>"
                    "Symmetric: %{customdata[5]}<br>"
                    "Total: %{customdata[6]}"
                    "<extra></extra>"
                ),
            ),
            row=1, col=2,
        )

    fig.add_hline(y=50, line_dash="dash", line_color="grey", line_width=1, opacity=0.5, row=1, col=2)

    fig.update_layout(
        title_text=f"Bilateral lateralisation  --  {dataset_name}  ({synclass})",
        height=580,
        legend=dict(title="Functional class", x=1.01, y=0.5),
        plot_bgcolor="white",
    )
    fig.update_yaxes(title_text="Lateralised connections (%)", range=[0, 108], row=1, col=1,
                     gridcolor="#eeeeee", gridwidth=1)
    fig.update_yaxes(title_text="Lateralisation score (%)", range=[-5, 108], row=1, col=2,
                     gridcolor="#eeeeee", gridwidth=1)
    fig.update_xaxes(row=1, col=2,
                     title_text=f"Total connections (L + R, threshold >= {MIN_CONNECTIONS})",
                     gridcolor="#eeeeee", gridwidth=1)

    fig.show()


def plot_comparison(all_results: dict):
    """
    Compare lateralisation scores across datasets, one subplot per functional class.

    Mirrors the structure of Analysis.plot_symmetry: x-axis = datasets,
    y-axis = lateralisation %, one line per functional class per subplot.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    functional_classes = ["Sensory", "Interneuron", "Motor"]
    dataset_names = list(all_results.keys())

    fig = make_subplots(
        rows=1, cols=len(functional_classes),
        subplot_titles=[f"{fc} neurons" for fc in functional_classes],
        horizontal_spacing=0.08,
    )

    for col_idx, fc in enumerate(functional_classes, start=1):
        y_vals = []
        n_vals = []

        for ds_name in dataset_names:
            fc_summary = all_results[ds_name]["fc_summary"]
            entry = fc_summary.get(fc)
            y_vals.append(entry["lateralisation_pct"] if entry else None)
            n_vals.append(entry["total"] if entry else 0)

        fig.add_trace(
            go.Scatter(
                x=dataset_names,
                y=y_vals,
                mode="lines+markers",
                name=fc,
                line=dict(color=CLASS_COLORS[fc]),
                marker=dict(color=CLASS_COLORS[fc], size=8),
                customdata=[[n] for n in n_vals],
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"{fc} lateralisation: %{{y:.1f}}%<br>"
                    "Connections: %{customdata[0]}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=col_idx,
        )

        fig.add_hline(y=50, line_dash="dash", line_color="grey", line_width=1,
                      opacity=0.4, row=1, col=col_idx)

        fig.update_xaxes(tickangle=45, row=1, col=col_idx, gridcolor="#eeeeee")
        fig.update_yaxes(
            title_text="Lateralisation (%)" if col_idx == 1 else "",
            range=[0, 108],
            row=1, col=col_idx,
            gridcolor="#eeeeee",
        )

    fig.update_layout(
        title_text=f"Bilateral lateralisation across {len(all_results)} datasets",
        height=520,
        plot_bgcolor="white",
    )
    fig.show()


def run(get_instance_fn, dataset_name: str):
    """Load a dataset, compute lateralisation, print results, and return summary dicts."""
    cds = get_instance_fn()
    synclass = pick_chemical_synclass(cds)

    print(f"\nDataset : {dataset_name}")
    print(f"Nodes   : {len(cds.nodes)}")
    print(f"Synclass: {synclass}  (from {list(cds.connections.keys())})")

    pair_results = compute_pair_lateralisation(cds, synclass)
    if not pair_results:
        print("  No bilateral pairs found -- check synclass and dataset.")
        return None, None, None

    fc_summary = aggregate_by_functional_class(pair_results)
    print_results(pair_results, fc_summary, dataset_name, synclass)
    return pair_results, fc_summary, synclass


def run_all(datasets: dict = None) -> dict:
    """
    Run lateralisation analysis across multiple datasets.

    Uses importlib to load each reader module by its dotted module path,
    mirroring the pattern in cect.Comparison.generate_comparison_page.

    Parameters
    ----------
    datasets : dict, optional
        Maps display name -> module path string. Defaults to DATASETS.

    Returns
    -------
    dict
        Maps dataset_name -> {'pair_results': ..., 'fc_summary': ..., 'synclass': ...}
    """
    if datasets is None:
        datasets = DATASETS

    all_results = {}
    for dataset_name, module_path in datasets.items():
        print(f"\n{'-'*50}\nLoading {dataset_name} ...")
        try:
            mod = importlib.import_module(module_path)
            pair_results, fc_summary, synclass = run(mod.get_instance, dataset_name)
            if pair_results:
                all_results[dataset_name] = {
                    "pair_results": pair_results,
                    "fc_summary": fc_summary,
                    "synclass": synclass,
                }
        except Exception as exc:
            print(f"  Skipping {dataset_name}: {exc}")

    return all_results


def compute_developmental_trajectory(all_results: dict, min_datasets: int = 3) -> dict:
    """
    For each bilateral class, track its lateralisation score across all datasets
    and compute how much it changes over development.

    Only includes classes that appear in at least `min_datasets` datasets.

    Returns
    -------
    dict mapping class_name -> {
        'scores'          : {dataset_name: lateralisation_pct, ...},
        'range'           : max_score - min_score,
        'mean'            : mean lateralisation across datasets,
        'functional_class': Sensory / Interneuron / Motor / Other,
        'n_datasets'      : number of datasets where this class appears,
    }
    """
    trajectory = {}

    for ds_name, ds in all_results.items():
        for class_name, info in ds["pair_results"].items():
            if class_name not in trajectory:
                trajectory[class_name] = {
                    "scores": {},
                    "functional_class": info["functional_class"],
                }
            trajectory[class_name]["scores"][ds_name] = info["lateralisation_pct"]

    result = {}
    for class_name, data in trajectory.items():
        scores = list(data["scores"].values())
        if len(scores) < min_datasets:
            continue
        result[class_name] = {
            "scores": data["scores"],
            "range": max(scores) - min(scores),
            "mean": sum(scores) / len(scores),
            "functional_class": data["functional_class"],
            "n_datasets": len(scores),
        }

    return result


def plot_trajectory(all_results: dict, n: int = 20):
    """
    Plot lateralisation trajectories across development for the top-n most-changed
    and bottom-n least-changed bilateral classes.

    Each line is one bilateral class, coloured by functional class.
    Datasets appear on the x-axis in the order they were run (developmental order).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    traj = compute_developmental_trajectory(all_results)
    if not traj:
        print("  No trajectory data to plot.")
        return

    dataset_names = list(all_results.keys())
    ranked = sorted(traj.items(), key=lambda x: x[1]["range"], reverse=True)

    top = ranked[:n]
    bottom = ranked[-n:]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Top {n} most-changed bilateral classes",
            f"Bottom {n} most-stable bilateral classes",
        ),
        horizontal_spacing=0.1,
    )

    def add_traces(pairs, col, show_legend):
        seen_fc = set()
        for class_name, data in pairs:
            fc = data["functional_class"]
            color = CLASS_COLORS.get(fc, "#aaaaaa")
            x_pts = [ds for ds in dataset_names if ds in data["scores"]]
            y_pts = [data["scores"][ds] for ds in x_pts]
            # Label only the last point on each line so the panel stays readable
            n_pts = len(x_pts)
            text_labels = [""] * (n_pts - 1) + [class_name] if n_pts > 0 else []
            fig.add_trace(
                go.Scatter(
                    x=x_pts,
                    y=y_pts,
                    mode="lines+markers+text",
                    name=fc,
                    legendgroup=fc,
                    showlegend=(show_legend and fc not in seen_fc),
                    line=dict(color=color, width=1.5),
                    marker=dict(color=color, size=5),
                    text=text_labels,
                    textposition="middle right",
                    textfont=dict(size=8, color=color),
                    hovertemplate=(
                        f"<b>{class_name}</b> ({fc})<br>"
                        "Dataset: %{x}<br>"
                        "Lateralisation: %{y:.1f}%"
                        "<extra></extra>"
                    ),
                ),
                row=1, col=col,
            )
            seen_fc.add(fc)

    add_traces(top, col=1, show_legend=True)
    add_traces(bottom, col=2, show_legend=False)

    for col in (1, 2):
        fig.add_hline(y=50, line_dash="dash", line_color="grey",
                      line_width=1, opacity=0.4, row=1, col=col)
        fig.update_xaxes(tickangle=45, row=1, col=col, gridcolor="#eeeeee")
        fig.update_yaxes(
            title_text="Lateralisation (%)" if col == 1 else "",
            range=[-5, 108],
            row=1, col=col,
            gridcolor="#eeeeee",
        )

    fig.update_layout(
        title_text=f"Bilateral class trajectories across development  (n={n} each end)",
        height=560,
        plot_bgcolor="white",
        legend=dict(title="Functional class"),
        margin=dict(r=100),  # extra right margin for end-of-line labels
    )
    fig.show()

    # Print ranked tables
    print(f"\nTOP {n} MOST-CHANGED bilateral classes (largest range across datasets)")
    print(f"  {'Class':<8} {'Func class':<13} {'Range':>7}  {'Mean':>7}  {'Datasets':>8}")
    print("  " + "-" * 50)
    for class_name, data in top:
        print(f"  {class_name:<8} {data['functional_class']:<13} "
              f"{data['range']:>6.1f}%  {data['mean']:>6.1f}%  {data['n_datasets']:>8}")

    print(f"\nBOTTOM {n} MOST-STABLE bilateral classes (smallest range across datasets)")
    print(f"  {'Class':<8} {'Func class':<13} {'Range':>7}  {'Mean':>7}  {'Datasets':>8}")
    print("  " + "-" * 50)
    for class_name, data in bottom:
        print(f"  {class_name:<8} {data['functional_class']:<13} "
              f"{data['range']:>6.1f}%  {data['mean']:>6.1f}%  {data['n_datasets']:>8}")


def compute_dauer_adult_shift(
    all_results: dict,
    dauer_key: str = "Yim2024",
    adult_key: str = "Cook2019Herm",
) -> dict:
    """
    Compare per-class lateralisation between the dauer stage and the adult hermaphrodite.

    Only includes bilateral classes present in *both* datasets so the comparison is direct.

    Returns
    -------
    dict mapping class_name -> {
        'dauer'            : lateralisation_pct in dauer,
        'adult'            : lateralisation_pct in adult,
        'delta'            : adult - dauer  (+ = more lateralised as adult),
        'abs_delta'        : abs(delta),
        'functional_class' : Sensory / Interneuron / Motor / Other,
    }
    """
    if dauer_key not in all_results:
        print(f"  Warning: '{dauer_key}' not in results - skipping dauer/adult shift analysis.")
        return {}
    if adult_key not in all_results:
        print(f"  Warning: '{adult_key}' not in results - skipping dauer/adult shift analysis.")
        return {}

    dauer_pairs = all_results[dauer_key]["pair_results"]
    adult_pairs = all_results[adult_key]["pair_results"]
    common = set(dauer_pairs.keys()) & set(adult_pairs.keys())

    result = {}
    for class_name in common:
        d_pct = dauer_pairs[class_name]["lateralisation_pct"]
        a_pct = adult_pairs[class_name]["lateralisation_pct"]
        result[class_name] = {
            "dauer": d_pct,
            "adult": a_pct,
            "delta": a_pct - d_pct,
            "abs_delta": abs(a_pct - d_pct),
            "functional_class": dauer_pairs[class_name]["functional_class"],
        }
    return result


def plot_dauer_adult_shift(
    all_results: dict,
    n_annotate: int = 10,
    dauer_key: str = "Yim2024",
    adult_key: str = "Cook2019Herm",
):
    """
    Scatter plot of dauer vs adult hermaphrodite lateralisation for each bilateral class.

    - x-axis  : lateralisation % in dauer (Yim2024)
    - y-axis  : lateralisation % in adult hermaphrodite (Cook2019Herm)
    - Diagonal: y = x  (no change)
    - Points above diagonal → more lateralised as adult
    - Points below diagonal → more symmetric as adult
    - Labels the n_annotate most-changed and n_annotate most-stable classes.

    Also prints ranked text tables.
    """
    import plotly.graph_objects as go

    shift = compute_dauer_adult_shift(all_results, dauer_key, adult_key)
    if not shift:
        return

    sorted_by_change = sorted(shift.items(), key=lambda x: x[1]["abs_delta"], reverse=True)
    most_changed = {name for name, _ in sorted_by_change[:n_annotate]}
    most_stable = {name for name, _ in sorted_by_change[-n_annotate:]}
    annotate_names = most_changed | most_stable

    fig = go.Figure()

    # y = x reference line
    fig.add_trace(go.Scatter(
        x=[0, 105], y=[0, 105],
        mode="lines",
        line=dict(color="grey", dash="dash", width=1),
        showlegend=False,
        hoverinfo="skip",
    ))
    # Shaded region: above diagonal = adult more lateralised, below = adult more symmetric
    fig.add_annotation(
        x=85, y=65, text="More symmetric<br>as adult", showarrow=False,
        font=dict(size=10, color="#888888"), xref="x", yref="y",
    )
    fig.add_annotation(
        x=30, y=75, text="More lateralised<br>as adult", showarrow=False,
        font=dict(size=10, color="#888888"), xref="x", yref="y",
    )

    for fc in CLASS_COLORS:
        pairs = [(name, info) for name, info in shift.items() if info["functional_class"] == fc]
        if not pairs:
            continue
        fig.add_trace(go.Scatter(
            x=[info["dauer"] for _, info in pairs],
            y=[info["adult"] for _, info in pairs],
            mode="markers+text",
            name=fc,
            text=[name if name in annotate_names else "" for name, _ in pairs],
            textposition="top right",
            textfont=dict(size=9),
            marker=dict(
                color=CLASS_COLORS[fc],
                size=10,
                opacity=0.8,
                line=dict(color="white", width=0.5),
                symbol=[
                    "star" if name in most_changed else
                    ("diamond" if name in most_stable else "circle")
                    for name, _ in pairs
                ],
            ),
            customdata=[
                [name, info["delta"], info["abs_delta"]]
                for name, info in pairs
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"Dauer ({dauer_key}): %{{x:.1f}}%<br>"
                f"Adult ({adult_key}): %{{y:.1f}}%<br>"
                "Delta: %{customdata[1]:+.1f}%"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title_text=(
            f"Dauer-to-adult lateralisation shift  ({dauer_key} → {adult_key})<br>"
            f"<sup>Stars = most changed (|delta| > threshold) | Diamonds = most stable | "
            f"n={len(shift)} shared bilateral classes</sup>"
        ),
        xaxis_title=f"Lateralisation % in dauer ({dauer_key})",
        yaxis_title=f"Lateralisation % in adult hermaphrodite ({adult_key})",
        height=650,
        width=700,
        plot_bgcolor="white",
        legend=dict(title="Functional class"),
    )
    fig.update_xaxes(range=[-5, 110], gridcolor="#eeeeee", zeroline=False)
    fig.update_yaxes(range=[-5, 110], gridcolor="#eeeeee", zeroline=False)
    fig.show()

    # Text output
    print(f"\nMOST-CHANGED bilateral classes: {dauer_key} -> {adult_key}")
    print(f"  {'Class':<8} {'Func class':<13} {dauer_key:>10} {adult_key:>13} {'Delta':>8}")
    print("  " + "-" * 60)
    for class_name, data in sorted_by_change[:n_annotate]:
        print(
            f"  {class_name:<8} {data['functional_class']:<13}"
            f"  {data['dauer']:>8.1f}%  {data['adult']:>11.1f}%  {data['delta']:>+7.1f}%"
        )

    print(f"\nMOST-STABLE bilateral classes: {dauer_key} -> {adult_key}")
    print(f"  {'Class':<8} {'Func class':<13} {dauer_key:>10} {adult_key:>13} {'Delta':>8}")
    print("  " + "-" * 60)
    for class_name, data in sorted_by_change[-n_annotate:]:
        print(
            f"  {class_name:<8} {data['functional_class']:<13}"
            f"  {data['dauer']:>8.1f}%  {data['adult']:>11.1f}%  {data['delta']:>+7.1f}%"
        )


def main():
    datasets = DATASETS_QUICK if "--quick" in sys.argv else DATASETS

    all_results = run_all(datasets)

    if all_results and "-nogui" not in sys.argv:
        if len(all_results) == 1:
            ds_name = next(iter(all_results))
            ds = all_results[ds_name]
            plot_results(ds["pair_results"], ds["fc_summary"], ds_name, ds["synclass"])
        else:
            plot_comparison(all_results)
            plot_trajectory(all_results, n=20)
            plot_dauer_adult_shift(all_results, n_annotate=10)


if __name__ == "__main__":
    main()
