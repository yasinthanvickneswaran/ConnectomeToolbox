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
    python -m cect.lateralisation_analysis           # show plots
    python -m cect.lateralisation_analysis -nogui    # text output only
"""

import sys
from collections import defaultdict

from cect.Cells import (
    get_contralateral_cell,
    is_bilateral_left,
    SENSORY_NEURONS_NONPHARYNGEAL_COOK,
    INTERNEURONS_NONPHARYNGEAL_COOK,
    MOTORNEURONS_NONPHARYNGEAL_COOK,
)
from cect.ConnectomeDataset import ConnectomeDataset


FUNCTIONAL_CLASSES = {
    "Sensory": set(SENSORY_NEURONS_NONPHARYNGEAL_COOK),
    "Interneuron": set(INTERNEURONS_NONPHARYNGEAL_COOK),
    "Motor": set(MOTORNEURONS_NONPHARYNGEAL_COOK),
}

CLASS_COLORS = {
    "Sensory": "#e07b54",
    "Interneuron": "#6baed6",
    "Motor": "#74c476",
    "Other": "#aaaaaa",
}

MIN_CONNECTIONS = 3  # minimum total connections (L+R combined) to include a class


def get_functional_class(cell):
    for cls, cells in FUNCTIONAL_CLASSES.items():
        if cell in cells:
            return cls
    return "Other"


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

        # Outgoing targets for each side (excluding self)
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

        # Mirror of each right target into "left space"
        # right connects to Y  =>  left equivalent would connect to contralateral(Y)
        right_targets_mirrored = {get_contralateral_cell(t) for t in right_targets}

        symmetric = 0
        left_only = 0
        right_only = 0

        for t in left_targets:
            if t in right_targets_mirrored:
                symmetric += 1
            else:
                left_only += 1

        for t in right_targets:
            if get_contralateral_cell(t) not in left_targets:
                right_only += 1

        total = symmetric + left_only + right_only
        if total < MIN_CONNECTIONS:
            continue

        class_name = left[:-1]  # e.g. "ALNL"[:-1] = "ALN"
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

    # --- Functional class summary ---
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

    # --- Per bilateral-class table ---
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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Functional class lateralisation", "Per bilateral-class lateralisation"),
        horizontal_spacing=0.1,
    )

    # --- Left panel: functional class bar chart ---
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

    # 50% reference line on bar chart
    fig.add_hline(y=50, line_dash="dash", line_color="grey", line_width=1, opacity=0.5, row=1, col=1)

    # --- Right panel: per bilateral-class scatter ---
    top8_names = {
        name for name, _ in
        sorted(pair_results.items(), key=lambda x: x[1]["lateralisation_pct"], reverse=True)[:8]
    }

    for fc in list(FUNCTIONAL_CLASSES.keys()) + ["Other"]:
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

    # 50% reference line on scatter
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


def run(get_instance_fn, dataset_name: str):
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


def main():
    from cect.Cook2019HermReader import get_instance

    pair_results, fc_summary, synclass = run(get_instance, "Cook2019Herm")

    if pair_results and "-nogui" not in sys.argv:
        plot_results(pair_results, fc_summary, "Cook2019Herm", synclass)


if __name__ == "__main__":
    main()
