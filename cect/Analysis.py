import sys
import numpy as np

from cect.Cells import get_contralateral_cell
from cect.ConnectomeReader import SYMMETRY_COLORMAP
from cect import print_
import json
import matplotlib.pyplot as plt


all_sym_info = {}


def register_symmetry_info(reader, view, synclass, percentage):
    if reader not in all_sym_info:
        all_sym_info[reader] = {}
    if view not in all_sym_info[reader]:
        all_sym_info[reader][view] = {}
    all_sym_info[reader][view][synclass] = percentage


def save_symmetry_info():
    with open("cect/cache/symmetry_measures.json", "w") as fp:
        json.dump(all_sym_info, fp, indent=4)


def array_info(conn_array):
    nonzero = np.count_nonzero(conn_array)
    print_(
        "- Connection - shape: %s, %i non-zero entries, %i total\n%s\n"
        % (
            conn_array.shape,
            nonzero,
            np.sum(conn_array),
            conn_array,
        )
    )


def convert_to_symmetry_array(cds, synclasses):
    dim = list(cds.connections.values())[0].shape[0]

    new_conn_array = np.zeros([dim, dim], dtype=np.float64)
    array_info(new_conn_array)

    """for synclass in [
        "Chemical",
        "Chemical",
        "Electrical",
        "Contact",
        "Functional",
        "Extrasynaptic",
    ]:"""
    for synclass in synclasses:
        print("   Adding conns of type: %s" % synclass)
        if synclass in cds.connections:
            conns_cs = cds.connections[synclass]

            array_info(conns_cs)
            for i in range(len(conns_cs)):
                for j in range(len(conns_cs[i])):
                    if (
                        i != j
                    ):  #  Kim et al 2024: Self-connections are treated as nonexistent (∀𝐴𝑖𝑖 = 0).
                        new_conn_array[i, j] = (
                            new_conn_array[i, j] or conns_cs[i, j] > 0
                        )

    print_("Symm array:")
    array_info(new_conn_array)

    conn_count = 0
    symm_conn_count = 0

    scaled_conn_array = np.array(new_conn_array)

    verbose = False
    for i in range(len(new_conn_array)):
        pre = cds.nodes[i]
        pre_ = get_contralateral_cell(pre)
        for j in range(len(new_conn_array[i])):
            post = cds.nodes[j]
            post_ = get_contralateral_cell(post)
            w = new_conn_array[i][j]
            if w != 0:
                if verbose:
                    print_(f"Connection {conn_count}:\t{pre}->{post} ({w})")
                assert pre is not post
                if pre_ not in cds.nodes:
                    if verbose:
                        print_(
                            f"   - Mirror conn:\t{pre_}->{post_} not possible as {pre_} missing!\n"
                        )
                    w_ = -1
                elif post_ not in cds.nodes:
                    if verbose:
                        print_(
                            f"   - Mirror conn:\t{pre_}->{post_} not possible as {post_} missing!\n"
                        )
                    w_ = -1
                else:
                    w_ = new_conn_array[cds.nodes.index(pre_), cds.nodes.index(post_)]
                    if verbose:
                        print_(f"   - Mirror:\t{pre_}->{post_} ({w_})\n")

                symm_conn_count += w_
                if w_ <= 0:
                    scaled_conn_array[i][j] = 0.5
                    if w_ == 0:
                        scaled_conn_array[
                            cds.nodes.index(pre_), cds.nodes.index(post_)
                        ] = -1
                conn_count += 1

    percentage = 100 * symm_conn_count / conn_count
    info = f"Of {(len(new_conn_array) ** 2)} possible edges, {conn_count} are connected, {int(symm_conn_count)} are mirrored - {'%.2f' % percentage}% "
    print_(info)

    return scaled_conn_array, info


def test_bilaterals():
    # from cect.White_whole import get_instance
    # from cect.Yim2024DataReader import get_instance

    # from cect.VarshneyDataReader import get_instance

    #
    from cect.White_whole import get_instance
    # from cect.WormNeuroAtlasFuncReader import get_instance

    # from cect.Cook2020DataReader import get_instance
    # from cect.TestDataReader import get_instance

    """
    from cect.RipollSanchezDataReader import get_instance
    from cect.WitvlietDataReader1 import get_instance # 65
    from cect.WitvlietDataReader2 import get_instance # 70.85
    from cect.WitvlietDataReader3 import get_instance # 67.5
    from cect.WitvlietDataReader4 import get_instance # 78.27
    from cect.WitvlietDataReader5 import get_instance # 71.65
    from cect.WitvlietDataReader6 import get_instance # 71.31
    from cect.WitvlietDataReader7 import get_instance # 66.57
    from cect.WitvlietDataReader8 import get_instance # 69.70

    from cect.Cook2019MaleReader import get_instance
    from cect.WitvlietDataReader8 import get_instance
    from cect.SpreadsheetDataReader import get_instance
    from cect.BrittinDataReader import get_instance
    from cect.WormNeuroAtlasMAReader import get_instance
    from cect.RipollSanchezDataReader import get_instance
    from cect.Cook2019HermReader import get_instance
    from cect.ConnectomeView import SOCIAL_VIEW as view"""
    #
    # from cect.ConnectomeView import NEURONS_VIEW as view

    # from cect.ConnectomeView import NONPHARYNGEAL_NEURONS_HM_VIEW as view
    from cect.ConnectomeView import RAW_VIEW as view
    # from cect.ConnectomeView import PHARYNX_VIEW as view

    print(
        "NOTE: For the sake of this paper, we excluded the pharyngeal neurons from the connectome data for both genders due to their distinction from the somatic nervous system."
    )

    cds = get_instance()

    cds2 = cds.get_connectome_view(view)

    synclass = "Chemical Inh"

    synclass = (
        "Chemical"
        if "Raw" not in view.name
        else ("Functional" if "Func" in view.name else "Chemical")
    )

    # print(cds2.connect)

    # synclass = "Chemical"
    # synclass = "Electrical"

    # import plotly.io as pio
    # pio.renderers.default = "browser"

    if "-nogui" not in sys.argv:
        # cds2.connection_number_plot

        print(f"Plotting synclass {synclass} for {view.name}")
        fig, info = cds2.to_plotly_matrix_fig(
            synclass, view, SYMMETRY_COLORMAP, bold_bilaterals=True, symmetry=True
        )

        fig.show()


def plot_symmetry(json_file="cect/cache/symmetry_measures.json", synclass: str = "Chemical", datasets_to_plot=None):
    """
    Plot symmetry for specified synclass across selected datasets.
    Produces 3 subplots for SensorySomaticH, MotorSomaticH, InterneuronsSomaticH.
    
    Parameters:
        json_file (str): Path to the cached JSON file.
        synclass (str): Synapse class to plot, e.g., "Chemical".
        datasets_to_plot (list, optional): List of datasets to include. Defaults to all.
    """
    # Load JSON data
    with open(json_file, "r") as f:
        data = json.load(f)

    views = ["SensorySomaticH", "MotorSomaticH", "InterneuronsSomaticH"]

    if datasets_to_plot is None:
        datasets_to_plot = list(data.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Symmetry Across SomaticH Views ({synclass})", fontsize=16)

    for i, view in enumerate(views):
        ax = axes[i]
        y_vals = []
        labels = []

        for dataset in datasets_to_plot:
            val = data.get(dataset, {}).get(view, {}).get(synclass)
            if val is not None:
                try:
                    val = float(val)
                except ValueError:
                    continue
                y_vals.append(val)
                labels.append(dataset)
        
        ax.plot(labels, y_vals, marker="o", linestyle="-", color="tab:blue")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")

        ax.set_title(view, fontsize=14)
        ax.set_xlabel("Dataset", fontsize=12)
        ax.set_ylabel("Symmetry (%)", fontsize=12)
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    

def main():
    datasets_to_plot = [
        "Witvliet1",
        "Witvliet2",
        "Witvliet3",
        "Witvliet4",
        "Witvliet5",
        "Yim2024",
        "Witvliet6",
        "Witvliet7",
        "Witvliet8",
        "Varshney",
        "White_whole",
        "Cook2019Herm",
        "Cook2019Male"
    ]

    synclass = "Chemical"
    plot_symmetry(synclass=synclass, datasets_to_plot=datasets_to_plot)
    
if __name__ == "__main__":
    main()