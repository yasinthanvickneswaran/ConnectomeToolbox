# Temporary class to allow this to be used in comparison notebook.
# Should be tidied up.

from cect.Cook2019DataReader import Cook2019DataReader

from cect.ConnectomeReader import analyse_connections


def get_instance():
    return Cook2019DataReader()


my_instance = get_instance()

read_data = my_instance.read_data
read_muscle_data = my_instance.read_muscle_data

READER_DESCRIPTION = (
    """Data extracted from **%s** for neuronal connectivity"""
    % my_instance.filename.split("/")[-1]
)


def main1():
    cells, neuron_conns = read_data(include_nonconnected_cells=True)
    neurons2muscles, muscles, muscle_conns = read_muscle_data()
    analyse_connections(cells, neuron_conns, neurons2muscles, muscles, muscle_conns)


if __name__ == "__main__":
    main1()
