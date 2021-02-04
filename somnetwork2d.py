import numpy as np
import math
from typing import Tuple, List, Any
from som_utils import get_neighbor_coords, check_coord, load_mp_data


class SOMNetwork2D:
    """SOM Network with two-dimensional square grid."""

    def __init__(self, data, nr_nodes, lr, neighbors):
        # Parameters.
        self.data = data
        self.nr_nodes = nr_nodes
        self.neighbors_start = neighbors
        self.neighbors_to_use = neighbors
        self.nr_datapoints, self.nr_attributes = data.shape
        self.weights = None
        self.weights_shape = (nr_nodes, nr_nodes, self.nr_attributes)
        self.lr = lr

        # Variable used while training.
        self.current_data_point = None
        self.current_neighbors = set()

        # Set up weights.
        self.init_weights()

    def init_weights(self) -> None:
        """Initialise the weights of the SOM Network."""
        self.weights = np.random.rand(*self.weights_shape)

    def update_neighbors_to_use(self, current_epoch: int, nr_epochs: int) -> None:
        """Decaying function for the number of neighbors used."""
        self.neighbors_to_use = int(
            self.neighbors_start - self.neighbors_start * current_epoch / nr_epochs
        )

    def update_weights(self) -> None:
        """Update the weights for the given nodes."""
        for node_index in self.current_neighbors:
            x, y = node_index[0], node_index[1]
            self.weights[x][y] += self.lr * (
                self.current_data_point - self.weights[x][y]
            )

    def find_direct_neighbors(self, coords) -> List[Tuple[int, int]]:
        """Returns the direct neighbors of a node"""
        x, y = coords
        neighbor_coords = get_neighbor_coords(x, y)

        return [
            coord for coord in neighbor_coords if check_coord(coord, self.nr_nodes - 1)
        ]

    def find_neighbors(self, coords: Tuple[int, int], depth: int):
        """With the index of the current node it finds all neighbors and return the indices of
        the neighbors.
        """
        direct_neighbors = self.find_direct_neighbors(coords)

        if depth < 2:  # TODO: check if this is the correct base-case
            self.current_neighbors.update(set(direct_neighbors))
            return direct_neighbors
        else:
            self.current_neighbors.update(set(direct_neighbors))
            for coords in direct_neighbors:
                self.find_neighbors(coords, depth - 1)

    def calc_distance(self, x):
        """Calculates distance between two vectors."""
        return np.linalg.norm(x - self.current_data_point)

    def get_closest_node_column(self, col) -> Tuple[Any, Any]:
        """Returns row id for smallest value in column."""
        distance_matrix = np.apply_along_axis(
            self.calc_distance, axis=1, arr=self.weights[col]
        )
        return np.argmin(distance_matrix), min(distance_matrix)

    def get_closest_node(self) -> Tuple[Any, Any]:
        """Calculate the distances to all nodes and return node index of closest node."""
        smallest_value = math.inf
        row_idx, col_idx = None, None
        for col in range(10):
            row_idx_t, value = self.get_closest_node_column(col)
            if value < smallest_value:
                row_idx, col_idx = row_idx_t, col
                smallest_value = value
        return row_idx, col_idx

    def run(self) -> None:
        """Run the algorithm."""
        # Iterate over the data points in data set (animals).
        for data_point_idx in range(self.nr_datapoints):
            # Get current data point.
            self.current_data_point = self.data[data_point_idx]

            # Find closest node for given data point.
            row_idx, col_idx = self.get_closest_node()

            # Find list of neighbors around the closest node.
            self.find_neighbors((row_idx, col_idx), self.neighbors_to_use)
            self.current_neighbors.update([(row_idx, col_idx)])

            # Update the weights.
            self.update_weights()

            # Empty the set.
            self.current_neighbors = set()


if __name__ == "__main__":
    """4.3 Data Clustering: Votes of MP's"""
    # Parameters.
    number_nodes = 10  # Square grid.
    learning_rate = 0.2
    neighbors_start = 4
    epochs = 20

    # Load data.
    data = load_mp_data()

    # Initiate and train model.
    model = SOMNetwork2D(data, number_nodes, learning_rate, neighbors_start)
    for epoch in range(epochs):
        model.run()
        model.update_neighbors_to_use(epoch, epochs)

    # Now print the result for all the 349 members of parliament.
