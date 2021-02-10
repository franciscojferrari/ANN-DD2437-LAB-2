import numpy as np
import math
from typing import Tuple, List, Any
from som_utils import (
    load_mp_data,
    plot_2d_grid,
    plot_winner_nodes,
    plot_occupancy,
    plot_k_best_points,
)


class SOMNetwork2D:
    """SOM Network with two-dimensional square grid."""

    def __init__(self, data, nr_nodes, lr, neighbors):
        # Parameters.
        self.data = data
        self.nr_nodes = nr_nodes
        self.neighbors_start = neighbors
        self.neighbors_to_use = neighbors
        self.nr_datapoints, self.nr_attributes = data.shape
        print(f"Number  of data points: {self.nr_datapoints},  Number of attributes: {self.nr_attributes}")
        self.weights = None
        self.weights_shape = (nr_nodes, nr_nodes, self.nr_attributes)
        self.lr = lr

        # Variable used while training.
        self.current_data_point = None
        self.current_neighbors = []

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
            x, y = node_index
            self.weights[x][y] += self.lr * (
                self.current_data_point - self.weights[x][y]
            )

    def find_neighbors(self, coords: Tuple[int, int]):
        """With the index of the current node it finds all neighbors and return the indices of
        the neighbors.
        """
        neighbors_coords = []
        x,  y = coords
        for xx in range(10):
            for yy in range(10):
                if abs(x - xx) + abs(y-yy) <= self.neighbors_to_use:
                    neighbors_coords.append([xx, yy])
        self.current_neighbors = neighbors_coords

    def calc_distance(self, x):
        """Calculates distance between two vectors."""
        return np.linalg.norm(x - self.current_data_point)

    def get_closest_node(self) -> Tuple[Any, Any]:
        """Calculate the distances to all nodes and return node index of closest node."""
        smallest_value = math.inf
        row_idx, col_idx = None, None

        for row in range(10):
            for col in range(10):
                value = self.calc_distance(self.weights[row][col])
                if value < smallest_value:
                    row_idx, col_idx = row, col
                    smallest_value = value
        return row_idx, col_idx

    def get_set_of_closest_nodes(self) -> np.array:
        """Calculate the distances to all nodes and return nodes with distances."""
        distance_matrix = np.zeros((10, 10))
        for row in range(10):
            for col in range(10):
                distance_matrix[row][col] = self.calc_distance(self.weights[row][col])
        return distance_matrix

    def calc_distances_all_data(self) -> np.array:
        distances = []
        for data_point_idx in range(self.nr_datapoints):
            self.current_data_point = self.data[data_point_idx]
            distances.append(self.get_set_of_closest_nodes())

        return np.array(distances)

    def run(self) -> None:
        """Run the algorithm."""
        # Iterate over the data points in data set.
        for data_point_idx in range(self.nr_datapoints):
            # Get current data point.
            self.current_data_point = self.data[data_point_idx]

            # Find closest node for given data point.
            row_idx, col_idx = self.get_closest_node()

            # Find list of neighbors around the closest node.
            self.find_neighbors((row_idx, col_idx))

            # Update the weights.
            self.update_weights()

            # Empty the set.
            self.current_neighbors.clear()

    def predict(self) -> List:
        """Find closest node for each data point in data set."""
        winner_nodes = []
        for data_point_idx in range(self.nr_datapoints):
            self.current_data_point = self.data[data_point_idx]
            winner_nodes.append(self.get_closest_node())

        return winner_nodes

    def get_k_best_points(self, k: int = 5) -> List:
        k_best_points = []
        m = self.calc_distances_all_data()
        for row in range(10):
            for col in range(10):
                idx = np.argpartition(m[:, row, col], k)[:k]
                k_best_points.append(idx)

        return k_best_points


if __name__ == "__main__":
    """4.3 Data Clustering: Votes of MP's"""
    # Parameters.
    number_nodes = 10  # Square grid.
    learning_rate = 0.2
    neighbors_start = 4
    epochs = 10

    # Load data.
    data, mp_district, mp_party, mp_sex = load_mp_data()

    # Initiate and train model.
    model = SOMNetwork2D(data, number_nodes, learning_rate, neighbors_start)
    for epoch in range(epochs):
        model.run()
        model.update_neighbors_to_use(epoch, epochs)
        if epoch % 5 == 0:
            print(f"Epoch number: {epoch}")

    # Now print the result for all the 349 members of parliament.
    winner_nodes = model.predict()

    # plot_2d_grid(title="Occupancy of nodes.")
    # plot_occupancy(winner_nodes)

    plot_2d_grid(title="Distribution of sex.")
    plot_winner_nodes(winner_nodes, color_codes=mp_sex)

    # plot_2d_grid(title="Distribution of districts.")
    # plot_winner_nodes(winner_nodes, color_codes=mp_district)

    plot_2d_grid(title="Distribution of parties.")
    plot_winner_nodes(winner_nodes, color_codes=mp_party)

    for k in [5]:
        k_best_points = model.get_k_best_points(k=k)
        plot_2d_grid(title=f"Plotting {k}-best points  - sex")
        plot_k_best_points(k_best_points, color_codes=mp_sex)

        # plot_2d_grid(title=f"Plotting {k}-best points  - district")
        # plot_k_best_points(k_best_points, color_codes=mp_district)

        plot_2d_grid(title=f"Plotting {k}-best points  - party")
        plot_k_best_points(k_best_points, color_codes=mp_party)
