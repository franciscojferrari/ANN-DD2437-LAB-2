import numpy as np
from typing import List
from som_utils import (
    load_animals,
    update_index,
    load_cities,
    plot_city_data,
    plot_weight_data,
    plot_animals,
)


class SOMNetwork:
    """SOM Network with one-dimensional grid."""

    def __init__(self, data, nr_nodes, lr, neighbors):
        self.data = data
        self.nr_nodes = nr_nodes
        self.neighbors_start = neighbors
        self.neighbors_to_use = neighbors
        self.nr_animals, self.nr_attributes = data.shape
        self.weights = None
        self.lr = lr
        self.current_data_point = None

        self.init_weights()

    def init_weights(self) -> None:
        """Initialise the weights of the SOM Network."""
        self.weights = np.random.rand(self.nr_nodes, self.nr_attributes)

    def update_neighbors_to_use(self, current_epoch: int, nr_epochs: int) -> None:
        """Decaying function for the number of neighbors used."""
        self.neighbors_to_use = int(
            self.neighbors_start - self.neighbors_start * current_epoch / nr_epochs
        )

    def update_weights(self, node_indices: np.array) -> None:
        """Update the weights for the given nodes."""
        for node_index in node_indices:
            self.weights[node_index] += self.lr * (
                self.current_data_point - self.weights[node_index]
            )

    def find_neighbors(self, ind):
        """With the index of the current node it finds all neighbors and return the indices of
        the neighbors.

        Note: only works if number of neighbors is smaller than total number of nodes.
        """
        assert self.neighbors_to_use < self.nr_nodes
        neighborhood_list = np.array(
            [
                update_index(i + ind, self.nr_nodes - 1)
                for i in range(
                    -self.neighbors_to_use // 2, self.neighbors_to_use // 2 + 1
                )
            ]
        )
        if self.neighbors_to_use == 0:
            return [ind]
        return neighborhood_list

    def calc_distance(self, x):
        """Calculates distance between two vectors."""
        return np.linalg.norm(x - self.current_data_point)

    def get_closest_node(self) -> int:
        """Calculate the distances to all nodes and return node index of closest node."""
        distance_matrix = np.apply_along_axis(
            self.calc_distance, axis=1, arr=self.weights
        )
        return np.argmin(distance_matrix)

    def run(self) -> None:
        """Run the algorithm."""
        # Iterate over the data points in data set (animals).
        for data_point_idx in range(self.nr_animals):
            # Get current data point.
            self.current_data_point = self.data[data_point_idx]

            # Find closest node for given data point.
            ind = self.get_closest_node()

            # Find list of neighbors around the closest node.
            neighborhood_list = self.find_neighbors(ind)

            # Update the weights.
            self.update_weights(neighborhood_list)

    def predict(self) -> List[int]:
        """Find the closest node for every animal.

        Returns a list containing the closest node for each datapoint (animal).
        """
        animal_node_list = []
        for data_point_idx in range(self.nr_animals):
            self.current_data_point = self.data[data_point_idx]
            animal_node_list.append(self.get_closest_node())

        return animal_node_list


if __name__ == "__main__":

    """4.1 Topological Ordering of Animal Species"""
    # Parameters.
    number_nodes = 100
    learning_rate = 0.2
    neighbors_start = 25
    epochs = 20

    data, animal_names = load_animals()

    # Initiate model and train model.
    model = SOMNetwork(data, number_nodes, learning_rate, neighbors_start)
    for i in range(epochs):
        model.run()
        model.update_neighbors_to_use(i, epochs)

    # Check the results.
    idx_list = model.predict()
    result_dict = {animal: idx for animal, idx in zip(animal_names, idx_list)}
    result_dict = dict(sorted(result_dict.items(), key=lambda item: item[1]))
    print(result_dict)
    plot_animals(result_dict)

    """4.2 Cyclic Tour"""
    # Parameters.
    number_nodes = 10
    learning_rate = 0.25
    neighbors_start = 3
    epochs = 40

    city_data = load_cities()
    model = SOMNetwork(city_data, number_nodes, learning_rate, neighbors_start)
    for epoch in range(epochs):
        model.run()
        model.update_neighbors_to_use(epoch, epochs)

    # Plot the cities and the nodes with it's  connections.
    plot_city_data(city_data)
    plot_weight_data(model.weights)
