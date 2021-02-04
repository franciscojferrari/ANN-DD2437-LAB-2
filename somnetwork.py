import numpy as np
from som_utils import load_animals, update_index


class SOMNetwork:
    def __init__(self, data, nr_nodes, lr, neighbors):
        self.data = data
        self.nr_nodes = nr_nodes
        self.neighbors_start = neighbors
        self.neighbors_to_use = neighbors
        self.nr_animals, self.nr_attributes = data.shape
        self.weights = None
        self.lr = lr
        self.init_weights()

        self.current_data_point = None

    def init_weights(self) -> None:
        self.weights = np.random.rand(self.nr_nodes, self.nr_attributes)

    def update_neighbors_to_use(self, current_epoch: int, nr_epochs: int) -> None:
        """Decaying function for the number of neighbors used."""
        self.neighbors_to_use = int(
            self.neighbors_start - self.neighbors_start * current_epoch / nr_epochs
        ) + 1

    def update_weights(self, node_indices: np.array) -> None:
        """Update the weights for the given nodes."""
        for node_index in node_indices:
            self.weights[node_index] += self.lr * (
                self.current_data_point - self.weights[node_index]
            )

    def find_neighbors(self, ind):
        """With the indice of the current node it finds all neighbors and return the indices of
        the neighbors.

        Note: only works if number of neighbors is smaller than total number of nodes.
        """
        assert self.neighbors_to_use < self.nr_nodes
        neighborhood_list = (
            np.array(
                [
                    update_index(i + ind, self.nr_nodes - 1)
                    for i in range(
                        -self.neighbors_to_use // 2, self.neighbors_to_use // 2 + 1
                    )
                ]
            )
        )
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

    def run(self):
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

    def predict(self):
        """Find the closest node for every animal."""
        animal_node_list = []
        for data_point_idx in range(self.nr_animals):
            self.current_data_point = self.data[data_point_idx]
            animal_node_list.append(self.get_closest_node())

        return animal_node_list


if __name__ == "__main__":

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

    result_dict = {animal: idx for animal, idx in zip(animal_names,  idx_list)}
    result_dict = dict(sorted(result_dict.items(), key=lambda item: item[1]))
