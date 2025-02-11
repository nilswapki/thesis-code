import gym
from gym import spaces
import numpy as np
import networkx as nx  # For generating connected graphs
import random


class NetworkDefenderEnv(gym.Env):
    """
    A simplified environment representing an internal airplane network.

    - The network consists of a flexible number of nodes with a random, connected topology.
    - An attacker infiltrates one node and moves every timestep.
    - The defender observes each node's isolation status and (if analyzed) the infiltration time.
    - Defender actions:
        0: No-op.
        1: Analyze a node (reveals with certainty whether it is compromised and, if so, the current timestep).
        2: Isolate a node (removes it from the network, stopping traffic; if it is the attacker’s node, the episode ends).
    - Reward:
        Each timestep, the defender gets –1 if the attacker occupies a non-critical node or –10 if the node is critical.
        If the defender isolates the attacker’s node, a positive reward is given and the episode ends.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, n_nodes=100, extra_edge_prob=0.1, episode_length=100, num_critical_nodes=3, critical_nodes=None, graph=None, seed=None):
        super(NetworkDefenderEnv, self).__init__()

        self.name = 'network-defender'

        self.n_nodes = n_nodes
        self.extra_edge_prob = extra_edge_prob
        self.episode_length = episode_length

        # Set the seed for reproducibility
        if seed is not None:
            self.seed(seed)

        # Generate a connected graph (adjacency matrix) and neighbor list.
        if graph is None:
            self.graph = self._generate_connected_graph(self.n_nodes, self.extra_edge_prob)
        else:
            self.graph = graph

        self.connection_matrix, self.neighbors = self._create_adj_and_neighbors(self.graph)

        # Define critical nodes (if not provided, default to nodes that are as far apart as possible)
        if critical_nodes is None:
            self.critical_nodes = self._select_distant_nodes(self.graph, num_critical_nodes)
        else:
            assert all(0 <= node < self.n_nodes for node in
                       critical_nodes), "Critical nodes must be within the range of n_nodes"
            self.critical_nodes = list(critical_nodes)

        #print("Critical Nodes: ", self.critical_nodes)

        # analysis (-1,episode_length), time (0,episode_length), critical_node_infiltrated (0,1)
        self.observation_space = spaces.Box(low=-1, high=self.episode_length, shape=(3,), dtype=np.int32)

        # Action space: A tuple (action_type, node_index)
        self.action_space = spaces.Discrete(2*self.n_nodes)  # 1:n_nodes for analyzing, n_nodes+1:2*n_nodes for restoring

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)


    def _generate_connected_graph(self, n_nodes, extra_edge_prob):
        """
        Generates a random connected graph.
        Returns an adjacency matrix (numpy array) of shape (n_nodes, n_nodes)
        and a neighbor list (list of lists).
        """
        # Use NetworkX to generate a random spanning tree (which is connected)
        tree = nx.from_prufer_sequence(np.random.randint(0, n_nodes, size=n_nodes-2))
        G = nx.Graph(tree)

        # Add extra edges randomly with probability extra_edge_prob
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if not G.has_edge(i, j) and np.random.rand() < extra_edge_prob:
                    G.add_edge(i, j)

        return G


    def _create_adj_and_neighbors(self, G):
        # Convert to adjacency matrix
        adj = nx.adjacency_matrix(G).todense()
        adj = np.array(adj, dtype=np.int32)

        # Create neighbor list
        neighbors = []
        for i in G.nodes:
            neigh = np.where(adj[i] == 1)[0].tolist()
            neighbors.append(neigh)

        return adj, neighbors

    def _select_distant_nodes(self, G, num_nodes):
        """
        Selects `num_nodes` nodes that are as far apart as possible in the graph `G`.
        """
        import networkx as nx
        from itertools import combinations

        # Compute all-pairs shortest path lengths
        lengths = dict(nx.all_pairs_shortest_path_length(G))

        # Find the pair of nodes with the maximum shortest path length
        max_dist = 0
        best_pair = None
        for u, v in combinations(G.nodes, 2):
            if lengths[u][v] > max_dist:
                max_dist = lengths[u][v]
                best_pair = (u, v)

        # Start with the best pair
        selected_nodes = list(best_pair)

        # Greedily add nodes that maximize the minimum distance to the selected nodes
        while len(selected_nodes) < num_nodes:
            best_node = None
            best_min_dist = 0
            for node in G.nodes:
                if node not in selected_nodes:
                    min_dist = min(lengths[node][other] for other in selected_nodes)
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_node = node
            selected_nodes.append(best_node)

        return selected_nodes

    def reset(self, seed=None, return_info=False, options=None):
        # Reset time and reward
        self.timestep = 0
        self.reward = 0

        self.analysis = 0

        # Randomly choose an attacker starting node (ensure it's not restored, which it won't be at reset)
        self.attacker_node = np.random.randint(0, self.n_nodes)

        # For breadcrumb trail: record the time when the attacker first infiltrated the current node.
        self.attacker_infiltration_time = np.zeros(self.n_nodes, dtype=np.int32)

        self.current_critical_node_index = -1
        np.random.shuffle(self.critical_nodes)

        # Episode not done
        self.done = False

        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.analysis,
                         1 if self.attacker_node in self.critical_nodes else 0,
                         0], dtype=np.int32).reshape(1, 3)

    def step(self, action):

        self.reward = 0

        # Increment timestep
        self.timestep += 1

        if self.done:
            raise RuntimeError("Episode is done. Please reset the environment.")

        action_type = action % 2
        node = action % self.n_nodes

        # Process defender action first:
        if action_type == 0:  # Analyze node
            # When analyzing a node, update analysis info:
            if node == self.attacker_node:
                # If the node is currently compromised, record the current timestep.
                self.analysis = self.attacker_infiltration_time[node]
            else:
                # If not compromised, mark as analyzed clean (0).
                self.analysis = 0

        elif action_type == 1:  # restore node
            if not self.timestep == 1:  # prevent episode from ending on first timestep (error in learner logic)

                if node == self.attacker_node:
                    # Successful restoration ends the episode.
                    self.reward += 100  # A positive reward for stopping the attack.
                    self.done = True
                else:
                    # Restoration has a penalty.
                    self.reward -= 10

        # Compute penalty for the attacker's current infiltration:
        # Negative reward for every timestep that the attacker occupies a node.
        # Larger penalty if the node is critical.
        if self.attacker_node in self.critical_nodes:
            self.reward -= 5
        else:
            self.reward -= 1

        # Attacker movement strategy
        current_neighbors = self.neighbors[self.attacker_node]

        if current_neighbors:
            # If the attacker is at a critical node, move to the next critical node.
            if self.attacker_node in self.critical_nodes:
                #print("Attacker on Node ", self.attacker_node)
                self.current_critical_node_index = (self.current_critical_node_index + 1) % len(self.critical_nodes)

            # Choose the move that minimizes the distance to the next critical node.
            best_move = min(current_neighbors, key=lambda n: nx.shortest_path_length(
                self.graph, n, self.critical_nodes[self.current_critical_node_index]))

            # Move attacker
            self.attacker_node = best_move

            # Record infiltration time
            self.attacker_infiltration_time[self.attacker_node] = self.timestep

        # End episode if maximum steps reached.
        if self.timestep >= self.episode_length:
            self.done = True

        obs = self._get_obs()
        info = {"attacker_node": self.attacker_node}
        return obs, self.reward, self.done, info

    def render(self, mode='human'):
        print("Time:", self.timestep)
        print("Critical Node Infiltrated:", self.attacker_node if self.attacker_node in self.critical_nodes else -1)
        print("Analysis info:", self.analysis)
        print("Attacker is at node:", self.attacker_node)
        print("")

    def close(self):
        pass

    def eval(self):
        # Add evaluation-specific logic here if needed
        pass


# Example usage:
if __name__ == "__main__":
    env = NetworkDefenderEnv(n_nodes=300, extra_edge_prob=0.01, episode_length=100, seed=42)
    cumulated_reward = 0
    obs, _ = env.reset()
    #env.render()
    done = False
    while not done:
        # For illustration, we choose random actions.
        action = (np.random.randint(0, 2*env.n_nodes))
        obs, reward, done, info = env.step(action)
        cumulated_reward += reward
        print(f"Action: {'Analyzing' if action % 2 == 0 else 'Restoring'} Node {action % env.n_nodes}, Reward: {reward}")
        #env.render()
    print("Cumulated Reward:", cumulated_reward)
