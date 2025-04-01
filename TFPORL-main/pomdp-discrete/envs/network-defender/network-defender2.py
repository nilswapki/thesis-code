import gym
from gym import spaces
import numpy as np
import networkx as nx
import random
import math
from collections import deque
import matplotlib.pyplot as plt


class NetworkDefenderEnv(gym.Env):
    """
    A simplified, sensor-based network defender environment.

    Environment details:
    - The environment consists of a flexible number of nodes with a random, connected topology.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 n_nodes=15,
                 extra_edge_prob=0.2,
                 episode_length=100,
                 noise_mean=0.2,
                 noise_95_interval=0.15,
                 seed=None,
                 recursive=False):
        super(NetworkDefenderEnv, self).__init__()

        self.name = 'network-defender'
        self.recursive = recursive

        self.n_nodes = n_nodes
        self.extra_edge_prob = extra_edge_prob
        self.episode_length = episode_length

        self.noise_mean = noise_mean
        self.noise_std = noise_95_interval / 1.96

        if seed is not None:
            self.seed(seed)

        # Generate network graph and derive connection matrix and neighbor lists.
        self.graph = self._generate_connected_graph(self.n_nodes, self.extra_edge_prob)
        self.connection_matrix, self.neighbors = self._create_adj_and_neighbors(self.graph)


        ### Global node statuses:
        # infiltration: flag for each node, >=1 if infiltrated (by attacker), 0 otherwise.
        self.infiltrated = np.zeros(self.n_nodes, dtype=np.int32)
        self.passively_infiltrated = []
        self.last_infiltrated = None
        # Sensor readings
        self.sensor_reading = np.zeros(self.n_nodes, dtype=np.float32)


        # Define observation space:
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_nodes,), dtype=np.float32)

        # Define action space:
        # Fixed-size discrete action space.
        # Actions: Do nothing or Restore.
        self.action_space = spaces.Discrete(self.n_nodes+1)

        self.timestep = 0
        self.summary_reward = 0
        self.done = False

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)

    def _generate_connected_graph(self, n_nodes, extra_edge_prob):
        # Use NetworkX to generate a spanning tree, then add extra edges.
        tree = nx.from_prufer_sequence(np.random.randint(0, n_nodes, size=n_nodes - 2))
        G = nx.Graph(tree)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if not G.has_edge(i, j) and np.random.rand() < extra_edge_prob:
                    G.add_edge(i, j)
        # Ensure connectivity.
        if not nx.is_connected(G):
            G = nx.connected_component_subgraphs(G).__next__()
        return G

    def _create_adj_and_neighbors(self, G):
        # Create adjacency matrix and neighbor list.
        adj = nx.adjacency_matrix(G).todense()
        adj = np.array(adj, dtype=np.int32)
        neighbors = []
        for i in range(self.n_nodes):
            neigh = np.where(adj[i] == 1)[0].tolist()
            neighbors.append(neigh)
        return adj, neighbors

    def is_freestanding(self, node):
        """
        Check if a node is freestanding, meaning that none of its neighbors
        (except possibly the starting infiltrated node) are infiltrated.
        """
        # Here, a node is considered freestanding if all its neighbors
        # are not infiltrated.
        for neighbor in self.neighbors[node]:
            if self.infiltrated[neighbor] > 0:
                return False
        return True

    def dfs_longest_path(self, current, visited):
        """
        Recursively search for the longest path starting from 'current'
        that only visits nodes that are not infiltrated and are freestanding.

        Args:
            graph (nx.Graph): The input graph.
            current (node): The current node.
            visited (set): Nodes already visited in this path.
            infiltrated (dict): Mapping from node to infiltration value.

        Returns:
            list: The longest valid path (as a list of nodes) starting at 'current'.
        """
        best_path = [current]
        for neighbor in self.neighbors[current]:
            if neighbor in visited:
                continue
            # We only allow neighbors that are not infiltrated...
            if self.infiltrated[neighbor] > 0:
                continue
            # ...and that are freestanding (none of their neighbors are infiltrated).
            if not self.is_freestanding(neighbor):
                continue

            # Explore deeper from this neighbor.
            new_visited = visited | {neighbor}
            candidate_path = self.dfs_longest_path(neighbor, new_visited)
            if len(candidate_path) + 1 > len(best_path):
                best_path = [current] + candidate_path
        return best_path

    def find_longest_freestanding_path(self):
        """
        Find the longest path in 'graph' starting from an infiltrated node and
        going only over non-infiltrated nodes that are freestanding (none of their
        neighbors are infiltrated).

        Args:
            graph (nx.Graph): The graph.
            infiltrated (dict): Mapping from node to infiltration status/number.

        Returns:
            list: The longest path found, as a list of nodes.
        """
        best_overall = []
        # Iterate over every node that is already infiltrated.
        for node in self.graph.nodes():
            if self.infiltrated[node] > 0:
                # For each neighbor of the infiltrated node, if it's not infiltrated
                # and is freestanding, try to build a path from it.
                for neighbor in self.neighbors[node]:
                    if self.infiltrated[neighbor] == 0 and self.is_freestanding(neighbor):
                        # Use a visited set that initially includes the starting infiltrated node
                        # (to avoid stepping back to it) and the neighbor.
                        path = self.dfs_longest_path(self, neighbor, {node, neighbor})
                        if len(path) > len(best_overall):
                            best_overall = path
        return best_overall


    def _update_sensor_reading(self):
        """
        Returns a sensor reading for each node.
        The suspicion level increases by 0.1 per timestep if infiltrated
        A noise term is added to the reading
        """
        self.sensor_reading = self.infiltrated / 10.0
        self.sensor_reading[self.initial_attacker_node] = 0  # the initial foothold does not raise any suspicion
        self.sensor_reading += np.random.normal(self.noise_mean, self.noise_std, size=self.n_nodes)
        self.sensor_reading = np.clip(self.sensor_reading, 0.0, 1.0)

        return self.sensor_reading

    def _get_obs(self):
        return self.sensor_reading.copy().reshape(1, -1)

    def reset(self, seed=None, return_info=False, options=None):
        self.timestep = 0
        self.done = False

        # Reset node statuses.
        self.infiltrated = np.zeros(self.n_nodes, dtype=np.int32)
        self.sensor_reading = np.zeros(self.n_nodes, dtype=np.float32)

        # Place attacker at a random node.
        self.initial_attacker_node = np.random.randint(0, self.n_nodes)

        self.infiltrated[self.initial_attacker_node] = 1
        self.passively_infiltrated = []
        self.last_infiltrated = self.initial_attacker_node

        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done. Reset before stepping.")

        # increase the value of every infiltrated node by 1 to indicate the time passed
        for node in range(self.n_nodes):
            if self.infiltrated[node] > 0:
                self.infiltrated[node] += 1

        reward = 0
        self.timestep += 1
        restore_necessary = 0

        # -------- Defender --------
        # 0:n_nodes-1: Restore node
        # n_nodes: Do nothing
        if action != self.n_nodes:
            # If node to be restored is actually infiltrated, restore it
            if (action != self.initial_attacker_node) and (self.infiltrated[action] > 0):  # cannot restore the initial foothold of the attacker
                reward += max((2 - (0.2 * self.infiltrated[action])), 0)
                self.infiltrated[action] = 0
            # If node to be restored is passively infiltrated, restore it
            elif self.recursive and (action != self.initial_attacker_node) and (action in self.passively_infiltrated):
                self.infiltrated[action] = 0
                self.passively_infiltrated.remove(action)
                reward += 0.5
            # If node to be restored is not actively or passively infiltrated, penalize.
            else:
                reward -= 1
                restore_necessary = 1

            if self.recursive:
                # check for every infiltrated node if there still is a path to the initial foothold, if not, restore it
                for node in range(self.n_nodes):
                    if self.infiltrated[node] > 0:
                        # if infiltrated node is cut off from initial foothold, restore it
                        if not nx.has_path(self.graph.subgraph(np.where(self.infiltrated > 0)[0]), node, self.initial_attacker_node):
                            reward += (2 / self.infiltrated[node])
                            self.infiltrated[node] = 0  # reset the node
                            # add the node to a list of passively infiltrated nodes (attacker still has knowledge)
                            self.passively_infiltrated.append(node)

        # -------- Attacker --------
        # on average, the attacker infiltrates a new node every two timesteps
        # draw from poisson distribution to allow the attacker to potentially infiltrate several times in one timestep
        for i in range(np.random.poisson(0.5)):

            # obtain all nodes adjacent to infiltrated nodes
            adj_nodes = []
            for node in range(self.n_nodes):
                if self.infiltrated[node] > 0:
                    adj_nodes.extend(self.neighbors[node])
            # check for duplicates and remove already infiltrated nodes
            adj_nodes = list(set(adj_nodes) - set(np.where(self.infiltrated > 0)[0]))

            if len(adj_nodes) > 0:
                # if there is an adjacent node that would connect the attacker to a passively infiltrated node, pick it
                # if no such node exists, pick a random node from adj_nodes
                picked = False
                if self.recursive:
                    for node in adj_nodes:
                        if any(n in self.passively_infiltrated for n in self.neighbors[node]):
                            self.last_infiltrated = node
                            self.infiltrated[self.last_infiltrated] = 1
                            picked = True
                            break

                # if no node was picked
                if (not self.recursive) or (not picked):
                    # if a neighbor to the previously infiltrated node is available, choose that one
                    if self.neighbors[self.last_infiltrated] in adj_nodes:
                        self.last_infiltrated = random.choice(self.neighbors[self.last_infiltrated])
                        self.infiltrated[self.last_infiltrated] = 1
                    # pick random node from available nodes
                    else:
                        self.last_infiltrated = random.choice(adj_nodes)
                        self.infiltrated[self.last_infiltrated] = 1

            if self.recursive:
                # for every passively infiltrated node, check if it can be reached by the attacker
                # via the infiltrated nodes and passively infiltrated nodes
                for node in self.passively_infiltrated:
                    if nx.has_path(self.graph.subgraph(np.where((self.infiltrated > 0) | (np.isin(np.arange(self.n_nodes), self.passively_infiltrated)))[0]), node, self.initial_attacker_node):
                        self.passively_infiltrated.remove(node)
                        self.infiltrated[node] = 10

        # reward is the negative sum of all infiltrated nodes --> confuses the agent
        reward -= (np.sum(self.infiltrated > 0) - 1) * 0.05  # dont give negative reward for initial foothold


        # End episode if maximum timesteps reached.
        if self.timestep >= self.episode_length:
            self.done = True

        self._update_sensor_reading()
        obs = self._get_obs()
        info = {"initial_attacker_node": self.initial_attacker_node, "infiltrated_nodes": self.infiltrated,
                "restored": restore_necessary}
        return obs, reward, self.done, info


    def eval(self):
        # Add evaluation-specific logic here if needed
        pass

    def render(self, mode='human'):
        print(f"Time: {self.timestep}")
        print(f"Infiltrated Nodes: {self.infiltrated}")
        print(f"Sensor Reading: {self.sensor_reading}")
        print("")

    def close(self):
        pass


# Example usage:
if __name__ == "__main__":
    env = NetworkDefenderEnv(n_nodes=15, extra_edge_prob=0.2, episode_length=100, seed=36,
                             noise_mean=0.2, noise_95_interval=0.2, recursive=True)

    #nx.draw(env.graph, with_labels=True, node_color='lightblue', font_weight='bold', node_size=500, font_size=12)
    #plt.savefig('graph')

    obs, _ = env.reset()
    env.render()
    total_reward = 0
    done = False
    while not done:
        # For demonstration, choose random actions.
        action = np.random.randint(0, env.n_nodes+1)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Action: {action}, Reward: {reward}")
        env.render()
    print("Total Reward:", total_reward)
