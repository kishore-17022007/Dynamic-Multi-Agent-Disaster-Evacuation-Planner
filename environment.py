from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

Node = Tuple[int, int]
Edge = frozenset


class DisasterEnvironment:
    def __init__(
        self,
        rows: int = 12,
        cols: int = 12,
        origin_latlon: Tuple[float, float] = (37.7749, -122.4194),
        cell_step: float = 0.002,
        seed: Optional[int] = None,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.origin_latlon = origin_latlon
        self.cell_step = cell_step
        self.random = random.Random(seed)

        self.graph: nx.Graph = nx.grid_2d_graph(rows, cols)
        for node in self.graph.nodes:
            self.graph.nodes[node]["pos"] = node
            self.graph.nodes[node]["latlon"] = self.node_to_latlon(node)

        self.exits: List[Node] = [
            (0, 0),
            (0, cols - 1),
            (rows - 1, 0),
            (rows - 1, cols - 1),
        ]

        self.fire_nodes: Set[Node] = set()
        self.blocked_nodes: Set[Node] = set()
        self.blocked_edges: Set[Edge] = set()
        self.congestion: Dict[Node, int] = defaultdict(int)

    def node_to_latlon(self, node: Node) -> Tuple[float, float]:
        r, c = node
        lat = self.origin_latlon[0] + r * self.cell_step
        lon = self.origin_latlon[1] + c * self.cell_step
        return lat, lon

    def distance(self, a: Node, b: Node) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def is_traversable(self, node: Node) -> bool:
        return (
            node in self.graph
            and node not in self.fire_nodes
            and node not in self.blocked_nodes
        )

    def is_edge_traversable(self, u: Node, v: Node) -> bool:
        return self.graph.has_edge(u, v) and frozenset((u, v)) not in self.blocked_edges

    def add_fire(self, node: Node) -> bool:
        if node in self.graph and node not in self.exits:
            self.fire_nodes.add(node)
            self.blocked_nodes.discard(node)
            return True
        return False

    def add_blocked_node(self, node: Node) -> bool:
        if node in self.graph and node not in self.exits and node not in self.fire_nodes:
            self.blocked_nodes.add(node)
            return True
        return False

    def add_blocked_road(self, u: Node, v: Node) -> bool:
        if self.graph.has_edge(u, v):
            self.blocked_edges.add(frozenset((u, v)))
            return True
        return False

    def remove_blocked_road(self, u: Node, v: Node) -> None:
        self.blocked_edges.discard(frozenset((u, v)))

    def random_safe_node(self, exclude: Optional[Set[Node]] = None) -> Optional[Node]:
        exclude = exclude or set()
        candidates = [
            n
            for n in self.graph.nodes
            if self.is_traversable(n) and n not in self.exits and n not in exclude
        ]
        if not candidates:
            return None
        return self.random.choice(candidates)

    def update_congestion(self, agent_nodes: Iterable[Node]) -> Dict[Node, int]:
        counter: Dict[Node, int] = defaultdict(int)
        for node in agent_nodes:
            counter[node] += 1
        self.congestion = counter
        return self.congestion

    def congested_nodes(self, threshold: int = 3) -> Set[Node]:
        return {node for node, count in self.congestion.items() if count >= threshold}

    def probability_update(
        self,
        p_fire_spread: float = 0.15,
        p_new_block_node: float = 0.06,
        p_new_block_edge: float = 0.08,
    ) -> Dict[str, List[Node]]:
        updates = {"new_fire": [], "new_blocked_nodes": [], "new_blocked_roads": []}

        new_fire: Set[Node] = set()
        for fire_node in list(self.fire_nodes):
            for nbr in self.graph.neighbors(fire_node):
                if (
                    nbr in self.exits
                    or nbr in self.fire_nodes
                    or nbr in self.blocked_nodes
                ):
                    continue
                if self.random.random() < p_fire_spread:
                    new_fire.add(nbr)

        for node in new_fire:
            self.fire_nodes.add(node)
            self.blocked_nodes.discard(node)
            updates["new_fire"].append(node)

        if self.random.random() < p_new_block_node:
            candidate = self.random_safe_node()
            if candidate is not None:
                self.blocked_nodes.add(candidate)
                updates["new_blocked_nodes"].append(candidate)

        if self.random.random() < p_new_block_edge:
            candidate_edges = [
                (u, v)
                for u, v in self.graph.edges
                if frozenset((u, v)) not in self.blocked_edges
                and self.is_traversable(u)
                and self.is_traversable(v)
            ]
            if candidate_edges:
                u, v = self.random.choice(candidate_edges)
                self.blocked_edges.add(frozenset((u, v)))
                updates["new_blocked_roads"].append((u, v))

        return updates

    def reset_disaster(self) -> None:
        self.fire_nodes.clear()
        self.blocked_nodes.clear()
        self.blocked_edges.clear()
        self.congestion.clear()
