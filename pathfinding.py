from __future__ import annotations

from collections import deque
import heapq
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

Node = Tuple[int, int]
Edge = frozenset


def reconstruct_path(parent: Dict[Node, Optional[Node]], goal: Node) -> List[Node]:
    path: List[Node] = []
    current: Optional[Node] = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


def _valid_neighbors(
    graph: nx.Graph,
    node: Node,
    blocked_nodes: Set[Node],
    blocked_edges: Set[Edge],
) -> Iterable[Node]:
    for nbr in graph.neighbors(node):
        if nbr in blocked_nodes:
            continue
        if frozenset((node, nbr)) in blocked_edges:
            continue
        yield nbr


def bfs_path(
    graph: nx.Graph,
    start: Node,
    goal: Node,
    blocked_nodes: Optional[Set[Node]] = None,
    blocked_edges: Optional[Set[Edge]] = None,
) -> List[Node]:
    blocked_nodes = blocked_nodes or set()
    blocked_edges = blocked_edges or set()

    if start in blocked_nodes or goal in blocked_nodes:
        return []

    queue = deque([start])
    parent: Dict[Node, Optional[Node]] = {start: None}

    while queue:
        node = queue.popleft()
        if node == goal:
            return reconstruct_path(parent, goal)

        for nbr in _valid_neighbors(graph, node, blocked_nodes, blocked_edges):
            if nbr in parent:
                continue
            parent[nbr] = node
            queue.append(nbr)

    return []


def dfs_path(
    graph: nx.Graph,
    start: Node,
    goal: Node,
    blocked_nodes: Optional[Set[Node]] = None,
    blocked_edges: Optional[Set[Edge]] = None,
) -> List[Node]:
    blocked_nodes = blocked_nodes or set()
    blocked_edges = blocked_edges or set()

    if start in blocked_nodes or goal in blocked_nodes:
        return []

    stack: List[Node] = [start]
    parent: Dict[Node, Optional[Node]] = {start: None}

    while stack:
        node = stack.pop()
        if node == goal:
            return reconstruct_path(parent, goal)

        for nbr in _valid_neighbors(graph, node, blocked_nodes, blocked_edges):
            if nbr in parent:
                continue
            parent[nbr] = node
            stack.append(nbr)

    return []


def euclidean(a: Node, b: Node) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def manhattan(a: Node, b: Node) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar_path(
    graph: nx.Graph,
    start: Node,
    goal: Node,
    blocked_nodes: Optional[Set[Node]] = None,
    blocked_edges: Optional[Set[Edge]] = None,
    heuristic: str = "manhattan",
) -> List[Node]:
    blocked_nodes = blocked_nodes or set()
    blocked_edges = blocked_edges or set()

    if start in blocked_nodes or goal in blocked_nodes:
        return []

    h = manhattan if heuristic == "manhattan" else euclidean

    g_cost: Dict[Node, float] = {start: 0.0}
    f_cost: Dict[Node, float] = {start: h(start, goal)}
    parent: Dict[Node, Optional[Node]] = {start: None}

    open_heap: List[Tuple[float, Node]] = [(f_cost[start], start)]
    visited: Set[Node] = set()

    while open_heap:
        _, node = heapq.heappop(open_heap)
        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return reconstruct_path(parent, goal)

        for nbr in _valid_neighbors(graph, node, blocked_nodes, blocked_edges):
            tentative_g = g_cost[node] + 1.0
            if tentative_g < g_cost.get(nbr, float("inf")):
                parent[nbr] = node
                g_cost[nbr] = tentative_g
                f_cost[nbr] = tentative_g + h(nbr, goal)
                heapq.heappush(open_heap, (f_cost[nbr], nbr))

    return []
