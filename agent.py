from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from pathfinding import astar_path, bfs_path, dfs_path

Node = Tuple[int, int]
Edge = frozenset


@dataclass
class EvacuationAgent:
    agent_id: int
    current_node: Node
    goal_node: Optional[Node] = None
    planned_path: List[Node] = field(default_factory=list)
    status: str = "moving"
    known_fire_nodes: Set[Node] = field(default_factory=set)
    known_blocked_nodes: Set[Node] = field(default_factory=set)
    known_blocked_edges: Set[Edge] = field(default_factory=set)

    def perceive(self, environment, radius: int = 2) -> None:
        for node in environment.graph.nodes:
            if abs(node[0] - self.current_node[0]) + abs(node[1] - self.current_node[1]) <= radius:
                if node in environment.fire_nodes:
                    self.known_fire_nodes.add(node)
                if node in environment.blocked_nodes:
                    self.known_blocked_nodes.add(node)

        for u, v in environment.graph.edges:
            if (
                abs(u[0] - self.current_node[0]) + abs(u[1] - self.current_node[1]) <= radius
                or abs(v[0] - self.current_node[0]) + abs(v[1] - self.current_node[1]) <= radius
            ):
                edge_key = frozenset((u, v))
                if edge_key in environment.blocked_edges:
                    self.known_blocked_edges.add(edge_key)

    def communicate(self, others: List["EvacuationAgent"]) -> None:
        for other in others:
            if other.agent_id == self.agent_id:
                continue
            self.known_fire_nodes |= other.known_fire_nodes
            self.known_blocked_nodes |= other.known_blocked_nodes
            self.known_blocked_edges |= other.known_blocked_edges

    def _run_algorithm(self, graph, start: Node, goal: Node, algorithm: str, blocked_nodes, blocked_edges):
        if algorithm == "bfs":
            return bfs_path(graph, start, goal, blocked_nodes, blocked_edges)
        if algorithm == "dfs":
            return dfs_path(graph, start, goal, blocked_nodes, blocked_edges)
        return astar_path(graph, start, goal, blocked_nodes, blocked_edges, heuristic="manhattan")

    def plan_route(self, environment, algorithm: str = "astar", avoid_nodes: Optional[Set[Node]] = None) -> None:
        avoid_nodes = avoid_nodes or set()
        blocked_nodes = set(environment.fire_nodes) | set(environment.blocked_nodes)
        blocked_nodes |= self.known_fire_nodes | self.known_blocked_nodes | avoid_nodes
        blocked_nodes -= set(environment.exits)
        blocked_nodes.discard(self.current_node)

        blocked_edges = set(environment.blocked_edges) | self.known_blocked_edges

        exit_order = sorted(environment.exits, key=lambda e: environment.distance(self.current_node, e))

        for exit_node in exit_order:
            path = self._run_algorithm(
                environment.graph,
                self.current_node,
                exit_node,
                algorithm,
                blocked_nodes,
                blocked_edges,
            )
            if path:
                self.goal_node = exit_node
                self.planned_path = path
                self.status = "moving"
                return

        self.planned_path = []
        self.goal_node = None
        self.status = "stuck"

    def step(self, environment, algorithm: str = "astar", congestion_avoid: Optional[Set[Node]] = None) -> None:
        if self.status in {"evacuated", "stuck"}:
            return

        if self.current_node in environment.exits:
            self.status = "evacuated"
            self.goal_node = self.current_node
            self.planned_path = [self.current_node]
            return

        if not self.planned_path or len(self.planned_path) < 2:
            self.plan_route(environment, algorithm=algorithm, avoid_nodes=congestion_avoid)
            if not self.planned_path or len(self.planned_path) < 2:
                self.status = "stuck"
                return

        next_node = self.planned_path[1]

        needs_replan = (
            not environment.is_traversable(next_node)
            or not environment.is_edge_traversable(self.current_node, next_node)
            or (congestion_avoid is not None and next_node in congestion_avoid)
        )

        if needs_replan:
            self.plan_route(environment, algorithm=algorithm, avoid_nodes=congestion_avoid)
            if not self.planned_path or len(self.planned_path) < 2:
                self.status = "stuck"
                return
            next_node = self.planned_path[1]

        self.current_node = next_node
        self.planned_path = self.planned_path[1:]

        if self.current_node in environment.exits:
            self.status = "evacuated"
