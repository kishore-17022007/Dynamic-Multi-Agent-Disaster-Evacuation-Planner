from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from agent import EvacuationAgent
from environment import DisasterEnvironment

Node = Tuple[int, int]


class DisasterSimulation:
    def __init__(
        self,
        rows: int = 12,
        cols: int = 12,
        num_agents: int = 20,
        algorithm: str = "astar",
        seed: Optional[int] = 42,
    ) -> None:
        self.algorithm = algorithm.lower()
        self.num_agents = num_agents
        self.seed = seed

        self.env = DisasterEnvironment(rows=rows, cols=cols, seed=seed)
        self.agents: List[EvacuationAgent] = []
        self.logs: List[str] = []
        self.step_count = 0

        self.p_fire_spread = 0.12
        self.p_flood_spread = 0.10
        self.p_new_landslide = 0.05
        self.p_new_block_node = 0.05
        self.p_new_block_edge = 0.06
        self.congestion_threshold = 3

        self._spawn_agents()

    def _scaled_node(self, row_ratio: float, col_ratio: float) -> Node:
        row = min(max(int(round((self.env.rows - 1) * row_ratio)), 0), self.env.rows - 1)
        col = min(max(int(round((self.env.cols - 1) * col_ratio)), 0), self.env.cols - 1)
        return (row, col)

    def _scenario_catalog(self) -> Dict[str, Dict]:
        return {
            "Urban Fire": {
                "origin_latlon": (28.6139, 77.2090),
                "probabilities": {
                    "p_fire_spread": 0.20,
                    "p_flood_spread": 0.04,
                    "p_new_landslide": 0.03,
                    "p_new_block_node": 0.04,
                    "p_new_block_edge": 0.05,
                    "congestion_threshold": 3,
                },
                "fires": [
                    self._scaled_node(0.45, 0.45),
                    self._scaled_node(0.55, 0.50),
                    self._scaled_node(0.50, 0.62),
                ],
                "floods": [
                    self._scaled_node(0.62, 0.18),
                ],
                "landslides": [
                    self._scaled_node(0.78, 0.78),
                ],
                "blocked_roads": [
                    (
                        self._scaled_node(0.50, 0.35),
                        self._scaled_node(0.50, 0.42),
                    ),
                    (
                        self._scaled_node(0.35, 0.52),
                        self._scaled_node(0.42, 0.52),
                    ),
                ],
            },
            "Road Collapse": {
                "origin_latlon": (28.7041, 77.1025),
                "probabilities": {
                    "p_fire_spread": 0.08,
                    "p_flood_spread": 0.05,
                    "p_new_landslide": 0.08,
                    "p_new_block_node": 0.10,
                    "p_new_block_edge": 0.14,
                    "congestion_threshold": 3,
                },
                "fires": [
                    self._scaled_node(0.20, 0.75),
                ],
                "floods": [
                    self._scaled_node(0.75, 0.20),
                ],
                "landslides": [
                    self._scaled_node(0.35, 0.28),
                    self._scaled_node(0.67, 0.58),
                ],
                "blocked_roads": [
                    (
                        self._scaled_node(0.30, 0.30),
                        self._scaled_node(0.30, 0.38),
                    ),
                    (
                        self._scaled_node(0.30, 0.38),
                        self._scaled_node(0.30, 0.46),
                    ),
                    (
                        self._scaled_node(0.30, 0.46),
                        self._scaled_node(0.30, 0.54),
                    ),
                    (
                        self._scaled_node(0.60, 0.58),
                        self._scaled_node(0.67, 0.58),
                    ),
                ],
            },
            "Congestion Surge": {
                "origin_latlon": (12.9716, 77.5946),
                "probabilities": {
                    "p_fire_spread": 0.10,
                    "p_flood_spread": 0.06,
                    "p_new_landslide": 0.03,
                    "p_new_block_node": 0.04,
                    "p_new_block_edge": 0.06,
                    "congestion_threshold": 2,
                },
                "fires": [
                    self._scaled_node(0.70, 0.30),
                ],
                "floods": [
                    self._scaled_node(0.55, 0.65),
                ],
                "landslides": [
                    self._scaled_node(0.26, 0.74),
                ],
                "blocked_roads": [
                    (
                        self._scaled_node(0.45, 0.40),
                        self._scaled_node(0.45, 0.48),
                    ),
                    (
                        self._scaled_node(0.45, 0.48),
                        self._scaled_node(0.45, 0.56),
                    ),
                ],
            },
            "Mumbai Monsoon Flood": {
                "origin_latlon": (19.0760, 72.8777),
                "probabilities": {
                    "p_fire_spread": 0.05,
                    "p_flood_spread": 0.22,
                    "p_new_landslide": 0.05,
                    "p_new_block_node": 0.06,
                    "p_new_block_edge": 0.10,
                    "congestion_threshold": 2,
                },
                "fires": [
                    self._scaled_node(0.18, 0.80),
                ],
                "floods": [
                    self._scaled_node(0.55, 0.30),
                    self._scaled_node(0.60, 0.36),
                    self._scaled_node(0.65, 0.42),
                ],
                "landslides": [
                    self._scaled_node(0.82, 0.24),
                ],
                "blocked_roads": [
                    (
                        self._scaled_node(0.58, 0.25),
                        self._scaled_node(0.58, 0.33),
                    ),
                    (
                        self._scaled_node(0.64, 0.35),
                        self._scaled_node(0.64, 0.42),
                    ),
                ],
            },
            "Himalayan Landslide": {
                "origin_latlon": (30.3165, 78.0322),
                "probabilities": {
                    "p_fire_spread": 0.04,
                    "p_flood_spread": 0.08,
                    "p_new_landslide": 0.16,
                    "p_new_block_node": 0.10,
                    "p_new_block_edge": 0.12,
                    "congestion_threshold": 3,
                },
                "fires": [
                    self._scaled_node(0.22, 0.60),
                ],
                "floods": [
                    self._scaled_node(0.74, 0.62),
                ],
                "landslides": [
                    self._scaled_node(0.30, 0.38),
                    self._scaled_node(0.36, 0.44),
                    self._scaled_node(0.43, 0.50),
                ],
                "blocked_roads": [
                    (
                        self._scaled_node(0.34, 0.34),
                        self._scaled_node(0.34, 0.42),
                    ),
                    (
                        self._scaled_node(0.40, 0.46),
                        self._scaled_node(0.47, 0.46),
                    ),
                ],
            },
            "Fire Prone India": {
                "origin_latlon": (23.3441, 85.3096),
                "probabilities": {
                    "p_fire_spread": 0.26,
                    "p_flood_spread": 0.04,
                    "p_new_landslide": 0.04,
                    "p_new_block_node": 0.06,
                    "p_new_block_edge": 0.08,
                    "congestion_threshold": 3,
                },
                "fires": [
                    self._scaled_node(0.22, 0.22),
                    self._scaled_node(0.30, 0.28),
                    self._scaled_node(0.38, 0.24),
                    self._scaled_node(0.46, 0.32),
                ],
                "floods": [
                    self._scaled_node(0.70, 0.72),
                ],
                "landslides": [
                    self._scaled_node(0.58, 0.48),
                ],
                "blocked_roads": [
                    (
                        self._scaled_node(0.34, 0.20),
                        self._scaled_node(0.34, 0.28),
                    ),
                    (
                        self._scaled_node(0.42, 0.28),
                        self._scaled_node(0.42, 0.36),
                    ),
                ],
            },
            "Flood Prone India": {
                "origin_latlon": (26.2006, 92.9376),
                "probabilities": {
                    "p_fire_spread": 0.05,
                    "p_flood_spread": 0.28,
                    "p_new_landslide": 0.06,
                    "p_new_block_node": 0.08,
                    "p_new_block_edge": 0.11,
                    "congestion_threshold": 2,
                },
                "fires": [
                    self._scaled_node(0.15, 0.80),
                ],
                "floods": [
                    self._scaled_node(0.62, 0.30),
                    self._scaled_node(0.68, 0.36),
                    self._scaled_node(0.74, 0.42),
                    self._scaled_node(0.80, 0.48),
                ],
                "landslides": [
                    self._scaled_node(0.82, 0.24),
                    self._scaled_node(0.28, 0.66),
                ],
                "blocked_roads": [
                    (
                        self._scaled_node(0.66, 0.28),
                        self._scaled_node(0.66, 0.36),
                    ),
                    (
                        self._scaled_node(0.72, 0.34),
                        self._scaled_node(0.72, 0.42),
                    ),
                    (
                        self._scaled_node(0.78, 0.40),
                        self._scaled_node(0.78, 0.48),
                    ),
                ],
            },
        }

    def scenario_names(self) -> List[str]:
        return list(self._scenario_catalog().keys())

    def apply_scenario(self, name: str) -> bool:
        scenario = self._scenario_catalog().get(name)
        if scenario is None:
            return False

        origin_latlon = scenario.get("origin_latlon")
        if origin_latlon is not None:
            self.env.set_origin_latlon(origin_latlon)

        self.env.reset_disaster()

        probs = scenario.get("probabilities", {})
        self.p_fire_spread = float(probs.get("p_fire_spread", self.p_fire_spread))
        self.p_flood_spread = float(probs.get("p_flood_spread", self.p_flood_spread))
        self.p_new_landslide = float(probs.get("p_new_landslide", self.p_new_landslide))
        self.p_new_block_node = float(probs.get("p_new_block_node", self.p_new_block_node))
        self.p_new_block_edge = float(probs.get("p_new_block_edge", self.p_new_block_edge))
        self.congestion_threshold = int(probs.get("congestion_threshold", self.congestion_threshold))

        for node in scenario.get("fires", []):
            self.env.add_fire(node)

        for node in scenario.get("floods", []):
            self.env.add_flood(node)

        for node in scenario.get("landslides", []):
            self.env.add_landslide(node)

        for edge in scenario.get("blocked_roads", []):
            u, v = edge
            self.env.add_blocked_road(u, v)

        for agent in self.agents:
            agent.known_fire_nodes.clear()
            agent.known_flood_nodes.clear()
            agent.known_landslide_nodes.clear()
            agent.known_blocked_nodes.clear()
            agent.known_blocked_edges.clear()
            if (
                agent.current_node in self.env.fire_nodes
                or agent.current_node in self.env.flood_nodes
                or agent.current_node in self.env.landslide_nodes
                or agent.current_node in self.env.blocked_nodes
            ):
                safe = self.env.random_safe_node()
                if safe is not None:
                    agent.current_node = safe
            agent.status = "moving"
            agent.plan_route(self.env, algorithm=self.algorithm)

        self.logs.append(f"Scenario preset applied: {name}.")
        return True

    def _spawn_agents(self) -> None:
        self.agents = []
        occupied = set()
        for idx in range(self.num_agents):
            node = self.env.random_safe_node(exclude=occupied)
            if node is None:
                break
            occupied.add(node)
            agent = EvacuationAgent(agent_id=idx, current_node=node)
            agent.plan_route(self.env, algorithm=self.algorithm)
            self.agents.append(agent)

        self.logs.append(f"Initialized with {len(self.agents)} agents.")

    def reset(self, num_agents: Optional[int] = None, algorithm: Optional[str] = None) -> None:
        if num_agents is not None:
            self.num_agents = num_agents
        if algorithm is not None:
            self.algorithm = algorithm.lower()

        self.env.reset_disaster()
        self.logs = []
        self.step_count = 0
        self._spawn_agents()

    def add_fire_location(self, node: Node) -> bool:
        ok = self.env.add_fire(node)
        if ok:
            self.logs.append(f"Manual fire added at {node}.")
        return ok

    def add_flood_location(self, node: Node) -> bool:
        ok = self.env.add_flood(node)
        if ok:
            self.logs.append(f"Manual flood added at {node}.")
        return ok

    def add_landslide_location(self, node: Node) -> bool:
        ok = self.env.add_landslide(node)
        if ok:
            self.logs.append(f"Manual landslide added at {node}.")
        return ok

    def add_blocked_road(self, edge: Tuple[Node, Node]) -> bool:
        u, v = edge
        ok = self.env.add_blocked_road(u, v)
        if ok:
            self.logs.append(f"Manual blocked road added between {u} and {v}.")
        return ok

    def _broadcast_knowledge(self) -> None:
        all_fire = set()
        all_flood = set()
        all_landslide = set()
        all_blocked_nodes = set()
        all_blocked_edges = set()

        for agent in self.agents:
            all_fire |= agent.known_fire_nodes
            all_flood |= agent.known_flood_nodes
            all_landslide |= agent.known_landslide_nodes
            all_blocked_nodes |= agent.known_blocked_nodes
            all_blocked_edges |= agent.known_blocked_edges

        for agent in self.agents:
            agent.known_fire_nodes |= all_fire
            agent.known_flood_nodes |= all_flood
            agent.known_landslide_nodes |= all_landslide
            agent.known_blocked_nodes |= all_blocked_nodes
            agent.known_blocked_edges |= all_blocked_edges

    def tick(self) -> None:
        self.step_count += 1

        updates = self.env.probability_update(
            p_fire_spread=self.p_fire_spread,
            p_flood_spread=self.p_flood_spread,
            p_new_landslide=self.p_new_landslide,
            p_new_block_node=self.p_new_block_node,
            p_new_block_edge=self.p_new_block_edge,
        )

        for node in updates["new_fire"]:
            self.logs.append(f"Step {self.step_count}: Fire spread to {node}.")
        for node in updates["new_flood"]:
            self.logs.append(f"Step {self.step_count}: Flood expanded to {node}.")
        for node in updates["new_landslide"]:
            self.logs.append(f"Step {self.step_count}: Landslide occurred at {node}.")
        for node in updates["new_blocked_nodes"]:
            self.logs.append(f"Step {self.step_count}: Node blocked at {node}.")
        for u, v in updates["new_blocked_roads"]:
            self.logs.append(f"Step {self.step_count}: Road blocked between {u} and {v}.")

        for agent in self.agents:
            agent.perceive(self.env)

        self._broadcast_knowledge()

        self.env.update_congestion([a.current_node for a in self.agents if a.status != "evacuated"])
        congestion_avoid = self.env.congested_nodes(self.congestion_threshold)

        for agent in self.agents:
            if agent.status == "evacuated":
                continue

            if agent.planned_path and len(agent.planned_path) >= 2:
                next_node = agent.planned_path[1]
                if next_node in congestion_avoid and next_node not in self.env.exits:
                    self.logs.append(
                        f"Step {self.step_count}: Agent {agent.agent_id} rerouting due to congestion near {next_node}."
                    )

            prev = agent.current_node
            prev_status = agent.status
            agent.step(self.env, algorithm=self.algorithm, congestion_avoid=congestion_avoid)

            if agent.current_node != prev:
                self.logs.append(
                    f"Step {self.step_count}: Agent {agent.agent_id} moved {prev} -> {agent.current_node}."
                )
            if prev_status != "evacuated" and agent.status == "evacuated":
                self.logs.append(f"Step {self.step_count}: Agent {agent.agent_id} evacuated.")
            if prev_status != "stuck" and agent.status == "stuck":
                self.logs.append(f"Step {self.step_count}: Agent {agent.agent_id} is stuck.")

        self.env.update_congestion([a.current_node for a in self.agents if a.status != "evacuated"])

    def is_finished(self) -> bool:
        return all(agent.status in {"evacuated", "stuck"} for agent in self.agents)

    def stats(self) -> Dict[str, float]:
        evacuated = sum(1 for a in self.agents if a.status == "evacuated")
        stuck = sum(1 for a in self.agents if a.status == "stuck")
        moving = sum(1 for a in self.agents if a.status == "moving")
        avg_congestion = (
            sum(self.env.congestion.values()) / len(self.env.congestion)
            if self.env.congestion
            else 0.0
        )

        return {
            "step": self.step_count,
            "total_agents": len(self.agents),
            "evacuated": evacuated,
            "moving": moving,
            "stuck": stuck,
            "fire_nodes": len(self.env.fire_nodes),
            "flood_nodes": len(self.env.flood_nodes),
            "landslide_nodes": len(self.env.landslide_nodes),
            "blocked_nodes": len(self.env.blocked_nodes),
            "blocked_roads": len(self.env.blocked_edges),
            "avg_congestion": round(avg_congestion, 2),
        }
