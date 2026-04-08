from __future__ import annotations

from typing import List, Tuple

import folium

Node = Tuple[int, int]


def _agent_color(status: str) -> str:
    if status == "evacuated":
        return "green"
    if status == "stuck":
        return "orange"
    return "blue"


def _agent_bike_icon(status: str) -> folium.DivIcon:
    color = _agent_color(status)
    html = (
        "<div style=\""
        "width:24px;"
        "height:24px;"
        "border-radius:50%;"
        f"background:{color};"
        "display:flex;"
        "align-items:center;"
        "justify-content:center;"
        "font-size:14px;"
        "border:2px solid white;"
        "box-shadow:0 0 4px rgba(0,0,0,0.35);"
        "\">🚴</div>"
    )
    return folium.DivIcon(html=html, icon_size=(24, 24), icon_anchor=(12, 12), class_name="agent-bike")


def _hotspot_center(environment) -> Tuple[float, float]:
    node_score = {node: 0 for node in environment.graph.nodes}

    for node in environment.fire_nodes:
        node_score[node] += 5

    for node in environment.blocked_nodes:
        node_score[node] += 3

    for edge in environment.blocked_edges:
        u, v = tuple(edge)
        node_score[u] += 2
        node_score[v] += 2

    for node, count in environment.congestion.items():
        node_score[node] += count

    hotspot_node = max(node_score, key=node_score.get)
    if node_score[hotspot_node] == 0:
        center_lat = environment.origin_latlon[0] + (environment.rows * environment.cell_step) / 2
        center_lon = environment.origin_latlon[1] + (environment.cols * environment.cell_step) / 2
        return (center_lat, center_lon)

    return environment.node_to_latlon(hotspot_node)


def render_leaflet_map(environment, agents: List, show_heatmap: bool = True) -> folium.Map:
    center_lat, center_lon = _hotspot_center(environment)
    fmap = folium.Map(location=(center_lat, center_lon), zoom_start=14, control_scale=True)

    for u, v in environment.graph.edges:
        latlon_u = environment.node_to_latlon(u)
        latlon_v = environment.node_to_latlon(v)
        blocked = frozenset((u, v)) in environment.blocked_edges
        folium.PolyLine(
            locations=[latlon_u, latlon_v],
            color="black" if blocked else "#7f8c8d",
            weight=5 if blocked else 2,
            opacity=0.9 if blocked else 0.6,
        ).add_to(fmap)

    for fire in environment.fire_nodes:
        folium.Circle(
            location=environment.node_to_latlon(fire),
            radius=90,
            color="red",
            fill=True,
            fill_opacity=0.35,
            popup=f"Fire: {fire}",
        ).add_to(fmap)

    for blocked_node in environment.blocked_nodes:
        folium.CircleMarker(
            location=environment.node_to_latlon(blocked_node),
            radius=4,
            color="black",
            fill=True,
            fill_opacity=1.0,
            popup=f"Blocked Node: {blocked_node}",
        ).add_to(fmap)

    for exit_node in environment.exits:
        folium.Marker(
            location=environment.node_to_latlon(exit_node),
            popup=f"Exit: {exit_node}",
            icon=folium.Icon(color="green", icon="ok-sign"),
        ).add_to(fmap)

    if show_heatmap:
        for node, count in environment.congestion.items():
            if count <= 1:
                continue
            folium.Circle(
                location=environment.node_to_latlon(node),
                radius=35 + 20 * count,
                color="#e67e22",
                fill=True,
                fill_color="#e67e22",
                fill_opacity=min(0.15 + count * 0.05, 0.5),
                popup=f"Congestion {node}: {count}",
            ).add_to(fmap)

    for agent in agents:
        folium.Marker(
            location=environment.node_to_latlon(agent.current_node),
            icon=_agent_bike_icon(agent.status),
            popup=(
                f"Agent {agent.agent_id}<br>"
                f"Status: {agent.status}<br>"
                f"Location: {agent.current_node}<br>"
                f"Goal: {agent.goal_node}"
            ),
        ).add_to(fmap)

    return fmap
