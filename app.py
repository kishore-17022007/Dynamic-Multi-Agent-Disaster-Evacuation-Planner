from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Dict, List, Tuple

import streamlit as st
from streamlit_folium import st_folium

from simulation import DisasterSimulation
from visualization import render_leaflet_map

Node = Tuple[int, int]

st.set_page_config(page_title="Dynamic Multi-Agent Disaster Evacuation Planner", layout="wide")
st.title("Dynamic Multi-Agent Disaster Evacuation Planner")


def _init_state() -> None:
    if "sim" not in st.session_state:
        st.session_state.sim = DisasterSimulation(rows=12, cols=12, num_agents=20, algorithm="astar", seed=42)
    if "running" not in st.session_state:
        st.session_state.running = False
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_recorded_step" not in st.session_state:
        st.session_state.last_recorded_step = -1
    if "history_token" not in st.session_state:
        st.session_state.history_token = ""
    if "max_log_lines" not in st.session_state:
        st.session_state.max_log_lines = 200
    if "alerts" not in st.session_state:
        st.session_state.alerts = []
    if "alert_triggers" not in st.session_state:
        st.session_state.alert_triggers = {}
    if "dismissed_alerts" not in st.session_state:
        st.session_state.dismissed_alerts = set()
    if "alert_history" not in st.session_state:
        st.session_state.alert_history = []
    if "last_alert_step" not in st.session_state:
        st.session_state.last_alert_step = -1
    if "prev_fires" not in st.session_state:
        st.session_state.prev_fires = set()
    if "prev_floods" not in st.session_state:
        st.session_state.prev_floods = set()
    if "prev_landslides" not in st.session_state:
        st.session_state.prev_landslides = set()
    if "prev_blocked_roads" not in st.session_state:
        st.session_state.prev_blocked_roads = set()
    if "prev_blocked_nodes" not in st.session_state:
        st.session_state.prev_blocked_nodes = set()


def _sim_token(sim: DisasterSimulation) -> str:
    return f"{id(sim)}:{sim.env.rows}x{sim.env.cols}:{sim.num_agents}:{sim.seed}"


def _reset_history_if_needed(sim: DisasterSimulation) -> None:
    token = _sim_token(sim)
    if st.session_state.history_token != token:
        st.session_state.history = []
        st.session_state.last_recorded_step = -1
        st.session_state.history_token = token


def _record_history(stats: Dict[str, float]) -> None:
    step = int(stats["step"])
    if step == st.session_state.last_recorded_step:
        return

    st.session_state.history.append(
        {
            "step": step,
            "evacuated": int(stats["evacuated"]),
            "moving": int(stats["moving"]),
            "stuck": int(stats["stuck"]),
            "fire_nodes": int(stats["fire_nodes"]),
            "flood_nodes": int(stats["flood_nodes"]),
            "landslide_nodes": int(stats["landslide_nodes"]),
            "blocked_nodes": int(stats["blocked_nodes"]),
            "blocked_roads": int(stats["blocked_roads"]),
            "avg_congestion": float(stats["avg_congestion"]),
        }
    )
    st.session_state.history = st.session_state.history[-2000:]
    st.session_state.last_recorded_step = step


def _build_risk_index(stats: Dict[str, float]) -> float:
    raw = (
        float(stats["fire_nodes"]) * 4.0
        + float(stats["flood_nodes"]) * 3.0
        + float(stats["landslide_nodes"]) * 3.5
        + float(stats["blocked_nodes"]) * 2.0
        + float(stats["blocked_roads"]) * 1.5
        + float(stats["avg_congestion"]) * 8.0
        + float(stats["stuck"]) * 2.5
    )
    return round(min(raw, 100.0), 1)


def _timeline(history: List[Dict[str, float]], key: str) -> List[float]:
    return [point[key] for point in history]


def _status_counts(sim: DisasterSimulation) -> Dict[str, int]:
    counts = {"moving": 0, "stuck": 0, "evacuated": 0}
    for agent in sim.agents:
        counts[agent.status] = counts.get(agent.status, 0) + 1
    return counts


def _top_congested_nodes(sim: DisasterSimulation, top_k: int = 8) -> List[Dict[str, object]]:
    ranked = sorted(sim.env.congestion.items(), key=lambda item: item[1], reverse=True)
    results: List[Dict[str, object]] = []
    for node, count in ranked[:top_k]:
        if count <= 0:
            continue
        results.append({"node": str(node), "congestion": int(count)})
    return results


def _generate_alerts(sim: DisasterSimulation, stats: Dict[str, float], history: List[Dict[str, float]]) -> List[Dict[str, str]]:
    """Generate alerts based on critical simulation conditions."""
    active_alerts = []
    evac_rate = (stats["evacuated"] / stats["total_agents"] * 100.0) if stats["total_agents"] else 0.0
    stuck_rate = (stats["stuck"] / stats["total_agents"] * 100.0) if stats["total_agents"] else 0.0

    # === FIRE-SPECIFIC ALERTS ===
    
    # Critical: Very high fire spread
    if stats["fire_nodes"] > sim.env.rows * sim.env.cols * 0.25:
        active_alerts.append({
            "severity": "critical",
            "title": "🔥 MASSIVE FIRE SPREAD",
            "message": f"Fire has consumed {stats['fire_nodes']} zones ({stats['fire_nodes'] / (sim.env.rows * sim.env.cols) * 100:.1f}% of grid). IMMEDIATE INTERVENTION REQUIRED.",
            "id": "fire_critical"
        })

    # Critical: Fire expanding rapidly
    if stats["step"] > 5 and len(history) > 5:
        fire_trend = stats["fire_nodes"] - history[-1].get("fire_nodes", 0)
        if fire_trend >= 3:
            active_alerts.append({
                "severity": "critical",
                "title": "🔥 RAPID FIRE EXPANSION",
                "message": f"Fire spread to {fire_trend} NEW zones in last step! Grid coverage now {stats['fire_nodes']} zones.",
                "id": "fire_rapid_expansion"
            })

    # Warning: Fire zones reaching critical level
    fire_percentage = (stats["fire_nodes"] / (sim.env.rows * sim.env.cols) * 100) if (sim.env.rows * sim.env.cols) > 0 else 0
    if 15 < fire_percentage <= 25:
        active_alerts.append({
            "severity": "warning",
            "title": "🔥 Fire Spreading Out of Control",
            "message": f"Fire now covers {fire_percentage:.1f}% of grid ({stats['fire_nodes']} zones). Contain immediately!",
            "id": "fire_spreading"
        })
    elif 8 < fire_percentage <= 15:
        active_alerts.append({
            "severity": "warning",
            "title": "🔥 Fire Zone Expanding",
            "message": f"Fire coverage at {fire_percentage:.1f}% ({stats['fire_nodes']} zones). Monitor spread rate carefully.",
            "id": "fire_expanding"
        })

    # Info: Fire detected but manageable
    if stats["fire_nodes"] > 0 and stats["fire_nodes"] <= 3 and stats["step"] <= 50:
        active_alerts.append({
            "severity": "info",
            "title": "🔥 Fire Zone Detected",
            "message": f"Active fire in {stats['fire_nodes']} zone(s). Agents routing around fires.",
            "id": "fire_detected"
        })

    # === FLOOD-SPECIFIC ALERTS ===
    if stats["flood_nodes"] > sim.env.rows * sim.env.cols * 0.20:
        active_alerts.append({
            "severity": "critical",
            "title": "🌊 FLOODING ACROSS CITY",
            "message": f"Flood has affected {stats['flood_nodes']} zones. Mobility corridors are shrinking rapidly.",
            "id": "flood_critical"
        })
    elif stats["flood_nodes"] >= 4:
        active_alerts.append({
            "severity": "warning",
            "title": "🌊 Flood Zones Expanding",
            "message": f"{stats['flood_nodes']} flood zone(s) active. Agents are rerouting from waterlogged nodes.",
            "id": "flood_warning"
        })

    # === LANDSLIDE-SPECIFIC ALERTS ===
    if stats["landslide_nodes"] >= 3:
        active_alerts.append({
            "severity": "critical",
            "title": "⛰️ MULTIPLE LANDSLIDES",
            "message": f"{stats['landslide_nodes']} landslide nodes detected. Terrain instability blocking key paths.",
            "id": "landslide_critical"
        })
    elif stats["landslide_nodes"] >= 1:
        active_alerts.append({
            "severity": "warning",
            "title": "⛰️ Landslide Event",
            "message": f"{stats['landslide_nodes']} landslide node(s) active. Route recalculation in progress.",
            "id": "landslide_warning"
        })

    # === BLOCKAGE-SPECIFIC ALERTS ===

    # Critical: Critical route blockage affecting evacuation
    if stats["blocked_roads"] >= 5 and stats["stuck"] > stats["moving"]:
        active_alerts.append({
            "severity": "critical",
            "title": "🚫 CRITICAL ROUTE BLOCKAGE",
            "message": f"{stats['blocked_roads']} roads blocked! More agents stuck ({stats['stuck']}) than moving ({stats['moving']}).",
            "id": "blockage_critical_routes"
        })

    # Warning: Multiple roads blocked
    if stats["blocked_roads"] >= 3:
        active_alerts.append({
            "severity": "warning",
            "title": "🚧 MULTIPLE ROADS BLOCKED",
            "message": f"{stats['blocked_roads']} roads are blocked. Evacuation routes severely limited. {stats['blocked_nodes']} node(s) also blocked.",
            "id": "blockage_multiple_roads"
        })
    elif stats["blocked_roads"] >= 1:
        active_alerts.append({
            "severity": "warning",
            "title": "🛣️ Road Blockage Detected",
            "message": f"{stats['blocked_roads']} road(s) blocked. Agents rerouting around affected areas.",
            "id": "blockage_road_detected"
        })

    # Warning: Blocked nodes cutting off zones
    if stats["blocked_nodes"] >= 2:
        active_alerts.append({
            "severity": "warning",
            "title": "🏘️ NODES BLOCKED",
            "message": f"{stats['blocked_nodes']} zones are blocked/inaccessible. May trap agents or cut off exits.",
            "id": "blockage_nodes"
        })

    # === COMBINED HAZARD ALERTS ===

    # Critical: Many agents stuck
    if stuck_rate > 40:
        active_alerts.append({
            "severity": "critical",
            "title": "🚫 High Agent Failure Rate",
            "message": f"{stuck_rate:.1f}% of agents are stuck. Evacuation routes are severely compromised.",
            "id": "stuck_critical"
        })

    # Warning: Evacuation stalled
    if len(history) > 20 and evac_rate < 50 and stats["step"] > 50:
        recent_evac = [h["evacuated"] for h in history[-10:]]
        if len(set(recent_evac)) == 1:  # evacuation count not changing
            active_alerts.append({
                "severity": "warning",
                "title": "⏸️ Evacuation Stalled",
                "message": f"No agents evacuated in last 10 steps. Current rate: {evac_rate:.1f}%.",
                "id": "evac_stalled"
            })

    # Warning: High congestion
    if stats["avg_congestion"] > 3.5:
        active_alerts.append({
            "severity": "warning",
            "title": "🚗 Severe Congestion",
            "message": f"Average congestion level {stats['avg_congestion']:.1f} is critically high. Agents may be unable to move.",
            "id": "congestion_high"
        })

    # Warning: Significant blockages
    total_blockages = stats["blocked_nodes"] + stats["blocked_roads"]
    if total_blockages > 8:
        active_alerts.append({
            "severity": "warning",
            "title": "🚧 Network Fragmentation",
            "message": f"{total_blockages} network elements blocked. Limited escape routes available.",
            "id": "blockage_high"
        })

    # Info: Agent distribution imbalance
    moving_ratio = (stats["moving"] / stats["total_agents"]) if stats["total_agents"] else 0
    if moving_ratio < 0.1 and stats["step"] > 30 and not (stats["evacuated"] > 0.9 * stats["total_agents"]):
        active_alerts.append({
            "severity": "info",
            "title": "ℹ️ Low Movement Activity",
            "message": f"Only {stats['moving']} agents actively moving. Most may be queued or rerouting.",
            "id": "movement_low"
        })

    # Info: Fire accelerating slowly
    if stats["step"] > 0 and len(history) > 5:
        fire_trend = stats["fire_nodes"] - history[-1].get("fire_nodes", 0)
        if 1 <= fire_trend < 3:
            active_alerts.append({
                "severity": "info",
                "title": "🔥 Fire Spread Accelerating",
                "message": f"Fire spread {fire_trend} new zone(s) in the last step.",
                "id": "fire_spread_trend"
            })

    return active_alerts


def _display_alerts(alerts: List[Dict[str, str]]) -> None:
    """Display alerts in the dashboard with dismiss functionality."""
    if not alerts:
        return

    for alert in alerts:
        alert_id = alert.get("id", "unknown")
        
        # Skip if already dismissed
        if alert_id in st.session_state.dismissed_alerts:
            continue

        severity = alert.get("severity", "info")
        
        if severity == "critical":
            container = st.error
            icon = "🔴"
        elif severity == "warning":
            container = st.warning
            icon = "🟡"
        else:
            container = st.info
            icon = "🔵"

        col1, col2 = st.columns([0.95, 0.05])
        with col1:
            container(f"{icon} **{alert.get('title', 'Alert')}**\n\n{alert.get('message', '')}")
        with col2:
            if st.button("✕", key=f"dismiss_{alert_id}", help="Dismiss this alert"):
                st.session_state.dismissed_alerts.add(alert_id)
                st.rerun()


def _record_alert_history(active_alerts: List[Dict[str, str]], step: int) -> None:
    """Track alert history for timeline and statistics."""
    if step == st.session_state.last_alert_step:
        return
    
    for alert in active_alerts:
        if alert.get("id") not in st.session_state.dismissed_alerts:
            st.session_state.alert_history.append({
                "step": step,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "severity": alert.get("severity", "info"),
                "title": alert.get("title", "Unknown"),
                "message": alert.get("message", ""),
                "id": alert.get("id", "unknown"),
            })
    
    st.session_state.alert_history = st.session_state.alert_history[-1000:]
    st.session_state.last_alert_step = step


def _get_alert_stats(alerts: List[Dict[str, str]]) -> Dict[str, int]:
    """Get alert counts by severity."""
    return {
        "critical": sum(1 for a in alerts if a.get("severity") == "critical"),
        "warning": sum(1 for a in alerts if a.get("severity") == "warning"),
        "info": sum(1 for a in alerts if a.get("severity") == "info"),
        "total": len(alerts),
    }


def _detect_state_changes(sim: DisasterSimulation, step: int) -> List[Dict[str, str]]:
    """Detect new fires, blockages, and generate immediate event alerts."""
    event_alerts = []
    
    current_fires = set(sim.env.fire_nodes)
    current_floods = set(sim.env.flood_nodes)
    current_landslides = set(sim.env.landslide_nodes)
    current_blocked_roads = set(frozenset(e) for e in sim.env.blocked_edges)
    current_blocked_nodes = set(sim.env.blocked_nodes)
    
    # New fires detected
    new_fires = current_fires - st.session_state.prev_fires
    for fire_node in sorted(new_fires):
        event_alerts.append({
            "severity": "critical",
            "title": f"🔥 FIRE IGNITED AT {fire_node}",
            "message": f"New fire zone active at node {fire_node}. Agents must avoid this area immediately.",
            "id": f"fire_event_{step}_{fire_node}",
        })

    # New floods detected
    new_floods = current_floods - st.session_state.prev_floods
    for flood_node in sorted(new_floods):
        event_alerts.append({
            "severity": "critical",
            "title": f"🌊 FLOOD AT {flood_node}",
            "message": f"New flood zone active at node {flood_node}. Agents are avoiding submerged routes.",
            "id": f"flood_event_{step}_{flood_node}",
        })

    # New landslides detected
    new_landslides = current_landslides - st.session_state.prev_landslides
    for landslide_node in sorted(new_landslides):
        event_alerts.append({
            "severity": "critical",
            "title": f"⛰️ LANDSLIDE AT {landslide_node}",
            "message": f"Landslide detected at node {landslide_node}. Nearby paths may be inaccessible.",
            "id": f"landslide_event_{step}_{landslide_node}",
        })
    
    # New blocked roads detected
    new_blocked_roads = current_blocked_roads - st.session_state.prev_blocked_roads
    for blocked_edge in sorted(new_blocked_roads):
        edge_list = list(blocked_edge)
        u, v = edge_list[0], edge_list[1]
        event_alerts.append({
            "severity": "critical",
            "title": f"🚫 ROAD BLOCKED {u} ↔ {v}",
            "message": f"Route between {u} and {v} is now impassable. Agents rerouting around blockage.",
            "id": f"blockage_road_{step}_{u}_{v}",
        })
    
    # New blocked nodes detected
    new_blocked_nodes = current_blocked_nodes - st.session_state.prev_blocked_nodes
    for blocked_node in sorted(new_blocked_nodes):
        event_alerts.append({
            "severity": "critical",
            "title": f"🏘️ NODE BLOCKED {blocked_node}",
            "message": f"Zone {blocked_node} is now inaccessible. Evacuation routes may be affected.",
            "id": f"blockage_node_{step}_{blocked_node}",
        })
    
    # Update previous state
    st.session_state.prev_fires = current_fires
    st.session_state.prev_floods = current_floods
    st.session_state.prev_landslides = current_landslides
    st.session_state.prev_blocked_roads = current_blocked_roads
    st.session_state.prev_blocked_nodes = current_blocked_nodes
    
    return event_alerts


def _display_alert_center(active_alerts: List[Dict[str, str]]) -> None:
    """Display comprehensive alert center tab."""
    alert_stats = _get_alert_stats(active_alerts)
    
    # Alert summary cards
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("🔴 Critical", alert_stats["critical"])
    s2.metric("🟡 Warnings", alert_stats["warning"])
    s3.metric("🔵 Info", alert_stats["info"])
    s4.metric("📊 Total Active", alert_stats["total"])
    
    st.divider()
    
    # Active alerts with details
    if active_alerts:
        st.subheader("Active Alerts (Current Simulation)")
        
        # Filter controls
        col1, col2 = st.columns(2)
        with col1:
            severity_filter = st.multiselect(
                "Filter by severity",
                options=["critical", "warning", "info"],
                default=["critical", "warning", "info"],
            )
        
        with col2:
            search_term = st.text_input("Search alerts", value="")
        
        # Display filtered alerts
        filtered_alerts = [
            a for a in active_alerts
            if a.get("severity") in severity_filter
            and (not search_term or search_term.lower() in a.get("message", "").lower())
            and a.get("id") not in st.session_state.dismissed_alerts
        ]
        
        if filtered_alerts:
            for idx, alert in enumerate(filtered_alerts):
                with st.expander(f"{alert.get('title', 'Alert')} (Step {alert.get('step', 'Unknown')})", expanded=(idx == 0)):
                    col1, col2 = st.columns([0.9, 0.1])
                    with col1:
                        st.write(alert.get("message", ""))
                    with col2:
                        if st.button("Dismiss", key=f"alert_dismiss_{idx}"):
                            st.session_state.dismissed_alerts.add(alert.get("id", "unknown"))
                            st.rerun()
        else:
            st.info("No alerts match the current filters.")
    
    else:
        st.success("✅ No active alerts! Evacuation proceeding normally.")
    
    st.divider()
    
    # Alert history
    if st.session_state.alert_history:
        st.subheader("Alert History")
        history_df = []
        for event in st.session_state.alert_history[-100:]:  # Last 100 events
            history_df.append({
                "Step": event["step"],
                "Severity": event["severity"],
                "Title": event["title"],
                "Message": event["message"][:60] + "..." if len(event["message"]) > 60 else event["message"],
            })
        
        if history_df:
            st.dataframe(history_df, use_container_width=True, hide_index=True)
            
            # Alert trend
            st.subheader("Alert Timeline")
            severity_timeline = {}
            for event in st.session_state.alert_history:
                step = event["step"]
                severity = event["severity"]
                if severity not in severity_timeline:
                    severity_timeline[severity] = []
                severity_timeline[severity].append(step)
            
            timeline_data = {
                "step": sorted(set(e["step"] for e in st.session_state.alert_history)),
            }
            for severity in ["critical", "warning", "info"]:
                timeline_data[severity] = [
                    sum(1 for e in st.session_state.alert_history if e["step"] == s and e["severity"] == severity)
                    for s in timeline_data["step"]
                ]
            
            st.line_chart(
                {
                    "step": timeline_data["step"],
                    "critical": timeline_data["critical"],
                    "warning": timeline_data["warning"],
                    "info": timeline_data["info"],
                },
                x="step",
            )
    else:
        st.info("No alert history yet.")


_init_state()
sim: DisasterSimulation = st.session_state.sim
_reset_history_if_needed(sim)

with st.sidebar:
    st.header("Simulation Controls")

    rows = st.slider("Grid Rows", min_value=8, max_value=25, value=sim.env.rows)
    cols = st.slider("Grid Columns", min_value=8, max_value=25, value=sim.env.cols)
    num_agents = st.slider("Number of Agents", min_value=1, max_value=120, value=sim.num_agents)
    algo_options = ["A*", "BFS", "DFS"]
    algo_to_index = {"astar": 0, "bfs": 1, "dfs": 2}
    algorithm_label = st.selectbox("Pathfinding Algorithm", algo_options, index=algo_to_index.get(sim.algorithm, 0))
    algorithm = algorithm_label.lower().replace("*", "star")
    step_delay = st.slider("Auto Step Delay (seconds)", min_value=0.1, max_value=2.0, value=0.6, step=0.1)
    map_height = st.slider("Map Height", min_value=420, max_value=900, value=680, step=20)

    st.markdown("---")
    st.subheader("Scenario Presets")
    scenario_choice = st.selectbox("Preset", options=sim.scenario_names())
    auto_start_after_preset = st.toggle("Auto-start after preset", value=True)
    if st.button("Apply Scenario Preset", use_container_width=True):
        st.session_state.running = False
        if sim.apply_scenario(scenario_choice):
            # Reset alert state tracking to detect scenario fires/blockages
            st.session_state.prev_fires = set()
            st.session_state.prev_floods = set()
            st.session_state.prev_landslides = set()
            st.session_state.prev_blocked_roads = set()
            st.session_state.prev_blocked_nodes = set()
            if auto_start_after_preset:
                st.session_state.running = True
            st.success(f"Applied scenario: {scenario_choice}")
        else:
            st.error("Failed to apply scenario preset.")
        st.rerun()

    st.markdown("---")
    st.subheader("Disaster Probabilities")
    sim.p_fire_spread = st.slider("Fire Spread Probability", 0.0, 1.0, float(sim.p_fire_spread), 0.01)
    sim.p_flood_spread = st.slider("Flood Spread Probability", 0.0, 1.0, float(sim.p_flood_spread), 0.01)
    sim.p_new_landslide = st.slider("New Landslide Probability", 0.0, 1.0, float(sim.p_new_landslide), 0.01)
    sim.p_new_block_node = st.slider("New Blocked Node Probability", 0.0, 1.0, float(sim.p_new_block_node), 0.01)
    sim.p_new_block_edge = st.slider("New Blocked Road Probability", 0.0, 1.0, float(sim.p_new_block_edge), 0.01)
    sim.congestion_threshold = st.slider("Congestion Threshold", min_value=2, max_value=10, value=int(sim.congestion_threshold))

    col1, col2 = st.columns(2)
    if col1.button("Start Simulation", use_container_width=True):
        st.session_state.running = True
    if col2.button("Pause Simulation", use_container_width=True):
        st.session_state.running = False

    col3, col4 = st.columns(2)
    if col3.button("Run One Step", use_container_width=True):
        st.session_state.running = False
        sim.tick()
    if col4.button("Reset Simulation", use_container_width=True):
        st.session_state.sim = DisasterSimulation(
            rows=rows,
            cols=cols,
            num_agents=num_agents,
            algorithm=algorithm,
            seed=42,
        )
        st.session_state.running = False
        # Reset alert state tracking
        st.session_state.prev_fires = set()
        st.session_state.prev_floods = set()
        st.session_state.prev_landslides = set()
        st.session_state.prev_blocked_roads = set()
        st.session_state.prev_blocked_nodes = set()
        st.session_state.dismissed_alerts = set()
        st.rerun()

    st.markdown("---")
    st.subheader("Dashboard Settings")
    st.session_state.max_log_lines = st.slider(
        "Visible Log Lines", min_value=50, max_value=600, value=st.session_state.max_log_lines, step=50
    )

    st.markdown("---")
    st.subheader("Alert Management")
    if st.button("Clear Dismissed Alerts", use_container_width=True, help="Restore all dismissed alerts"):
        st.session_state.dismissed_alerts.clear()
        st.rerun()

    st.markdown("---")
    st.subheader("Manual Hazard Injection")

    node_options = list(sim.env.graph.nodes)
    fire_node: Node = st.selectbox("Add Fire Location", options=node_options, format_func=lambda n: f"Node {n}")
    if st.button("Add Fire", use_container_width=True):
        if not sim.add_fire_location(fire_node):
            st.warning("Cannot place fire on this node (likely an exit).")

    flood_node: Node = st.selectbox("Add Flood Location", options=node_options, format_func=lambda n: f"Node {n}")
    if st.button("Add Flood", use_container_width=True):
        if not sim.add_flood_location(flood_node):
            st.warning("Cannot place flood on this node (likely an exit).")

    landslide_node: Node = st.selectbox(
        "Add Landslide Location", options=node_options, format_func=lambda n: f"Node {n}"
    )
    if st.button("Add Landslide", use_container_width=True):
        if not sim.add_landslide_location(landslide_node):
            st.warning("Cannot place landslide on this node (likely an exit).")

    edge_options = list(sim.env.graph.edges)
    blocked_edge = st.selectbox(
        "Add Blocked Road",
        options=edge_options,
        format_func=lambda e: f"Road {e[0]} ↔ {e[1]}",
    )
    if st.button("Add Blocked Road", use_container_width=True):
        sim.add_blocked_road(blocked_edge)

sim.algorithm = algorithm

stats = sim.stats()
_record_history(stats)
history: List[Dict[str, float]] = st.session_state.history

risk_index = _build_risk_index(stats)
evac_rate = (stats["evacuated"] / stats["total_agents"] * 100.0) if stats["total_agents"] else 0.0
status_counts = _status_counts(sim)

timeline_steps = _timeline(history, "step")
timeline_evac = _timeline(history, "evacuated")
timeline_stuck = _timeline(history, "stuck")
timeline_fire = _timeline(history, "fire_nodes")
timeline_flood = _timeline(history, "flood_nodes")
timeline_landslide = _timeline(history, "landslide_nodes")
timeline_congestion = _timeline(history, "avg_congestion")

# Generate and display active alerts
active_alerts = _generate_alerts(sim, stats, history)

# Detect state changes (new fires, blockages) and add immediate alerts
state_change_alerts = _detect_state_changes(sim, int(stats["step"]))
active_alerts = state_change_alerts + active_alerts  # Prepend state change alerts for priority display

_record_alert_history(active_alerts, stats["step"])

if active_alerts:
    st.markdown("## 🚨 Active Alerts")
    _display_alerts(active_alerts)
    st.markdown("---")

overview_tab, alerts_tab, analytics_tab, ops_tab = st.tabs(["Command Center", "🚨 Alerts", "Analytics", "Operations"])

with overview_tab:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Step", stats["step"])
    c2.metric("Evacuated", f"{stats['evacuated']}/{stats['total_agents']}")
    c3.metric("Moving", stats["moving"])
    c4.metric("Stuck", stats["stuck"])
    c5.metric("Evacuation Rate", f"{evac_rate:.1f}%")

    c6, c7, c8, c9, c10, c11 = st.columns(6)
    c6.metric("Fire Zones", stats["fire_nodes"])
    c7.metric("Flood Zones", stats["flood_nodes"])
    c8.metric("Landslides", stats["landslide_nodes"])
    c9.metric("Blocked Nodes", stats["blocked_nodes"])
    c10.metric("Blocked Roads", stats["blocked_roads"])
    c11.metric("Risk Index", f"{risk_index:.1f}/100")

    # Alert summary
    critical_count = sum(1 for a in active_alerts if a.get("severity") == "critical")
    warning_count = sum(1 for a in active_alerts if a.get("severity") == "warning")
    info_count = sum(1 for a in active_alerts if a.get("severity") == "info")
    
    if critical_count > 0 or warning_count > 0 or info_count > 0:
        alert_col1, alert_col2, alert_col3 = st.columns(3)
        if critical_count > 0:
            alert_col1.error(f"🔴 **{critical_count}** Critical Alert{'' if critical_count == 1 else 's'}")
        if warning_count > 0:
            alert_col2.warning(f"🟡 **{warning_count}** Warning{'' if warning_count == 1 else 's'}")
        if info_count > 0:
            alert_col3.info(f"🔵 **{info_count}** Info")
        st.divider()

    st.progress(min(max(evac_rate / 100.0, 0.0), 1.0), text="Evacuation progress")

    show_heatmap = st.checkbox("Show Congestion Heatmap", value=True)
    fmap = render_leaflet_map(sim.env, sim.agents, show_heatmap=show_heatmap)
    st_folium(fmap, width=None, height=map_height, returned_objects=[])

with alerts_tab:
    _display_alert_center(active_alerts)

with analytics_tab:
    left, right = st.columns(2)
    with left:
        st.markdown("### Agent Status Timeline")
        st.line_chart(
            {
                "step": timeline_steps,
                "evacuated": timeline_evac,
                "stuck": timeline_stuck,
                "moving": _timeline(history, "moving"),
            },
            x="step",
        )

        st.markdown("### Hazard Timeline")
        st.line_chart(
            {
                "step": timeline_steps,
                "fire_nodes": timeline_fire,
                "flood_nodes": timeline_flood,
                "landslide_nodes": timeline_landslide,
                "blocked_nodes": _timeline(history, "blocked_nodes"),
                "blocked_roads": _timeline(history, "blocked_roads"),
            },
            x="step",
        )

    with right:
        st.markdown("### Congestion Trend")
        st.area_chart(
            {
                "step": timeline_steps,
                "avg_congestion": timeline_congestion,
            },
            x="step",
        )

        st.markdown("### Live Agent Distribution")
        st.bar_chart(
            {
                "status": ["moving", "stuck", "evacuated"],
                "count": [status_counts["moving"], status_counts["stuck"], status_counts["evacuated"]],
            },
            x="status",
        )

        top_nodes = _top_congested_nodes(sim)
        st.markdown("### Top Congested Nodes")
        if top_nodes:
            st.dataframe(top_nodes, use_container_width=True, hide_index=True)
        else:
            st.info("No congestion hotspots yet.")

with ops_tab:
    st.markdown("### Agent Inspector")
    agent_ids = [agent.agent_id for agent in sim.agents]
    selected_agent_id = st.selectbox("Select Agent", options=agent_ids)
    selected_agent = next((a for a in sim.agents if a.agent_id == selected_agent_id), None)
    if selected_agent is not None:
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Current Node", str(selected_agent.current_node))
        i2.metric("Goal Node", str(selected_agent.goal_node))
        i3.metric("Status", selected_agent.status)
        i4.metric("Planned Path Length", len(selected_agent.planned_path))

        st.dataframe(
            [
                {
                    "known_fire_nodes": len(selected_agent.known_fire_nodes),
                    "known_flood_nodes": len(selected_agent.known_flood_nodes),
                    "known_landslide_nodes": len(selected_agent.known_landslide_nodes),
                    "known_blocked_nodes": len(selected_agent.known_blocked_nodes),
                    "known_blocked_roads": len(selected_agent.known_blocked_edges),
                }
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Simulation Logs")
    recent_logs = sim.logs[-st.session_state.max_log_lines :]
    keyword = st.text_input("Filter logs by keyword", value="")
    if keyword.strip():
        filtered = [line for line in recent_logs if keyword.lower() in line.lower()]
    else:
        filtered = recent_logs

    st.text_area("Recent Events", value="\n".join(filtered), height=280)

    snapshot = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "stats": stats,
        "status_counts": status_counts,
        "top_congested_nodes": _top_congested_nodes(sim, top_k=12),
        "recent_logs": sim.logs[-120:],
    }
    st.download_button(
        label="Download Dashboard Snapshot (JSON)",
        data=json.dumps(snapshot, indent=2),
        file_name="evacuation_dashboard_snapshot.json",
        mime="application/json",
        use_container_width=True,
    )

if st.session_state.running:
    if sim.is_finished():
        st.session_state.running = False
        st.success("Simulation complete: all agents are evacuated or stuck.")
    else:
        sim.tick()
        time.sleep(step_delay)
        st.rerun()
