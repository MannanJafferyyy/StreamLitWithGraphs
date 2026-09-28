import heapq
import math
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st


# Task 3 hospital graph
HOSPITAL_COORDINATES: Dict[str, Tuple[float, float]] = {
	"Pharmacy": (0, 0),
	"Main_Corridor": (2, 1),
	"Patient_Wing": (1, 4),
	"Nursing_Station": (4, 2),
	"Laboratory": (5, 5),
	"Emergency_Ward": (8, 6),
}

HOSPITAL_EDGES = [
	("Pharmacy", "Patient_Wing", 4.1),
	("Pharmacy", "Main_Corridor", 2.2),
	("Patient_Wing", "Laboratory", 5.0),
	("Main_Corridor", "Nursing_Station", 2.2),
	("Nursing_Station", "Laboratory", 3.2),
	("Nursing_Station", "Emergency_Ward", 6.0),
	("Laboratory", "Emergency_Ward", 3.2),
]


def build_hospital_graph() -> nx.DiGraph:
	graph = nx.DiGraph()
	graph.add_nodes_from(HOSPITAL_COORDINATES)
	graph.add_weighted_edges_from(HOSPITAL_EDGES)
	return graph


def euclidean_heuristic(node: str, goal: str) -> float:
	node_x, node_y = HOSPITAL_COORDINATES[node]
	goal_x, goal_y = HOSPITAL_COORDINATES[goal]
	return math.hypot(node_x - goal_x, node_y - goal_y)


def reconstruct_path(came_from: Dict[str, Optional[str]], goal: str) -> List[str]:
	path = []
	current: Optional[str] = goal
	while current is not None:
		path.append(current)
		current = came_from.get(current)
	path.reverse()
	return path


def gbfs(
	graph: nx.DiGraph, start: str, goal: str
) -> Tuple[List[str], List[str]]:
	frontier = [(euclidean_heuristic(start, goal), start)]
	came_from: Dict[str, Optional[str]] = {start: None}
	expanded: List[str] = []

	while frontier:
		_, current = heapq.heappop(frontier)
		expanded.append(current)

		if current == goal:
			return reconstruct_path(came_from, goal), expanded

		for neighbor in graph.successors(current):
			if neighbor not in came_from:
				came_from[neighbor] = current
				heapq.heappush(
					frontier, (euclidean_heuristic(neighbor, goal), neighbor)
				)

	return [], expanded


def a_star(
	graph: nx.DiGraph, start: str, goal: str
) -> Tuple[List[str], List[str]]:
	frontier = [(euclidean_heuristic(start, goal), 0.0, start)]
	came_from: Dict[str, Optional[str]] = {start: None}
	g_score = {start: 0.0}
	expanded: List[str] = []

	while frontier:
		_, current_g, current = heapq.heappop(frontier)
		if current_g > g_score.get(current, math.inf):
			continue

		expanded.append(current)
		if current == goal:
			return reconstruct_path(came_from, goal), expanded

		for neighbor in graph.successors(current):
			edge_cost = graph[current][neighbor]["weight"]
			tentative_g = current_g + edge_cost

			if tentative_g < g_score.get(neighbor, math.inf):
				g_score[neighbor] = tentative_g
				came_from[neighbor] = current
				f_score = tentative_g + euclidean_heuristic(neighbor, goal)
				heapq.heappush(frontier, (f_score, tentative_g, neighbor))

	return [], expanded


def path_cost(graph: nx.DiGraph, path: List[str]) -> float:
	return sum(
		graph[current][next_node]["weight"]
		for current, next_node in zip(path, path[1:])
	)


def draw_graph(graph: nx.DiGraph, path: List[str], start: str, goal: str):
	figure, axis = plt.subplots(figsize=(11, 6))
	positions = HOSPITAL_COORDINATES
	path_edges = list(zip(path, path[1:]))
	path_nodes = set(path)

	node_colors = [
		"#f28e8e" if node == start else
		"#ffd166" if node == goal else
		"#ff8c42" if node in path_nodes else
		"#8ecae6"
		for node in graph.nodes
	]

	nx.draw_networkx_nodes(
		graph,
		positions,
		ax=axis,
		node_color=node_colors,
		node_size=2300,
		edgecolors="#263238",
		linewidths=1.2,
	)
	nx.draw_networkx_labels(
		graph,
		positions,
		ax=axis,
		font_size=9,
		font_weight="bold",
	)
	nx.draw_networkx_edges(
		graph,
		positions,
		ax=axis,
		edge_color="#90a4ae",
		width=1.8,
		arrows=True,
		arrowsize=18,
		connectionstyle="arc3,rad=0.03",
	)
	if path_edges:
		nx.draw_networkx_edges(
			graph,
			positions,
			ax=axis,
			edgelist=path_edges,
			edge_color="#d62828",
			width=4,
			arrows=True,
			arrowsize=22,
			connectionstyle="arc3,rad=0.03",
		)

	edge_labels = {
		edge: f"{data['weight']:.1f}"
		for edge, data in graph.edges.items()
	}
	nx.draw_networkx_edge_labels(
		graph,
		positions,
		ax=axis,
		edge_labels=edge_labels,
		font_size=9,
		label_pos=0.5,
		bbox={"alpha": 0.8, "color": "white", "pad": 0.2},
	)
	axis.set_title("Hospital Emergency Supply Search", fontsize=15, pad=15)
	axis.axis("off")
	figure.tight_layout()
	return figure


def main() -> None:
	st.set_page_config(page_title="Hospital Search Visualization", layout="wide")
	st.title("Emergency Supply Robot: GBFS and A* Search")
	st.write(
		"Select a start location, destination, and informed search algorithm "
		"to visualize the route through the hospital corridors."
	)

	graph = build_hospital_graph()
	nodes = list(HOSPITAL_COORDINATES)
	controls, details = st.columns([1, 2])

	with controls:
		start = st.selectbox("Initial node", nodes, index=nodes.index("Pharmacy"))
		goal = st.selectbox(
			"Goal node", nodes, index=nodes.index("Emergency_Ward")
		)
		algorithm = st.selectbox("Search algorithm", ["GBFS", "A*"])
		run_search = st.button("Run search", type="primary", use_container_width=True)

	if run_search:
		if algorithm == "GBFS":
			path, expanded = gbfs(graph, start, goal)
		else:
			path, expanded = a_star(graph, start, goal)
		st.session_state["search_result"] = {
			"algorithm": algorithm,
			"start": start,
			"goal": goal,
			"path": path,
			"expanded": expanded,
		}

	result = st.session_state.get("search_result")
	if result is None:
		path = []
		expanded = []
		selected_algorithm = algorithm
		selected_start = start
		selected_goal = goal
	else:
		path = result["path"]
		expanded = result["expanded"]
		selected_algorithm = result["algorithm"]
		selected_start = result["start"]
		selected_goal = result["goal"]

	with details:
		st.subheader("NetworkX graph")
		st.pyplot(
			draw_graph(graph, path, selected_start, selected_goal),
			use_container_width=True,
		)

	if result is None:
		st.info("Choose the search settings and select Run search to see a solution.")
		return

	st.subheader("Search result")
	if path:
		metric_columns = st.columns(4)
		metric_columns[0].metric("Algorithm", selected_algorithm)
		metric_columns[1].metric("Total path cost", f"{path_cost(graph, path):.2f}")
		metric_columns[2].metric("Nodes expanded", len(expanded))
		metric_columns[3].metric("Goal reached", path[-1])

		st.write("**Solution path:** " + " -> ".join(path))
		st.write("**Expansion order:** " + " -> ".join(expanded))
		st.info(
			"GBFS ranks nodes using only h(n), so it follows the location that "
			"appears closest to the goal. A* ranks nodes using g(n) + h(n), "
			"balancing the cost already paid with the estimated remaining cost."
		)
	else:
		st.error(f"No path was found from {selected_start} to {selected_goal}.")


if __name__ == "__main__":
	main()
