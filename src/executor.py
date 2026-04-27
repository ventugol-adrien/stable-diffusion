from heapq import heappop, heappush, heapify
from collections import defaultdict
from src.nodes.base_node import BaseNode


def execute_dag(dag: dict[str, BaseNode], context: dict):
    """
    Executes the given DAG respecting dependencies and context propagation.

    Args:
        dag: Dictionary mapping node ids to BaseNode instances
        context: Dictionary containing execution context (inputs/outputs from previous nodes)

    Returns:
        The result of the terminal node(s) in the DAG
    """
    # Build in-degree map and adjacency list
    in_degree = defaultdict(int)
    graph = defaultdict(list)

    for node_id, node in dag.items():
        if node_id not in in_degree:
            in_degree[node_id] = 0
        for dep_id in node.dependencies:
            if dep_id in dag:
                graph[dep_id].append(node_id)
                in_degree[node_id] += 1

    # Initialize with source nodes (no dependencies)
    source_heap = [node_id for node_id, degree in in_degree.items() if degree == 0]
    heapify(source_heap)

    results = {}

    while source_heap:
        node_id = heappop(source_heap)
        current_node = dag[node_id]

        # Execute node with current context
        result = current_node(**current_node.params, **context)
        results[node_id] = result

        # Propagate results to dependent nodes
        for dependent_id in graph[node_id]:
            if dependent_id not in results:
                context[dependent_id] = result.model_dump()
            heappush(source_heap, dependent_id)

    # Return results from terminal nodes
    terminal_results = [
        results[node_id] for node_id in results if not dag[node_id].next_nodes
    ]

    if terminal_results:
        return terminal_results[0]
    return None
