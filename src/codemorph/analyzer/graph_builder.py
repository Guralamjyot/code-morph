"""
Dependency graph builder.

Analyzes code fragments and builds a directed acyclic graph (DAG) of dependencies.
Performs topological sorting to determine translation order.
"""

from pathlib import Path
from typing import Any

import networkx as nx

from codemorph.config.models import AnalysisResult, CodeFragment


class DependencyGraphBuilder:
    """Builds and analyzes dependency graphs for code fragments."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph(self, fragments: dict[str, CodeFragment]) -> nx.DiGraph:
        """
        Build a dependency graph from code fragments.

        Args:
            fragments: Dictionary of fragment_id -> CodeFragment

        Returns:
            Directed graph where edges represent dependencies
        """
        self.graph.clear()

        # Add all fragments as nodes
        for fragment_id, fragment in fragments.items():
            self.graph.add_node(
                fragment_id,
                fragment=fragment,
                type=fragment.fragment_type,
                name=fragment.name,
            )

        # Add edges based on dependencies
        for fragment_id, fragment in fragments.items():
            for dep_id in fragment.dependencies:
                if dep_id in fragments:
                    # Add edge: fragment depends on dep
                    self.graph.add_edge(dep_id, fragment_id)

        return self.graph

    def get_translation_order(self) -> list[str]:
        """
        Get the translation order using topological sort.

        Returns:
            List of fragment IDs in dependency order (dependencies first)

        Raises:
            ValueError: If the graph contains cycles
        """
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Dependency graph contains cycles. Cannot determine order.")

        # Topological sort: dependencies come before dependents
        return list(nx.topological_sort(self.graph))

    def detect_circular_dependencies(self) -> list[list[str]]:
        """
        Detect circular dependencies in the graph.

        Returns:
            List of cycles, where each cycle is a list of fragment IDs
        """
        if nx.is_directed_acyclic_graph(self.graph):
            return []

        # Find all simple cycles
        try:
            cycles = list(nx.simple_cycles(self.graph))
            return cycles
        except:
            return []

    def get_strongly_connected_components(self) -> list[list[str]]:
        """
        Get strongly connected components (groups of mutually dependent fragments).

        Useful for identifying groups that need interface extraction.

        Returns:
            List of components, each containing fragment IDs
        """
        components = list(nx.strongly_connected_components(self.graph))
        # Filter out single-node components (no circular dependencies)
        return [list(comp) for comp in components if len(comp) > 1]

    def calculate_complexity_scores(self, fragments: dict[str, CodeFragment]) -> dict[str, int]:
        """
        Calculate complexity scores for fragments.

        Complexity is based on:
        - Number of lines of code
        - Number of dependencies
        - Nesting depth (approximated)

        Args:
            fragments: Dictionary of fragments

        Returns:
            Dictionary of fragment_id -> complexity_score
        """
        scores = {}

        for fragment_id, fragment in fragments.items():
            # Base score: lines of code
            loc = fragment.end_line - fragment.start_line + 1

            # Add score for dependencies
            num_deps = len(fragment.dependencies)

            # Add score for indentation (proxy for nesting)
            avg_indent = self._calculate_avg_indentation(fragment.source_code)

            complexity = loc + (num_deps * 5) + (avg_indent * 2)
            scores[fragment_id] = complexity

        return scores

    def _calculate_avg_indentation(self, code: str) -> int:
        """Calculate average indentation level (proxy for complexity)."""
        lines = code.split("\n")
        total_indent = 0
        counted_lines = 0

        for line in lines:
            stripped = line.lstrip()
            if stripped and not stripped.startswith("#") and not stripped.startswith("//"):
                indent = len(line) - len(stripped)
                total_indent += indent
                counted_lines += 1

        return total_indent // max(counted_lines, 1)

    def get_dependencies_of(self, fragment_id: str) -> list[str]:
        """Get all fragments that a given fragment depends on."""
        if fragment_id not in self.graph:
            return []
        return list(self.graph.predecessors(fragment_id))

    def get_dependents_of(self, fragment_id: str) -> list[str]:
        """Get all fragments that depend on a given fragment."""
        if fragment_id not in self.graph:
            return []
        return list(self.graph.successors(fragment_id))

    def save_graph(self, output_path: Path):
        """Save the graph to a file (as GraphML)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Create a copy with only serializable attributes
        serializable_graph = nx.DiGraph()
        for node, data in self.graph.nodes(data=True):
            # Only include serializable attributes
            safe_data = {
                "name": data.get("name", ""),
                "type": str(data.get("type", "")),
            }
            serializable_graph.add_node(node, **safe_data)
        for u, v, data in self.graph.edges(data=True):
            serializable_graph.add_edge(u, v)
        nx.write_graphml(serializable_graph, str(output_path))

    def load_graph(self, input_path: Path):
        """Load a graph from a file."""
        self.graph = nx.read_graphml(str(input_path))

    def create_analysis_result(
        self, fragments: dict[str, CodeFragment], graph_path: Path | None = None
    ) -> AnalysisResult:
        """
        Create a comprehensive analysis result.

        Args:
            fragments: All extracted fragments
            graph_path: Optional path where graph was saved

        Returns:
            AnalysisResult object with all analysis data
        """
        # Build the graph
        self.build_graph(fragments)

        # Get translation order (or fallback to all fragments if there are cycles)
        try:
            translation_order = self.get_translation_order()
        except ValueError:
            # Fallback: use all fragments in arbitrary order when cycles exist
            # This allows translation to proceed, though some dependencies may not be ready
            translation_order = list(fragments.keys())

        # Detect circular dependencies
        circular_deps = self.detect_circular_dependencies()

        # Calculate complexity scores
        complexity_scores = self.calculate_complexity_scores(fragments)

        return AnalysisResult(
            fragments=fragments,
            translation_order=translation_order,
            circular_dependencies=circular_deps,
            dependency_graph_path=graph_path,
            complexity_scores=complexity_scores,
        )

    def visualize_graph(self, output_path: Path, format: str = "png"):
        """
        Visualize the dependency graph (requires graphviz).

        Args:
            output_path: Path for the output image
            format: Image format (png, svg, pdf, etc.)
        """
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(self.graph)

            # Draw nodes
            nx.draw_networkx_nodes(
                self.graph, pos, node_color="lightblue", node_size=500, alpha=0.9
            )

            # Draw edges
            nx.draw_networkx_edges(
                self.graph, pos, edge_color="gray", arrows=True, arrowsize=20
            )

            # Draw labels (abbreviated)
            labels = {
                node: data.get("name", node)[:15] for node, data in self.graph.nodes(data=True)
            }
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)

            plt.title("Code Fragment Dependency Graph")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(output_path, format=format, dpi=300)
            plt.close()

        except ImportError:
            # Fallback: save as GraphML
            self.save_graph(output_path.with_suffix(".graphml"))
            print("Matplotlib not available. Saved as GraphML instead.")
