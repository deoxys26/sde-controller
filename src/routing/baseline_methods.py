"""Routing plugins which use topology and present/past link telemetry only."""

from __future__ import annotations

from typing import Iterable, Mapping

import networkx as nx


class StaticShortestPath:
    """Caches a minimum-hop route per source/destination pair."""

    def __init__(self) -> None:
        self._paths: dict[tuple[object, object], list[object] | None] = {}

    def select_path(self, graph: nx.DiGraph, source: object, destination: object,
                    telemetry: Iterable[Mapping] | None = None) -> list[object] | None:
        key = (source, destination)
        if key not in self._paths:
            try:
                self._paths[key] = nx.shortest_path(graph, source, destination)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                self._paths[key] = None
        path = self._paths[key]
        return list(path) if path is not None else None


class ReactiveRouting:
    """Recomputes a minimum-cost route from supplied current/past telemetry."""

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        supplied = weights or {}
        self.weights = {
            "delay": float(supplied.get("delay", 0.4)),
            "packet_loss": float(supplied.get("packet_loss", 0.3)),
            "utilization": float(supplied.get("utilization", 0.3)),
        }

    @staticmethod
    def _telemetry_by_edge(telemetry: Iterable[Mapping] | None) -> dict[tuple, Mapping]:
        return {(item["source"], item["destination"]): item for item in (telemetry or [])}

    def select_path(self, graph: nx.DiGraph, source: object, destination: object,
                    telemetry: Iterable[Mapping] | None = None) -> list[object] | None:
        records = self._telemetry_by_edge(telemetry)
        max_delay = max((float(r.get("delay", 0.0)) for r in records.values()), default=1.0) or 1.0
        cost_graph = graph.copy()
        for u, v, attrs in cost_graph.edges(data=True):
            record = records.get((u, v), {})
            delay = float(record.get("delay", attrs.get("base_delay_ms", 0.0))) / max_delay
            loss = float(record.get("packet_loss", 0.0))
            utilization = float(record.get("utilization", record.get("bandwidth_utilization", 0.0)))
            attrs["routing_cost"] = (self.weights["delay"] * delay +
                                     self.weights["packet_loss"] * loss +
                                     self.weights["utilization"] * utilization)
        try:
            return nx.shortest_path(cost_graph, source, destination, weight="routing_cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
