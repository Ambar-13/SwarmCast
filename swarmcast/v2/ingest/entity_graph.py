"""
Entity graph for PolicyLab ingestion pipeline.

A lightweight in-memory knowledge graph. No external database, no network calls.
Nodes are regulatory entities (companies, enforcement bodies, systems, thresholds).
Edges are regulatory relationships (must_comply, exempted_from, enforces, triggers).

This is not a full GraphRAG system. The goal is narrower: extract the structural
relationships needed to derive a realistic agent type_distribution for the simulation
and to identify which provisions are mandatory vs optional.

MiroFish uses Neo4j/KuzuDB for this. We use a plain Python dict because:
  1. PolicyLab's queries are simple (no path traversal, no aggregation)
  2. Zero deployment friction — no database server to manage
  3. The graph is small (10-100 nodes per document)
  4. Serialisation to JSON is trivial for audit trails

The graph is populated by spec_builder.py from an ExtractionResult.
"""

from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# NODE TYPES
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_TYPES = {
    # Regulated parties
    "DEVELOPER":         "AI developer (company building or deploying AI systems)",
    "FRONTIER_LAB":      "Top-tier AI developer above compute threshold",
    "SME":               "Small or medium enterprise",
    "STARTUP":           "Early-stage company",
    "RESEARCHER":        "Research institution or academic body",
    "INVESTOR":          "Venture capital or financial institution",
    "CIVIL_SOCIETY":     "NGO, advocacy group, or public body",

    # Regulatory infrastructure
    "ENFORCEMENT_BODY":  "Regulator, agency, or authority responsible for enforcement",
    "EXEMPTED_ENTITY":   "Entity explicitly exempted from the regulation",

    # Regulatory triggers
    "COMPUTE_THRESHOLD": "FLOP-based trigger for regulatory requirements",
    "REVENUE_THRESHOLD": "Revenue-based trigger",
    "EMPLOYEE_THRESHOLD":"Employee count-based trigger",
    "CAPABILITY_TRIGGER":"Capability-based trigger (e.g. dual-use, biometric)",

    # Requirements
    "REQUIREMENT":       "A specific regulatory obligation",
    "PROHIBITION":       "A prohibited action",
    "EXEMPTION":         "An explicit carve-out from a requirement",
}

EDGE_TYPES = {
    "MUST_COMPLY":       "Entity must comply with requirement",
    "EXEMPTED_FROM":     "Entity is exempted from requirement",
    "ENFORCES":          "Body enforces requirement against entity",
    "TRIGGERED_BY":      "Requirement is triggered by threshold crossing",
    "APPLIES_TO":        "Regulation applies to entity type",
    "PENALISES":         "Enforcement action penalises violation",
}


@dataclasses.dataclass
class Node:
    """A node in the entity graph."""
    node_id: str
    node_type: str          # one of ENTITY_TYPES keys
    label: str              # human-readable name
    properties: dict        # arbitrary key-value pairs
    source_passages: list[str]  # document passages that grounded this node
    confidence: float       # [0, 1] how certain we are this node is real

    def __repr__(self) -> str:
        return f"Node({self.node_type}:{self.label!r}, conf={self.confidence:.2f})"


@dataclasses.dataclass
class Edge:
    """A directed edge in the entity graph."""
    edge_id: str
    source_id: str          # node_id of source
    target_id: str          # node_id of target
    edge_type: str          # one of EDGE_TYPES keys
    label: str              # human-readable description
    source_passages: list[str]
    confidence: float


class EntityGraph:
    """In-memory entity/relationship graph for a single regulatory document.

    Designed for the specific queries PolicyLab needs:
      - Which entity types are regulated? → type_distribution
      - What are the key thresholds? → compute_cost_factor, severity
      - Who enforces? → enforcement_mechanism
      - Which entities are exempted? → type_distribution adjustment

    Not designed for: arbitrary graph traversal, multi-hop queries,
    temporal reasoning, or cross-document merging.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}
        self._out_edges: dict[str, list[str]] = defaultdict(list)  # source → [edge_ids]
        self._in_edges: dict[str, list[str]] = defaultdict(list)   # target → [edge_ids]
        self._type_index: dict[str, list[str]] = defaultdict(list) # type → [node_ids]

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add_node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        properties: dict | None = None,
        source_passages: list[str] | None = None,
        confidence: float = 0.5,
    ) -> Node:
        """Add a node to the graph. Returns the node (creates or updates)."""
        if node_type not in ENTITY_TYPES:
            raise ValueError(f"Unknown node type: {node_type!r}. Valid: {list(ENTITY_TYPES)}")

        node = Node(
            node_id=node_id,
            node_type=node_type,
            label=label,
            properties=properties or {},
            source_passages=source_passages or [],
            confidence=confidence,
        )
        self._nodes[node_id] = node
        if node_id not in self._type_index[node_type]:
            self._type_index[node_type].append(node_id)
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        label: str = "",
        source_passages: list[str] | None = None,
        confidence: float = 0.5,
    ) -> Edge | None:
        """Add a directed edge. Returns None if either node doesn't exist."""
        if source_id not in self._nodes or target_id not in self._nodes:
            return None
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"Unknown edge type: {edge_type!r}. Valid: {list(EDGE_TYPES)}")

        edge_id = f"{source_id}_{edge_type}_{target_id}"
        edge = Edge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            label=label or edge_type.replace("_", " ").lower(),
            source_passages=source_passages or [],
            confidence=confidence,
        )
        self._edges[edge_id] = edge
        if edge_id not in self._out_edges[source_id]:
            self._out_edges[source_id].append(edge_id)
        if edge_id not in self._in_edges[target_id]:
            self._in_edges[target_id].append(edge_id)
        return edge

    # ── Query ─────────────────────────────────────────────────────────────────

    def nodes_of_type(self, node_type: str) -> list[Node]:
        return [self._nodes[nid] for nid in self._type_index.get(node_type, [])
                if nid in self._nodes]

    def regulated_entity_types(self, min_confidence: float = 0.40) -> list[str]:
        """Return PolicyLab agent type strings for entities that must comply.

        Maps graph node types to PolicyLab _TYPES vocabulary.
        """
        type_map = {
            "FRONTIER_LAB":  "frontier_lab",
            "DEVELOPER":     "large_company",
            "SME":           "mid_company",
            "STARTUP":       "startup",
            "RESEARCHER":    "researcher",
            "INVESTOR":      "investor",
            "CIVIL_SOCIETY": "civil_society",
        }
        result = []
        for node_type, policylab_type in type_map.items():
            nodes = self.nodes_of_type(node_type)
            if any(n.confidence >= min_confidence for n in nodes):
                if policylab_type not in result:
                    result.append(policylab_type)
        # mid_company is inferred from DEVELOPER presence: every ecosystem that
        # has large companies also has mid-size companies between them and startups.
        # Without this, mid_company never appears when no SME node was extracted.
        developer_nodes = self.nodes_of_type("DEVELOPER")
        if any(n.confidence >= min_confidence for n in developer_nodes):
            if "mid_company" not in result:
                result.append("mid_company")
        return result

    def exempted_entity_types(self, min_confidence: float = 0.50) -> list[str]:
        """Return agent types that are explicitly exempted from compliance."""
        exempted = set()
        for edge in self._edges.values():
            if edge.edge_type == "EXEMPTED_FROM" and edge.confidence >= min_confidence:
                node = self._nodes.get(edge.source_id)
                if node and node.node_type == "RESEARCHER":
                    exempted.add("researcher")
                elif node and node.node_type in ("SME", "STARTUP"):
                    exempted.add("startup")
        return list(exempted)

    def thresholds(self) -> list[Node]:
        """Return all threshold nodes (compute, revenue, employee count)."""
        result = []
        for t in ["COMPUTE_THRESHOLD", "REVENUE_THRESHOLD", "EMPLOYEE_THRESHOLD"]:
            result.extend(self.nodes_of_type(t))
        return result

    def enforcement_bodies(self) -> list[Node]:
        return self.nodes_of_type("ENFORCEMENT_BODY")

    def requirements(self, min_confidence: float = 0.40) -> list[Node]:
        return [n for n in self.nodes_of_type("REQUIREMENT")
                if n.confidence >= min_confidence]

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict for audit trails and the evidence pack."""
        return {
            "nodes": [
                {
                    "id": n.node_id, "type": n.node_type, "label": n.label,
                    "confidence": n.confidence, "properties": n.properties,
                    "source_passages": n.source_passages[:2],
                }
                for n in self._nodes.values()
            ],
            "edges": [
                {
                    "id": e.edge_id, "source": e.source_id, "target": e.target_id,
                    "type": e.edge_type, "label": e.label, "confidence": e.confidence,
                }
                for e in self._edges.values()
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def summary(self) -> str:
        """Human-readable summary for console output."""
        lines = [f"EntityGraph: {len(self._nodes)} nodes, {len(self._edges)} edges"]
        for nt in ENTITY_TYPES:
            nodes = self.nodes_of_type(nt)
            if nodes:
                labels = ", ".join(n.label[:30] for n in nodes[:3])
                lines.append(f"  {nt:<20}: {len(nodes)} ({labels})")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._nodes)


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH BUILDER — from ExtractionResult
# ─────────────────────────────────────────────────────────────────────────────

def build_entity_graph(extraction) -> EntityGraph:
    """Build an EntityGraph from a provision_extractor.ExtractionResult.

    This is the bridge between the raw extraction and the structured graph.
    Nodes are created for every regulatory entity mentioned; edges connect
    entities to their obligations, exemptions, and enforcement relationships.
    """
    g = EntityGraph()

    # ── Core regulation node ─────────────────────────────────────────────────
    pol_name = str(extraction.policy_name.value)
    g.add_node("regulation", "REQUIREMENT",
               label=pol_name,
               properties={"type": "primary_regulation"},
               confidence=extraction.policy_name.confidence)

    # ── Compute threshold node ───────────────────────────────────────────────
    thresh = extraction.compute_threshold_flops.value
    if thresh is not None:
        try:
            thresh_f = float(thresh)
            exp = int(round(__import__("math").log10(thresh_f)))
            thresh_label = f"10^{exp} FLOPS"
        except (ValueError, TypeError):
            thresh_label = str(thresh)

        g.add_node("compute_threshold", "COMPUTE_THRESHOLD",
                   label=thresh_label,
                   properties={"flops": thresh},
                   source_passages=[extraction.compute_threshold_flops.source_passage],
                   confidence=extraction.compute_threshold_flops.confidence)

        g.add_edge("regulation", "compute_threshold", "TRIGGERED_BY",
                   label=f"triggered when training exceeds {thresh_label}",
                   confidence=extraction.compute_threshold_flops.confidence)

    # ── Penalty node ─────────────────────────────────────────────────────────
    if extraction.penalty_type.value not in ("none", "voluntary"):
        cap = extraction.penalty_cap_usd.value
        cap_str = f"${cap/1e6:.0f}M" if cap else "% of turnover / uncapped"
        g.add_node("penalty", "REQUIREMENT",
                   label=f"{extraction.penalty_type.value} penalty ({cap_str})",
                   properties={
                       "penalty_type": extraction.penalty_type.value,
                       "cap_usd": cap,
                   },
                   source_passages=[extraction.penalty_type.source_passage],
                   confidence=extraction.penalty_type.confidence)
        g.add_edge("regulation", "penalty", "APPLIES_TO",
                   confidence=extraction.penalty_type.confidence)

    # ── Enforcement body ─────────────────────────────────────────────────────
    enf_mech = extraction.enforcement_mechanism.value
    g.add_node("enforcement_body", "ENFORCEMENT_BODY",
               label=f"Enforcement: {enf_mech.replace('_', ' ')}",
               properties={"mechanism": enf_mech},
               source_passages=[extraction.enforcement_mechanism.source_passage],
               confidence=extraction.enforcement_mechanism.confidence)
    g.add_edge("enforcement_body", "regulation", "ENFORCES",
               confidence=extraction.enforcement_mechanism.confidence)

    # ── Regulated entity nodes ───────────────────────────────────────────────
    scope = extraction.scope.value

    # Always add frontier_lab if compute threshold exists
    if thresh is not None or extraction.has_frontier_lab_focus.value:
        g.add_node("frontier_lab", "FRONTIER_LAB",
                   label="Frontier AI developer (above compute threshold)",
                   properties={"scope": scope},
                   source_passages=[extraction.has_frontier_lab_focus.source_passage],
                   confidence=max(
                       extraction.has_frontier_lab_focus.confidence,
                       extraction.compute_threshold_flops.confidence * 0.8,
                   ))
        g.add_edge("frontier_lab", "regulation", "MUST_COMPLY",
                   source_passages=[extraction.scope.source_passage],
                   confidence=extraction.has_frontier_lab_focus.confidence)

    # Large developer (always included unless scope is voluntary)
    if scope != "voluntary":
        developer_conf = {
            "frontier_only": 0.50,
            "large_developers_only": 0.85,
            "all": 0.90,
        }.get(scope, 0.60)
        g.add_node("large_developer", "DEVELOPER",
                   label="Large AI developer",
                   source_passages=[extraction.scope.source_passage],
                   confidence=developer_conf)
        g.add_edge("large_developer", "regulation", "MUST_COMPLY",
                   confidence=developer_conf)

    # SME / startup
    if extraction.has_sme_provisions.value:
        g.add_node("sme", "SME",
                   label="Small/medium AI developer",
                   source_passages=[extraction.has_sme_provisions.source_passage],
                   confidence=extraction.has_sme_provisions.confidence)
        if scope == "all":
            g.add_edge("sme", "regulation", "MUST_COMPLY",
                       confidence=extraction.has_sme_provisions.confidence * 0.8)
        elif scope in ("frontier_only", "large_developers_only"):
            g.add_edge("sme", "regulation", "EXEMPTED_FROM",
                       label="SMEs below threshold are exempt",
                       confidence=extraction.has_sme_provisions.confidence * 0.7)

    # Research institutions
    if extraction.has_research_exemptions.value:
        g.add_node("researcher", "RESEARCHER",
                   label="Research institution",
                   source_passages=[extraction.has_research_exemptions.source_passage],
                   confidence=extraction.has_research_exemptions.confidence)
        # Research is usually exempted
        g.add_edge("researcher", "regulation", "EXEMPTED_FROM",
                   label="Research exemption",
                   confidence=extraction.has_research_exemptions.confidence)

    # Investors
    if extraction.has_investor_provisions.value:
        g.add_node("investor", "INVESTOR",
                   label="AI investor / VC",
                   source_passages=[extraction.has_investor_provisions.source_passage],
                   confidence=extraction.has_investor_provisions.confidence)
        # Investors are not usually primary compliance targets
        g.add_node("investor_req", "REQUIREMENT",
                   label="Investor disclosure / due diligence",
                   confidence=extraction.has_investor_provisions.confidence * 0.7)
        g.add_edge("investor", "investor_req", "MUST_COMPLY",
                   confidence=extraction.has_investor_provisions.confidence * 0.7)

    # ── Key provision nodes ──────────────────────────────────────────────────
    for i, (text, source_id) in enumerate(extraction.key_provisions[:8]):
        nid = f"provision_{i}"
        g.add_node(nid, "REQUIREMENT",
                   label=text[:60] + ("..." if len(text) > 60 else ""),
                   properties={"full_text": text, "source_id": source_id},
                   confidence=0.75)
        g.add_edge("regulation", nid, "APPLIES_TO", confidence=0.75)

    return g
