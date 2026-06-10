"""Lightweight knowledge graph linking SQLite entities to PDF chunks.

Why a graph in a retrieval POC? Two concrete jobs:

1. ROUTING SIGNAL - if an entity mentioned in the question exists in the
   database *and* is mentioned in documents, that is strong evidence the
   question is hybrid.
2. EVIDENCE JOINING - after one side retrieves, the graph supplies the other
   side's context:
     - SQL found product AF-X200  -> pull the spec / bulletin chunks that
       mention it, even if lexical search ranked them low.
     - Docs found bulletin FSB-2025-03 -> resolve which product it affects so
       the SQL layer can answer "which customers bought it".

Graph shape (networkx MultiDiGraph, persisted as node-link JSON):

    (entity:product/customer/bulletin) <-MENTIONS- (chunk) <-HAS_CHUNK- (doc)
    (entity:customer) -PURCHASED-> (entity:product)        [derived from orders]
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import networkx as nx

from .config import settings

GRAPH_PATH = settings.index_dir / "graph.json"
ALIAS_PATH = settings.index_dir / "aliases.json"

# Hand-curated aliases on top of exact DB names. In production this table is
# fed by an alias service / embedding-based entity linker; for the POC a
# dictionary keeps the behaviour fully inspectable.
EXTRA_ALIASES: dict[str, list[str]] = {
    "product:AF-X200": ["x200", "af-x200", "aeroflow x200"],
    "product:AF-X100": ["x100", "af-x100", "aeroflow x100"],
    "product:HM-P50": ["p50", "hm-p50", "hydromax p50"],
    "product:CC-T8": ["t8", "cc-t8", "climacore t8"],
    "customer:1": ["acme", "acme corp"],
    "customer:2": ["borealis", "borealis labs"],
    "customer:3": ["cascade", "cascade foods"],
    "customer:4": ["delta", "delta logistics"],
    "customer:5": ["evergreen", "evergreen clinics"],
    "bulletin:FSB-2025-03": ["fsb-2025-03", "fsb 2025-03", "service bulletin fsb-2025-03"],
}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _db_entities(db_path: Path) -> dict[str, dict]:
    """Read canonical entities straight from the structured source."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    ents: dict[str, dict] = {}
    for r in conn.execute("SELECT id, sku, name FROM products"):
        ents[f"product:{r['sku']}"] = {
            "type": "product", "name": r["name"], "db_table": "products",
            "db_id": r["id"], "sku": r["sku"],
        }
    for r in conn.execute("SELECT id, name FROM customers"):
        ents[f"customer:{r['id']}"] = {
            "type": "customer", "name": r["name"], "db_table": "customers",
            "db_id": r["id"],
        }
    conn.close()
    # Document-native entities (no DB row, anchored by the doc corpus).
    ents["bulletin:FSB-2025-03"] = {
        "type": "bulletin", "name": "Field Service Bulletin FSB-2025-03",
        "db_table": None, "db_id": None,
    }
    return ents


def build_alias_index(db_path: Path) -> dict[str, str]:
    """alias (lowercase) -> canonical entity id."""
    ents = _db_entities(db_path)
    aliases: dict[str, str] = {}
    for ent_id, attrs in ents.items():
        aliases[attrs["name"].lower()] = ent_id
        if attrs.get("sku"):
            aliases[attrs["sku"].lower()] = ent_id
    for ent_id, extra in EXTRA_ALIASES.items():
        for a in extra:
            aliases[a] = ent_id
    return aliases


def extract_entities(text: str, aliases: dict[str, str]) -> set[str]:
    """Longest-match-first dictionary extraction with word boundaries."""
    found: set[str] = set()
    low = text.lower()
    for alias in sorted(aliases, key=len, reverse=True):
        if re.search(rf"(?<![\w-]){re.escape(alias)}(?![\w-])", low):
            found.add(aliases[alias])
    return found


def build_graph(db_path: Path, chunks: list[dict]) -> nx.MultiDiGraph:
    ents = _db_entities(db_path)
    aliases = build_alias_index(db_path)
    g = nx.MultiDiGraph()

    for ent_id, attrs in ents.items():
        g.add_node(ent_id, kind="entity", **attrs)

    for ch in chunks:
        doc_node = f"doc:{ch['doc']}"
        chunk_node = f"chunk:{ch['chunk_id']}"
        if not g.has_node(doc_node):
            g.add_node(doc_node, kind="doc", name=ch["doc"])
        g.add_node(chunk_node, kind="chunk", doc=ch["doc"], page=ch["page"],
                   section=ch["section"])
        g.add_edge(doc_node, chunk_node, rel="HAS_CHUNK")
        for ent_id in extract_entities(ch.get("index_text", ch["text"]), aliases):
            g.add_edge(chunk_node, ent_id, rel="MENTIONS")

    # Derived structured edges: who purchased what.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute(
        """SELECT DISTINCT o.customer_id, p.sku
           FROM orders o JOIN order_items oi ON oi.order_id = o.id
           JOIN products p ON p.id = oi.product_id""").fetchall()
    conn.close()
    for customer_id, sku in rows:
        g.add_edge(f"customer:{customer_id}", f"product:{sku}", rel="PURCHASED")

    nx.readwrite.json_graph.node_link_data(g)  # validate serialisable
    GRAPH_PATH.write_text(json.dumps(
        nx.readwrite.json_graph.node_link_data(g, edges="links"), indent=1))
    ALIAS_PATH.write_text(json.dumps(aliases, indent=1))
    return g


# ---------------------------------------------------------------------------
# Query-time API
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    def __init__(self) -> None:
        data = json.loads(GRAPH_PATH.read_text())
        self.g: nx.MultiDiGraph = nx.readwrite.json_graph.node_link_graph(
            data, directed=True, multigraph=True, edges="links")
        self.aliases: dict[str, str] = json.loads(ALIAS_PATH.read_text())

    # -- extraction ------------------------------------------------------
    def entities_in(self, text: str) -> set[str]:
        return extract_entities(text, self.aliases)

    def entity(self, ent_id: str) -> dict:
        return dict(self.g.nodes[ent_id]) if self.g.has_node(ent_id) else {}

    # -- linkage questions used by the router ----------------------------
    def has_db_record(self, ent_id: str) -> bool:
        return self.entity(ent_id).get("db_table") is not None

    def doc_mention_count(self, ent_id: str) -> int:
        if not self.g.has_node(ent_id):
            return 0
        return sum(1 for u, _, d in self.g.in_edges(ent_id, data=True)
                   if d.get("rel") == "MENTIONS")

    # -- evidence joining --------------------------------------------------
    def chunks_for_entities(self, ent_ids: set[str]) -> dict[str, set[str]]:
        """entity -> {chunk_id} mentioning it."""
        out: dict[str, set[str]] = {}
        for ent_id in ent_ids:
            if not self.g.has_node(ent_id):
                continue
            chunks = {u.removeprefix("chunk:")
                      for u, _, d in self.g.in_edges(ent_id, data=True)
                      if d.get("rel") == "MENTIONS"}
            if chunks:
                out[ent_id] = chunks
        return out

    def entities_in_chunk(self, chunk_id: str) -> set[str]:
        node = f"chunk:{chunk_id}"
        if not self.g.has_node(node):
            return set()
        return {v for _, v, d in self.g.out_edges(node, data=True)
                if d.get("rel") == "MENTIONS"}

    def co_mentioned(self, ent_id: str, want_type: str) -> set[str]:
        """Entities of `want_type` sharing at least one chunk with `ent_id`.

        This is how "bulletin FSB-2025-03" resolves to "product AF-X200":
        bulletin <-MENTIONS- chunk -MENTIONS-> product.
        """
        result: set[str] = set()
        for chunks in self.chunks_for_entities({ent_id}).values():
            for cid in chunks:
                for other in self.entities_in_chunk(cid):
                    if other != ent_id and self.entity(other).get("type") == want_type:
                        result.add(other)
        return result

    def expand_entities(self, ent_ids: set[str],
                        known: set[str] | None = None) -> dict[str, str]:
        """Resolve doc-native entities (bulletins) to DB entities (products).

        `known` is the set an expansion must be *new relative to* (defaults to
        `ent_ids`). The pipeline passes only the question entities here, so a
        product surfaced via a retrieved bulletin chunk still counts as a hop
        even though the same chunk happens to mention the product.

        Returns {new_entity_id: reason} so the trace can explain the hop.
        """
        known = ent_ids if known is None else known
        added: dict[str, str] = {}
        for ent_id in list(ent_ids):
            if self.entity(ent_id).get("type") == "bulletin":
                for prod in self.co_mentioned(ent_id, "product"):
                    if prod not in known:
                        added[prod] = f"co-mentioned with {ent_id}"
        return added
