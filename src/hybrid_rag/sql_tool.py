"""Structured retrieval over SQLite with provenance.

Layering (defence in depth — all three always apply):

1. SEMANTIC LAYER (default): a small library of parameterised, reviewed query
   templates keyed on intent + linked entities. This mirrors how mature teams
   ship text-to-SQL safely (approved queries, not free-form generation).
2. LLM NL->SQL (optional): when an API key is present and no template
   matches, the LLM drafts SQL — which still has to pass the gate below.
3. VALIDATION GATE + READ-ONLY EXECUTION: single statement, SELECT/WITH only,
   keyword denylist, hard LIMIT, and a `mode=ro` connection so even a missed
   pattern cannot mutate state.

Every result becomes an `Evidence` whose metadata carries the exact SQL, the
row count and the primary keys involved — that is the database citation.
"""
from __future__ import annotations

import datetime as dt
import re
import sqlite3
from dataclasses import dataclass, field

from . import llm
from .config import settings
from .kg import KnowledgeGraph
from .models import Evidence

BANNED = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|"
    r"replace|reindex|trigger)\b", re.IGNORECASE)

SCHEMA_DOC = """
customers(id, name, segment, region)
products(id, sku, name, category, unit_price, warranty_years)
orders(id, customer_id, order_date, status)
order_items(id, order_id, product_id, quantity, unit_price)
support_tickets(id, customer_id, product_id, opened_date, issue_type, status, summary)
""".strip()


# ---------------------------------------------------------------------------
# Validation + execution
# ---------------------------------------------------------------------------

class SqlValidationError(ValueError):
    pass


def _ast_validate(sql: str) -> None:
    """Parse with sqlglot and reject anything that isn't a single read query.

    Defence-in-depth on top of the keyword denylist: an AST catches statement
    types and injection shapes a regex can miss. No-op if sqlglot is absent.
    """
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError:
        return
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except Exception as exc:                       # unparseable -> reject
        raise SqlValidationError(f"could not parse SQL ({exc})")
    if len(statements) != 1:
        raise SqlValidationError("exactly one statement is allowed")
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create,
                 exp.Alter, exp.Command, exp.Set)
    bad = next(statements[0].find_all(*forbidden), None)
    if bad is not None:
        raise SqlValidationError(f"disallowed operation: {type(bad).__name__}")


def validate_sql(sql: str) -> str:
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        raise SqlValidationError("multiple statements are not allowed")
    first = s.split(None, 1)[0].lower() if s else ""
    if first not in {"select", "with"}:
        raise SqlValidationError("only SELECT/WITH statements are allowed")
    if BANNED.search(s):
        raise SqlValidationError("statement contains a banned keyword")
    _ast_validate(s)                              # sqlglot AST safety check
    if not re.search(r"\blimit\s+\d+\b", s, re.IGNORECASE):
        s = f"{s} LIMIT {settings.sql_row_limit}"
    return s


def execute(sql: str, params: dict | None = None) -> tuple[list[str], list[dict]]:
    safe_sql = validate_sql(sql)
    conn = sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(safe_sql, params or {})
        rows = [dict(r) for r in cur.fetchmany(settings.sql_row_limit)]
        cols = [d[0] for d in cur.description] if cur.description else []
        return cols, rows
    finally:
        conn.close()


def render_rows(cols: list[str], rows: list[dict]) -> str:
    if not rows:
        return "(no rows)"
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    head = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    body = "\n".join(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols)
                     for r in rows)
    return f"{head}\n{sep}\n{body}"


# ---------------------------------------------------------------------------
# Semantic layer (deterministic templates)
# ---------------------------------------------------------------------------

@dataclass
class SqlPlan:
    kind: str
    title: str
    sql: str
    params: dict = field(default_factory=dict)
    engine: str = "template"


def _date_window(question: str) -> tuple[str | None, str | None, str]:
    q = question.lower()
    if m := re.search(r"\bsince\s+(20\d{2})\b", q):
        return f"{m.group(1)}-01-01", None, f"since {m.group(1)}"
    if m := re.search(r"\bin\s+(20\d{2})\b", q):
        y = m.group(1)
        return f"{y}-01-01", f"{y}-12-31", f"in {y}"
    if "last year" in q:
        y = dt.date.today().year - 1
        return f"{y}-01-01", f"{y}-12-31", f"in {y}"
    return None, None, ""


def _pick(graph: KnowledgeGraph, entities: set[str], etype: str) -> dict | None:
    for ent in sorted(entities):
        attrs = graph.entity(ent)
        if attrs.get("type") == etype:
            return attrs
    return None


def _order_filters(cust, prod, date_from, date_to) -> tuple[str, dict]:
    where, params = ["1=1"], {}
    if cust:
        where.append("c.id = :cust_id")
        params["cust_id"] = cust["db_id"]
    if prod:
        where.append("p.id = :prod_id")
        params["prod_id"] = prod["db_id"]
    if date_from:
        where.append("o.order_date >= :date_from")
        params["date_from"] = date_from
    if date_to:
        where.append("o.order_date <= :date_to")
        params["date_to"] = date_to
    return " AND ".join(where), params


def plan_queries(question: str, entities: set[str], graph: KnowledgeGraph) -> list[SqlPlan]:
    q = question.lower()
    cust = _pick(graph, entities, "customer")
    prod = _pick(graph, entities, "product")
    date_from, date_to, date_label = _date_window(question)
    plans: list[SqlPlan] = []

    wants_units = (re.search(r"\bhow many\b|\bunits?\b|\bquantity\b", q)
                   and re.search(r"\bpurchas|\bbought\b|\border", q))
    wants_value = re.search(r"\bvalue\b|\brevenue\b|\bspend\b|\bworth\b", q)
    wants_customers = re.search(r"\bwhich customers\b|\bwho (bought|purchased)\b", q)
    wants_tickets = re.search(r"\bticket|\bissue|\bcomplaint|\breported\b|\bfailure", q)

    if wants_units and (cust or prod) and not wants_customers:
        cond, params = _order_filters(cust, prod, date_from, date_to)
        plans.append(SqlPlan(
            kind="units_purchased",
            title="orders ⋈ order_items ⋈ customers ⋈ products (unit totals)",
            sql=f"""SELECT c.name AS customer, p.name AS product,
       SUM(oi.quantity) AS total_units,
       COUNT(DISTINCT o.id) AS num_orders,
       MIN(o.order_date) AS first_order, MAX(o.order_date) AS last_order,
       GROUP_CONCAT(DISTINCT o.id) AS order_ids
FROM orders o
JOIN order_items oi ON oi.order_id = o.id
JOIN customers c ON c.id = o.customer_id
JOIN products p ON p.id = oi.product_id
WHERE {cond}
GROUP BY c.id, p.id
ORDER BY total_units DESC""",
            params=params | {"_date_label": date_label}))

    if wants_value and (cust or prod):
        cond, params = _order_filters(cust, prod, date_from, date_to)
        plans.append(SqlPlan(
            kind="order_value",
            title="orders ⋈ order_items (order value)",
            sql=f"""SELECT c.name AS customer,
       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_value,
       COUNT(DISTINCT o.id) AS num_orders,
       GROUP_CONCAT(DISTINCT o.id) AS order_ids
FROM orders o
JOIN order_items oi ON oi.order_id = o.id
JOIN customers c ON c.id = o.customer_id
JOIN products p ON p.id = oi.product_id
WHERE {cond}
GROUP BY c.id""",
            params=params | {"_date_label": date_label}))

    if wants_customers and prod:
        cond, params = _order_filters(None, prod, date_from, date_to)
        plans.append(SqlPlan(
            kind="customers_for_product",
            title=f"customers who purchased {prod['name']}",
            sql=f"""SELECT c.name AS customer, SUM(oi.quantity) AS total_units,
       COUNT(DISTINCT o.id) AS num_orders,
       GROUP_CONCAT(DISTINCT o.id) AS order_ids
FROM orders o
JOIN order_items oi ON oi.order_id = o.id
JOIN customers c ON c.id = o.customer_id
JOIN products p ON p.id = oi.product_id
WHERE {cond}
GROUP BY c.id
ORDER BY total_units DESC""",
            params=params | {"_product": prod["name"]}))

    if wants_tickets and (cust or prod):
        cond, params = ["1=1"], {}
        if cust:
            cond.append("t.customer_id = :cust_id")
            params["cust_id"] = cust["db_id"]
        if prod:
            cond.append("t.product_id = :prod_id")
            params["prod_id"] = prod["db_id"]
        for kw in ("compressor", "leak", "noise", "display"):
            if kw in q:
                cond.append("(t.issue_type LIKE :kw OR t.summary LIKE :kw)")
                params["kw"] = f"%{kw}%"
                break
        plans.append(SqlPlan(
            kind="support_tickets",
            title="support_tickets (matching issues)",
            sql=f"""SELECT t.id AS ticket_id, c.name AS customer, p.name AS product,
       t.opened_date, t.issue_type, t.status, t.summary
FROM support_tickets t
JOIN customers c ON c.id = t.customer_id
JOIN products p ON p.id = t.product_id
WHERE {' AND '.join(cond)}
ORDER BY t.opened_date""",
            params=params))

    return plans


# ---------------------------------------------------------------------------
# Optional LLM NL->SQL (still passes the gate)
# ---------------------------------------------------------------------------

_NL2SQL_SYSTEM = f"""You translate a question into a single SQLite SELECT.
Schema:
{SCHEMA_DOC}
Rules:
- One statement, SELECT/WITH only; add LIMIT 50.
- When the question asks for a total/sum/count/average or "how many", use the
  matching aggregate (SUM / COUNT / AVG) and return ONE summary row — never the
  individual rows.
- Order value = SUM(order_items.quantity * order_items.unit_price).
- Filter by id when ids are provided; otherwise join and filter by name.
- Apply ONLY the filters the question asks for; never invent others (e.g. do
  not filter on order status). Use a year range on order_date for "in <year>".
- Include identifying key columns for provenance, e.g.
  GROUP_CONCAT(DISTINCT o.id) AS order_ids (or ticket ids).
Reply ONLY with JSON: {{"sql": "...", "title": "<short description>"}}"""


def _value_domains() -> str:
    """Distinct values of low-cardinality text columns, so the model uses real
    filter values instead of inventing them (e.g. status='completed')."""
    lines: list[str] = []
    try:
        conn = sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        for t in tables:
            for col in conn.execute(f'PRAGMA table_info("{t}")').fetchall():
                if not str(col["type"]).upper().startswith("TEXT"):
                    continue
                vals = [str(r[0]) for r in conn.execute(
                    f'SELECT DISTINCT "{col["name"]}" FROM "{t}" '
                    f'WHERE "{col["name"]}" IS NOT NULL LIMIT 20')]
                if vals and len(vals) <= 12 and max(len(v) for v in vals) <= 32:
                    lines.append(f"- {t}.{col['name']}: " + ", ".join(sorted(vals)))
        conn.close()
    except sqlite3.Error:
        return ""
    return ("\nKnown categorical column values (use EXACT values; do not invent):\n"
            + "\n".join(lines)) if lines else ""


def _entity_hint(entities: set[str] | None, graph: KnowledgeGraph | None) -> str:
    """Map resolved entities to their primary keys so the SQL filters by id."""
    if not (entities and graph):
        return ""
    named = []
    for ent in sorted(entities):
        attrs = graph.entity(ent)
        if attrs.get("db_id") is not None:
            named.append(f"{attrs.get('type')} '{attrs.get('name')}' has id {attrs['db_id']}")
    return "\nKnown entities: " + "; ".join(named) + "." if named else ""


def _nl2sql_system() -> str:
    return _NL2SQL_SYSTEM + _value_domains()


def llm_plan(question: str, entities: set[str] | None = None,
             graph: KnowledgeGraph | None = None) -> SqlPlan | None:
    out = llm.complete_json(_nl2sql_system(),
                            question + _entity_hint(entities, graph), temperature=0)
    if not out or not out.get("sql"):
        return None
    try:
        validate_sql(out["sql"])
    except SqlValidationError:
        return None
    return SqlPlan(kind="llm_sql", title=out.get("title", "LLM-generated query"),
                   sql=out["sql"], engine="llm")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _all_null(rows: list[dict]) -> bool:
    """True when every cell is NULL — e.g. an aggregate over zero matched rows."""
    return bool(rows) and all(all(v is None for v in r.values()) for r in rows)


def _sql_evidence(plan: SqlPlan, cols: list[str], rows: list[dict],
                  params: dict) -> Evidence:
    id_col = next((c for c in ("order_ids", "ticket_id", "id") if c in cols), None)
    row_ids = ",".join(sorted({str(r[id_col]) for r in rows})) if id_col and rows else ""
    return Evidence(
        source_type="sqlite",
        locator=f"db://app.db?kind={plan.kind}",
        title=plan.title,
        content=render_rows(cols, rows),
        score=1.0 if rows else 0.1,
        metadata={"kind": plan.kind, "engine": plan.engine,
                  "sql": re.sub(r"\s+", " ", plan.sql).strip(),
                  "params": params, "row_count": len(rows),
                  "row_ids": row_ids, "rows": rows[:10],
                  "date_label": plan.params.get("_date_label", ""),
                  "product": plan.params.get("_product", "")},
    )


def _llm_sql_attempts(question: str, entities: set[str], graph: KnowledgeGraph,
                      trace: dict):
    """Generate -> safety-gate -> execute -> sanity-check, retrying with the
    error/empty-result fed back so the model rewrites its own SQL.

    Returns (plan, cols, rows) of the last query that executed (preferring a
    non-empty result), or None if nothing ever ran. Every attempt passes the
    full safety gate; retries never relax it.
    """
    system = _nl2sql_system()
    hint = _entity_hint(entities, graph)
    feedback = ""
    attempts: list[dict] = []
    last_exec = None
    for i in range(1, settings.sql_max_retries + 1):
        out = llm.complete_json(system, f"{question}{hint}{feedback}", temperature=0)
        sql = (out or {}).get("sql", "")
        title = (out or {}).get("title", "LLM-generated query")
        if not sql:
            attempts.append({"try": i, "error": "no SQL returned"})
            feedback = '\n\nYour previous reply had no SQL. Reply with JSON {"sql": "..."}.'
            continue
        try:
            safe = validate_sql(sql)
        except SqlValidationError as exc:
            attempts.append({"try": i, "sql": sql, "error": f"validation: {exc}"})
            feedback = (f"\n\nYour previous SQL:\n{sql}\nwas REJECTED by the safety gate "
                        f"({exc}). Rewrite it as a single read-only SELECT.")
            continue
        try:
            cols, rows = execute(safe)
        except sqlite3.Error as exc:
            attempts.append({"try": i, "sql": sql, "error": f"db: {exc}"})
            feedback = (f"\n\nYour previous SQL:\n{sql}\nFAILED to run ({exc}). The only "
                        f"tables/columns are:\n{SCHEMA_DOC}\nRewrite it.")
            continue
        plan = SqlPlan(kind="llm_sql", title=title, sql=safe, engine="llm")
        last_exec = (plan, cols, rows)
        ok = bool(rows) and not _all_null(rows)
        attempts.append({"try": i, "sql": re.sub(r"\s+", " ", safe).strip(),
                         "rows": len(rows), "ok": ok})
        if ok:
            break
        feedback = (f"\n\nYour previous SQL:\n{sql}\nran but returned no usable rows. "
                    f"You may be over-filtering or using a value not present in the data. "
                    f"Reconsider the filters and rewrite.")
    trace.setdefault("sql_attempts", []).extend(attempts)
    return last_exec


def retrieve(question: str, entities: set[str], graph: KnowledgeGraph,
             trace: dict, force_llm: bool = False) -> list[Evidence]:
    if force_llm:
        # LLM-only mode: the model writes the SQL every time, inside a bounded
        # self-correction loop. Every attempt still passes the safety gate.
        result = _llm_sql_attempts(question, entities, graph, trace)
        trace["sql_plans"] = [{"kind": "llm_sql", "engine": "llm"}] if result else []
        if result is None:
            trace.setdefault("sql_errors", []).append(
                "LLM-only NL->SQL produced no runnable query after retries")
            return []
        plan, cols, rows = result
        return [_sql_evidence(plan, cols, rows, {})]

    plans = plan_queries(question, entities, graph)
    if not plans:
        p = llm_plan(question, entities, graph)
        if p:
            plans = [p]
    trace["sql_plans"] = [{"kind": p.kind, "engine": p.engine} for p in plans]

    evidence: list[Evidence] = []
    for plan in plans:
        params = {k: v for k, v in plan.params.items() if not k.startswith("_")}
        try:
            cols, rows = execute(plan.sql, params)
        except (SqlValidationError, sqlite3.Error) as exc:
            trace.setdefault("sql_errors", []).append(f"{plan.kind}: {exc}")
            continue
        evidence.append(_sql_evidence(plan, cols, rows, params))
    return evidence
