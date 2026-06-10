import sqlite3

import pytest

from hybrid_rag import sql_tool
from hybrid_rag.config import settings
from hybrid_rag.sql_tool import SqlValidationError


@pytest.mark.parametrize("sql", [
    "DROP TABLE customers",
    "DELETE FROM orders",
    "UPDATE products SET unit_price = 0",
    "INSERT INTO customers VALUES (9, 'x', 'y', 'z')",
    "PRAGMA table_info(customers)",
    "ATTACH DATABASE 'x' AS x",
    "SELECT 1; SELECT 2",                       # multiple statements
    "SELECT * FROM customers; DROP TABLE customers",
])
def test_validation_gate_rejects(sql):
    with pytest.raises(SqlValidationError):
        sql_tool.validate_sql(sql)


def test_appends_limit_when_missing():
    out = sql_tool.validate_sql("SELECT name FROM customers")
    assert out.upper().endswith(f"LIMIT {settings.sql_row_limit}")


def test_connection_is_read_only():
    # Even if a write slipped past validation, mode=ro stops it at the engine.
    conn = sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("CREATE TABLE pwned (id INTEGER)")
    conn.close()


def test_units_template_not_triggered_by_value_question(graph):
    plans = sql_tool.plan_queries(
        "What was Acme Corp's total order value in 2025?",
        {"customer:1"}, graph)
    kinds = {p.kind for p in plans}
    assert "order_value" in kinds
    assert "units_purchased" not in kinds
