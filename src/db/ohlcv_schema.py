"""OHLCV ticks table schema definition and DDL utilities."""

from typing import Any, Optional

OHLCV_TICKS_TABLE_NAME: str = "ohlcv_ticks"


def get_create_ohlcv_ticks_table_ddl(
    table_name: str = OHLCV_TICKS_TABLE_NAME,
    schema: Optional[str] = None,
) -> str:
    """Generate DDL for creating the base unpartitioned OHLCV ticks table.

    Args:
        table_name: Name of the target table.
        schema: Optional schema qualifier.

    Returns:
        SQL DDL statement string.
    """
    qualified_name = f"{schema}.{table_name}" if schema else table_name
    return (
        f"CREATE TABLE IF NOT EXISTS {qualified_name} (\n"
        f"    time TIMESTAMPTZ NOT NULL,\n"
        f"    symbol VARCHAR(32) NOT NULL,\n"
        f"    open DOUBLE PRECISION NOT NULL,\n"
        f"    high DOUBLE PRECISION NOT NULL,\n"
        f"    low DOUBLE PRECISION NOT NULL,\n"
        f"    close DOUBLE PRECISION NOT NULL,\n"
        f"    volume DOUBLE PRECISION NOT NULL\n"
        f");"
    )


def create_ohlcv_ticks_table(
    conn: Any,
    table_name: str = OHLCV_TICKS_TABLE_NAME,
    schema: Optional[str] = None,
) -> None:
    """Execute DDL to create the base OHLCV ticks table.

    Args:
        conn: DB-API 2.0 compatible database connection.
        table_name: Name of the target table.
        schema: Optional schema qualifier.
    """
    ddl = get_create_ohlcv_ticks_table_ddl(table_name=table_name, schema=schema)
    with conn.cursor() as cursor:
        cursor.execute(ddl)
    conn.commit()