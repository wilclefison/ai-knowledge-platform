import psycopg
from psycopg.rows import dict_row
from src.config import settings
import logging

logger = logging.getLogger(__name__)

async def get_db_connection():
    """Returns an asynchronous PostgreSQL database connection."""
    conn = await psycopg.AsyncConnection.connect(
        settings.DATABASE_URL,
        row_factory=dict_row
    )
    return conn

async def init_db():
    """Initializes the database schema if not already present."""
    try:
        async with await get_db_connection() as conn:
            with open("src/db/schema.sql", "r", encoding="utf-8") as f:
                schema_sql = f.read()
            async with conn.cursor() as cur:
                await cur.execute(schema_sql)
            await conn.commit()
            logger.info("Database schema initialized successfully.")
    except Exception as e:
        logger.warning(f"Could not connect to live DB for auto-migration (check if Docker is running): {e}")
