"""Working memory and LangGraph checkpointer factory."""

import sys
from typing import Any, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from src.app.core.config import Settings, get_settings
from src.app.core.logging import get_logger

logger = get_logger(__name__)

_global_checkpointer: Optional[BaseCheckpointSaver] = None
_global_pool: Optional[Any] = None
_init_lock: Optional[Any] = None


def _get_init_lock() -> Any:
    """Return an asyncio.Lock tied to the active event loop for concurrency-safe initialization."""
    global _init_lock
    if _init_lock is None:
        import asyncio

        _init_lock = asyncio.Lock()
    return _init_lock


async def init_checkpointer(
    settings: Optional[Settings] = None,
) -> BaseCheckpointSaver:
    """Initialize checkpointer and connection pool at application startup."""
    global _global_checkpointer, _global_pool
    if _global_checkpointer is not None:
        return _global_checkpointer

    app_settings = settings or get_settings()

    if app_settings.checkpoint_backend == "memory":
        logger.info("Initializing in-memory MemorySaver checkpointer.")
        _global_checkpointer = MemorySaver()
        return _global_checkpointer

    pool: Optional[Any] = None
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool

        # On Windows, ensure selector event loop policy if needed for async psycopg
        if sys.platform == "win32":
            import asyncio

            try:
                if not isinstance(
                    asyncio.get_event_loop_policy(),
                    asyncio.WindowsSelectorEventLoopPolicy,
                ):
                    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            except Exception as loop_err:
                logger.debug("Windows event loop policy note: %s", loop_err)

        # Construct connection string from PostgreSQL settings
        db_url = app_settings.database_url
        if db_url.startswith("postgresql+asyncpg://"):
            db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

        pool = AsyncConnectionPool(
            conninfo=db_url,
            max_size=10,
            timeout=5.0,
            kwargs={"autocommit": True, "connect_timeout": 3},
            open=False,
        )
        await pool.open()
        _global_pool = pool

        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        logger.info("Initialized durable AsyncPostgresSaver connected to PostgreSQL.")
        _global_checkpointer = checkpointer
        return checkpointer
    except Exception as e:
        logger.error(
            "Failed to initialize PostgreSQL checkpointer: %s",
            e,
        )
        if pool is not None:
            try:
                await pool.close()
            except Exception:
                pass
        _global_pool = None
        _global_checkpointer = None
        raise RuntimeError(f"Failed to initialize PostgreSQL checkpointer backend: {e}") from e


async def close_checkpointer() -> None:
    """Close connection pool and release resources on application shutdown."""
    global _global_checkpointer, _global_pool, _init_lock
    if _global_pool is not None:
        try:
            logger.info("Closing PostgreSQL checkpointer connection pool.")
            await _global_pool.close()
        except Exception as e:
            logger.warning("Error closing checkpointer connection pool: %s", e)
        finally:
            _global_pool = None
    _global_checkpointer = None
    _init_lock = None


async def get_checkpointer(
    settings: Optional[Settings] = None,
    ephemeral: bool = False,
) -> BaseCheckpointSaver:
    """Return checkpoint saver (MemorySaver for tests, AsyncPostgresSaver for prod)."""
    global _global_checkpointer
    app_settings = settings or get_settings()

    if ephemeral:
        return MemorySaver()

    if _global_checkpointer is not None:
        return _global_checkpointer

    return await init_checkpointer(app_settings)


def reset_checkpointer() -> None:
    """Reset the global checkpointer instance (primarily for testing)."""
    global _global_checkpointer, _global_pool, _init_lock
    _global_checkpointer = None
    _global_pool = None
    _init_lock = None


def get_connection_pool() -> Optional[Any]:
    """Return current active connection pool if initialized."""
    return _global_pool


def set_connection_pool(
    pool: Optional[Any],
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> None:
    """Set global connection pool and optional checkpointer (primarily for testing/injection)."""
    global _global_pool, _global_checkpointer
    _global_pool = pool
    if checkpointer is not None:
        _global_checkpointer = checkpointer


async def check_postgres_readiness(
    timeout_seconds: float = 2.0,
    settings: Optional[Settings] = None,
) -> bool:
    """Perform a lightweight PostgreSQL connectivity check with self-healing initialization."""
    global _global_pool, _global_checkpointer
    app_settings = settings or get_settings()

    if app_settings.checkpoint_backend == "memory":
        return True

    # If pool is missing, attempt concurrency-safe self-healing initialization
    if _global_pool is None:
        lock = _get_init_lock()
        async with lock:
            if _global_pool is None:
                try:
                    await init_checkpointer(app_settings)
                except Exception as init_err:
                    logger.debug(
                        "PostgreSQL self-healing initialization deferred: %s",
                        init_err,
                    )
                    return False

    if _global_pool is None:
        return False

    try:
        import asyncio

        async def _ping() -> bool:
            async with _global_pool.connection(timeout=timeout_seconds) as conn:
                await conn.execute("SELECT 1;")
                return True

        return await asyncio.wait_for(_ping(), timeout=timeout_seconds)
    except Exception as e:
        logger.warning("PostgreSQL readiness check failed: %s", e)
        return False
