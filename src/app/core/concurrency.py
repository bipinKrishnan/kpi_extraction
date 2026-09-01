import asyncio
import functools
from collections.abc import Awaitable, Callable

from anyio import CapacityLimiter, to_thread


def limit_concurrency(limit: int):
    """Cap how many calls of the decorated coroutine function run at once.

    Use this for async work (network calls, subprocesses). Blocking work
    belongs in `run_limited`, which caps threads instead.

    The limit is shared by every caller in the process, so it applies across
    concurrent requests too (and is per worker process, not per deployment).
    """

    def decorator[**P, T](
        func: Callable[P, Awaitable[T]],
    ) -> Callable[P, Awaitable[T]]:
        limiter = asyncio.Semaphore(limit)

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            async with limiter:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


async def run_limited[T](
    limiter: CapacityLimiter,
    func: Callable[..., T],
    *args: object,
    **kwargs: object,
) -> T:
    """Run a blocking call in a worker thread, capped by its own limiter.

    Each workload passes a dedicated CapacityLimiter, so it never competes for
    the default thread limiter that FastAPI shares with sync endpoints, sync
    dependencies and spooled upload reads. That keeps one slow workload from
    starving another.
    """
    # The one sanctioned use of the raw API; every caller goes through here.
    return await to_thread.run_sync(  # noqa: TID251
        functools.partial(func, *args, **kwargs), limiter=limiter
    )
