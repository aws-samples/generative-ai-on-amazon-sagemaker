"""asyncio.gather with a progress bar.

The code RL (reinforcement learning) notebook 5b uses `gather` to wait for many sampling and grading requests at
once and show how many have finished.
"""

import asyncio

from tqdm.auto import tqdm


async def gather(coros, desc: str) -> list:
    """Run coroutines concurrently and return their results in order, updating a progress bar.

    Args:
      coros: an iterable of coroutines or other awaitables.
      desc: the label of the progress bar.

    Returns a list with the result of each coroutine, in the order of `coros`. If one raises, the exception is
    raised to the caller. Side effects: shows a tqdm progress bar that advances by one per finished coroutine and
    is removed at the end.
    """
    coros = list(coros)
    bar = tqdm(total=len(coros), desc=desc, leave=False)

    async def tick(c):
        """Await one coroutine, advance the progress bar by one, and return the coroutine's result.

        Args:
          c: the coroutine to await.
        """
        r = await c
        bar.update(1)
        return r

    try:
        return list(await asyncio.gather(*[tick(c) for c in coros]))
    finally:
        bar.close()
