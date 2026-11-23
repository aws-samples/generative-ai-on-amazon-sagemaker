"""SandboxFusion-compatible /run_code endpoint on 127.0.0.1.

The training sessions cookbook grades code by sending HTTP requests to a SandboxFusion server at SANDBOX_URL.
This module lets the code RL (reinforcement learning) notebook 5b and `coderl/eval_adapter.py` use Code
Interpreter instead, without changing the cookbook: SANDBOX_URL points at a local endpoint that passes each
request to a Python function.

Code that expects a SandboxFusion URL (for example the cookbook's sandbox client) can point
SANDBOX_URL at this endpoint. `LocalSandbox(run)` serves any function that maps a request to a
response, for example `CodeInterpreterSandbox.run`. Nothing listens on a public interface.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from aiohttp import web


class LocalSandbox:
    """HTTP server on 127.0.0.1 that answers POST /run_code by calling a run function.

    Attributes:
      run: the function that handles each request.
      pool: the thread pool in which `run` is called.
      sem: limits the number of requests in `run` at once; created on the server's event loop by `start`.
      concurrency: the size of the thread pool and of the semaphore.
      url: the endpoint URL, set by `start`.
    """

    def __init__(self, run, concurrency: int = 16):
        """Store the server settings and create the thread pool. Call `start` to serve.

        Args:
          run: function that takes a request dict and returns a response dict. It is called in a thread pool, so
            it may block.
          concurrency: maximum number of requests passed to `run` at the same time. Further requests wait.
        """
        self.run = run
        self.pool = ThreadPoolExecutor(concurrency)
        self.sem: asyncio.Semaphore | None = None
        self.concurrency = concurrency
        self.url = None
        self._loop = None

    def invoke(self, payload: dict) -> dict:
        """Call the run function with one request payload and return its response.

        Args:
          payload: the decoded JSON body of one /run_code request.

        Returns the response dict from `run`. Exceptions from `run` are raised to the caller.
        """
        return self.run(payload)

    async def _handle(self, request: web.Request) -> web.Response:
        """Handle one POST /run_code: run the JSON body in the thread pool and return the result as JSON.

        Args:
          request: the aiohttp request. Its body must be a JSON object in SandboxFusion request format.

        Returns a JSON response with the dict returned by `run`. Waits on the semaphore while `concurrency`
        requests are already running.
        """
        payload = await request.json()
        async with self.sem:
            body = await asyncio.get_running_loop().run_in_executor(self.pool, self.invoke, payload)
        return web.json_response(body)

    def start(self) -> str:
        """Start the server in a daemon thread and return its URL, http://127.0.0.1:<port>/run_code.

        The operating system picks a free port. Waits up to 30 seconds for the server to start; if it has not
        started by then, returns None. Side effects: sets `url`, `sem` and `_loop`, and starts a thread named
        sandbox-proxy that runs until the process exits.
        """
        started = threading.Event()

        def serve():
            """Run the aiohttp server on a new event loop in this thread until the process exits.

            Binds to 127.0.0.1 on a free port, sets `url`, then signals `started` and runs the event loop.
            """
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # The semaphore is created here so that it belongs to the server's event loop.
            self.sem = asyncio.Semaphore(self.concurrency)
            app = web.Application(client_max_size=64 * 1024 * 1024)  # accept request bodies up to 64 MiB
            app.router.add_post("/run_code", self._handle)
            runner = web.AppRunner(app)
            loop.run_until_complete(runner.setup())
            site = web.TCPSite(runner, "127.0.0.1", 0)  # port 0: the operating system picks a free port
            loop.run_until_complete(site.start())
            port = runner.addresses[0][1]
            self.url = f"http://127.0.0.1:{port}/run_code"
            self._loop = loop
            started.set()
            loop.run_forever()

        threading.Thread(target=serve, daemon=True, name="sandbox-proxy").start()
        started.wait(30)
        return self.url

