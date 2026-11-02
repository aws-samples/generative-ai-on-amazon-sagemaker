"""Run generated code in Amazon Bedrock AgentCore Code Interpreter.

This module is the sandbox backend of the code RL (reinforcement learning) notebook 5b and of
`coderl/eval_adapter.py`. Model-generated programs are untrusted, so they run in Code Interpreter instead of
on the notebook host.

`CodeInterpreterSandbox.run(request)` takes a SandboxFusion-style request
({"code": ..., "files": {name: base64}, "run_timeout": seconds}) and returns a
SandboxFusion-style response, so the cookbook's grading code can use it through
`sandbox_proxy.LocalSandbox`.

Each request runs in its own directory of one of a few long-lived sessions.
Create the code interpreter with network mode SANDBOX (no network access) and
no execution role (no AWS credentials inside).
"""

from __future__ import annotations

import base64
import queue
import shlex
import uuid

import boto3
from botocore.config import Config


class CodeInterpreterSandbox:
    """A pool of Code Interpreter sessions that runs Python programs in SandboxFusion request format.

    Attributes:
      id: the code interpreter ID.
      client: the boto3 `bedrock-agentcore` data-plane client used for all session calls.
      sessions: a queue of session IDs that are free to use.

    `run` is safe to call from several threads at once; each call holds one session while it runs.
    """

    def __init__(self, code_interpreter_id: str, region: str, sessions: int = 16):
        """Start `sessions` Code Interpreter sessions on the code interpreter `code_interpreter_id` in `region`.

        Args:
          code_interpreter_id: ID of an existing code interpreter that is in status READY.
          region: AWS Region of the code interpreter.
          sessions: number of sessions to start, which is the maximum number of programs that run at once.

        Side effects: starts `sessions` sessions, one API call each. The sessions are kept in a queue; each `run`
        call takes one, uses it, and puts it back. Call `close` to stop them.
        """
        self.id = code_interpreter_id
        self.client = boto3.client("bedrock-agentcore", region_name=region,
                                   config=Config(read_timeout=120, max_pool_connections=sessions,
                                                 retries={"max_attempts": 8, "mode": "adaptive"}))
        self.sessions = queue.Queue()
        for _ in range(sessions):
            self.sessions.put(self._start())

    def _start(self) -> str:
        """Start a new Code Interpreter session that times out after 3 hours and return its session ID.

        Side effects: one StartCodeInterpreterSession call. The session is not added to the queue here; the caller
        adds it.
        """
        return self.client.start_code_interpreter_session(
            codeInterpreterIdentifier=self.id, sessionTimeoutSeconds=3 * 3600)["sessionId"]

    def _call(self, session_id: str, name: str, arguments: dict) -> dict:
        """Invoke the Code Interpreter tool `name` with `arguments` in a session and return the tool result.

        Args:
          session_id: the session to run the tool in.
          name: the tool name, for example "writeFiles" or "executeCommand".
          arguments: the tool arguments, as documented for InvokeCodeInterpreter.

        Returns the result dict of the first event in the response stream that has one. If no event has a result,
        returns {"isError": True, "content": [{"text": "no result"}]}. API errors are raised to the caller.
        """
        out = self.client.invoke_code_interpreter(codeInterpreterIdentifier=self.id, sessionId=session_id,
                                                  name=name, arguments=arguments)
        for event in out["stream"]:
            if "result" in event:
                return event["result"]
        return {"isError": True, "content": [{"text": "no result"}]}

    def run(self, request: dict) -> dict:
        """Run one SandboxFusion-style request and return a SandboxFusion-style response.

        Args:
          request: a dict with "code" (the Python program), optionally "files" (a dict from file name to
            base64-encoded content) and optionally "run_timeout" (seconds, default 30).

        `request["code"]` is written to main.py and each entry of `request["files"]` is written next to it, in a new
        directory of a pooled session. main.py runs with python3 under `timeout`, with no stdin, then the directory
        is removed. Blocks until a session is free.

        Returns a dict with "status" ("Success" when main.py exits with code 0, "Failed" otherwise), "message",
        "compile_result" (always None) and "run_result". run_result has "status" ("Finished" or
        "TimeLimitExceeded"), "return_code" (None on timeout), "stdout", "stderr" and "execution_time". On a Code
        Interpreter error, returns status "SandboxError", the error message, and run_result None.

        Side effects: if the session has expired, a new session is started in its place.
        """
        work = f"job-{uuid.uuid4().hex}"
        timeout = float(request.get("run_timeout", 30))
        files = [{"path": f"{work}/main.py", "text": request["code"]}]
        files += [{"path": f"{work}/{name}", "blob": base64.b64decode(data)} for name, data in (request.get("files") or {}).items()]
        session_id = self.sessions.get()  # waits until a session is free
        try:
            self._call(session_id, "writeFiles", {"content": files})
            result = self._call(session_id, "executeCommand", {
                "command": f"cd {work} && timeout {timeout:g} python3 main.py < /dev/null; "
                           f"code=$?; cd .. && rm -rf {shlex.quote(work)}; exit $code"})
        except self.client.exceptions.ResourceNotFoundException:
            self.sessions.put(self._start())  # session expired: replace it
            return {"status": "SandboxError", "message": "session expired", "run_result": None}
        except Exception as exc:  # noqa: BLE001
            self.sessions.put(session_id)  # the session is still usable: return it to the pool
            return {"status": "SandboxError", "message": str(exc)[:500], "run_result": None}
        self.sessions.put(session_id)
        out = result.get("structuredContent") or {}
        code = out.get("exitCode")  # exit code of main.py, passed through by `exit $code`
        timed_out = code == 124  # exit code of `timeout`
        run = {"status": "TimeLimitExceeded" if timed_out else "Finished", "return_code": None if timed_out else code,
               "stdout": (out.get("stdout") or "").replace("\r\n", "\n"),
               "stderr": (out.get("stderr") or "").replace("\r\n", "\n"),
               "execution_time": out.get("executionTime")}
        return {"status": "Success" if code == 0 else "Failed", "message": "", "compile_result": None, "run_result": run}

    def close(self):
        """Stop every session that is in the pool and empty the pool.

        Errors from individual sessions are ignored. Sessions held by a `run` call that is still in progress are not
        stopped. The code interpreter itself is kept.
        """
        while not self.sessions.empty():
            try:
                self.client.stop_code_interpreter_session(codeInterpreterIdentifier=self.id,
                                                          sessionId=self.sessions.get_nowait())
            except Exception:  # noqa: BLE001
                pass
