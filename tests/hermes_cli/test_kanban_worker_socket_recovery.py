"""A Kanban worker whose gateway socket dies mid-turn resumes the same attempt on a new connection.

A dead socket is a transport event, not a turn outcome: the owner keeps the admission, answers
``prompt.receipt`` over the next connection and returns the same admission for a repeated
``kanban.run`` (one request id per attempt). Exiting 1 instead loses work that was already running,
because ``_watch_submitter`` then ends the owner's in-flight turn (#145062). An owner's own refusal
is still an answer, and an owner that stays silent still fails the worker.
"""

import asyncio
from contextlib import asynccontextmanager

import pytest

from hermes_cli import gateway_client, kanban_worker_client


class _Clock:
    """Monotonic time that moves only when the client sleeps or a scripted step advances it."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now


class _Owner:
    """Scripted owner. ``connects`` holds one step per dial, ``receipts`` one per ``prompt.receipt``
    and ``submits`` one per ``kanban.run``; a step is None (succeed), an exception (raised) or a
    callable (run first, so it can advance the clock or raise). ``refuse_dials_after`` fails every
    dial past the scripted ones with a refused connection."""

    def __init__(self, connects=(), receipts=(), submits=(), refuse_dials_after=False):
        self.connects, self.receipts, self.submits = list(connects), list(receipts), list(submits)
        self.refuse_dials_after = refuse_dials_after
        self.calls = []
        self.dials = 0

    @staticmethod
    def _step(steps, default=None):
        step = steps.pop(0) if steps else default
        if callable(step) and not isinstance(step, Exception):
            step = step()
        if isinstance(step, Exception):
            raise step
        return step

    def connect(self):
        owner = self

        @asynccontextmanager
        async def _connect():
            owner.dials += 1
            refused = gateway_client.GatewayClientError("Gateway connection failed; no local fallback")
            owner._step(owner.connects, refused if owner.refuse_dials_after else None)

            class _Client:
                def __init__(self):
                    self.events = asyncio.Queue()

                async def rpc(self, method, **params):
                    owner.calls.append(method)
                    if method == "kanban.run":
                        owner._step(owner.submits)
                        return {"session_id": "s1", "receipt": {"status": "pending", "admission_id": "a1"}}
                    assert params == {"session_id": "s1", "admission_id": "a1"}
                    return {"status": owner._step(owner.receipts, "terminal")}

            yield _Client()

        return _connect


def _disconnect():
    return gateway_client.GatewayClientError(
        "Gateway disconnected; turn outcome is unknown. Resume the printed session ID")


@pytest.fixture
def clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(kanban_worker_client, "time", clock)

    async def _sleep(seconds):
        clock.now += seconds
    monkeypatch.setattr(kanban_worker_client.asyncio, "sleep", _sleep)
    return clock


@pytest.fixture(autouse=True)
def _worker_env(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    from gateway import session_kanban
    monkeypatch.setattr(session_kanban, "worker_result",
                        lambda *_: {"exit_code": 0, "last_output": "card closed"})
    return tmp_path


async def _run(owner, monkeypatch, tmp_path):
    monkeypatch.setattr(gateway_client, "connect_gateway", owner.connect())
    return await kanban_worker_client.run({"task_id": "t1"}, str(tmp_path / "kanban.db"))


@pytest.mark.asyncio
@pytest.mark.parametrize("redial", [
    None,
    gateway_client.GatewayClientError("Gateway connection failed; no local fallback"),
    ConnectionRefusedError(61, "refused"),
    gateway_client.GatewayUnavailableError("Gateway starting: not_ready", "starting"),
], ids=["reconnects", "dial-refused", "ticket-oserror", "owner-starting"])
async def test_a_socket_lost_mid_turn_resumes_the_same_attempt(monkeypatch, _worker_env, clock, redial):
    owner = _Owner(connects=[None, redial, None], receipts=["pending", _disconnect(), "terminal"])

    assert await _run(owner, monkeypatch, _worker_env) == 0
    # The claim is submitted once: every later connection re-reads the receipt.
    assert owner.calls.count("kanban.run") == 1
    assert owner.dials == (2 if redial is None else 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("lost", [_disconnect(), gateway_client.GatewayRPCError("runtime_draining")],
                         ids=["answer-lost", "draining"])
async def test_a_claim_with_no_answer_is_resubmitted(monkeypatch, _worker_env, clock, lost):
    # No admission id reached the worker, so it asks again; the owner answers a repeated
    # kanban.run for the same attempt with the same admission.
    owner = _Owner(connects=[None, None], submits=[lost], receipts=["terminal"])

    assert await _run(owner, monkeypatch, _worker_env) == 0
    assert owner.calls.count("kanban.run") == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["kanban.run", "prompt.receipt"])
async def test_an_owner_refusal_is_not_retried(monkeypatch, _worker_env, clock, where):
    refusal = gateway_client.GatewayRPCError("claim_mismatch")
    owner = (_Owner(submits=[refusal]) if where == "kanban.run"
             else _Owner(receipts=["pending", refusal]))

    with pytest.raises(gateway_client.GatewayRPCError, match="claim_mismatch"):
        await _run(owner, monkeypatch, _worker_env)
    assert owner.dials == 1


@pytest.mark.asyncio
async def test_a_later_loss_gets_a_fresh_budget_from_the_owners_last_answer(monkeypatch, _worker_env, clock):
    def silent_then_lost():
        clock.now += kanban_worker_client.OWNER_WAIT_S - 10
        return _disconnect()

    # Each loss comes 170 s after the last answer; across both, far more than OWNER_WAIT_S passes.
    owner = _Owner(connects=[None, None, None],
                   receipts=["pending", silent_then_lost, "pending", silent_then_lost, "terminal"])

    assert await _run(owner, monkeypatch, _worker_env) == 0
    assert owner.dials == 3 and owner.calls.count("kanban.run") == 1


@pytest.mark.asyncio
async def test_an_owner_silent_for_the_whole_budget_fails_the_worker(monkeypatch, _worker_env, clock):
    owner = _Owner(connects=[None], receipts=["pending", _disconnect()], refuse_dials_after=True)

    with pytest.raises(gateway_client.GatewayClientError, match="unreachable for 180s"):
        await _run(owner, monkeypatch, _worker_env)
    assert clock.now >= kanban_worker_client.OWNER_WAIT_S
    # Jittered exponential backoff (capped at 8-12 s) between dials, not one dial a second.
    assert 3 < owner.dials < 40 and owner.calls.count("kanban.run") == 1
