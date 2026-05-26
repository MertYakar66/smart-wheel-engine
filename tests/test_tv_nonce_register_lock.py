"""Concurrency tests for ``engine_api._tv_seen_register``.

The S20 reliability arc (``docs/USAGE_TEST_LEDGER.md`` §S20 + PR #194's
``docs/RELIABILITY_ARC_REVIEW.md`` C3) observed that the nonce-register
check-then-set was *lock-free* on `main` — CPython's GIL plus dict-op
atomicity happened to keep the small window race-free at workers=4 in
the v5 backfill test, but the pattern is fragile (a future logging
call between the membership test and the insertion, higher concurrency,
or a move off CPython would surface >1-accept anomalies).

This file pins the **post-fix** invariant: ``_tv_seen_register`` is
serialised by an explicit ``_TV_SEEN_NONCES_LOCK`` and is therefore
thread-safe regardless of CPython implementation details.

Hardware-realistic concurrency test (TestRegisterUnderContention):
spawns 64 worker threads all trying to register the same digest at
the same timestamp. Without the lock, under sufficient bytecode-window
contention, more than one worker could observe ``digest not in cache``
before any of them writes — that is the race the S20 finding named.
With the lock, exactly one worker must succeed.

Pure-state test (TestLockExists): the lock object is present and is
a ``threading.Lock`` (or ``RLock``) instance — a regression guard
against accidental removal during a future refactor.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest


class TestLockExists:
    """Structural regression guards: the lock must remain present."""

    def test_lock_attribute_exists_on_module(self):
        """The module must expose a ``_TV_SEEN_NONCES_LOCK`` attribute.

        A future refactor that removes the lock (or renames it) would
        fail this test and force a deliberate update — preventing
        accidental reversion to the lock-free pattern.
        """
        import engine_api

        assert hasattr(engine_api, "_TV_SEEN_NONCES_LOCK")

    def test_lock_is_a_threading_primitive(self):
        """The lock object must be a real mutex (not, e.g., a no-op
        sentinel). We accept either ``threading.Lock`` or
        ``threading.RLock`` since both satisfy the safety contract."""
        from engine_api import _TV_SEEN_NONCES_LOCK

        # threading.Lock() returns an instance of ``_thread.lock`` /
        # threading.RLock() returns ``_thread.RLock``. The cheapest portable
        # check is that the object supports the context-manager protocol
        # AND has ``acquire`` / ``release`` methods.
        assert hasattr(_TV_SEEN_NONCES_LOCK, "acquire")
        assert hasattr(_TV_SEEN_NONCES_LOCK, "release")
        assert hasattr(_TV_SEEN_NONCES_LOCK, "__enter__")
        assert hasattr(_TV_SEEN_NONCES_LOCK, "__exit__")

    def test_lock_acquires_and_releases_cleanly(self):
        """Quick acquire / release sanity check."""
        from engine_api import _TV_SEEN_NONCES_LOCK

        # Non-blocking acquire (no other thread should be holding it).
        acquired = _TV_SEEN_NONCES_LOCK.acquire(blocking=False)
        assert acquired, "lock was already held — test isolation bug"
        _TV_SEEN_NONCES_LOCK.release()


class TestRegisterUnderContention:
    """High-contention concurrency tests that would fail under the
    pre-fix lock-free pattern (with sufficient luck / contention) and
    must always pass under the post-fix locked pattern.
    """

    @pytest.fixture(autouse=True)
    def _clear_nonces(self):
        """Each test starts with a clean nonce cache."""
        import engine_api

        engine_api._TV_SEEN_NONCES.clear()
        yield
        engine_api._TV_SEEN_NONCES.clear()

    def test_same_digest_64_concurrent_workers_yields_exactly_one_accept(self):
        """Race-the-window: 64 workers attempt to register the same digest
        at the same timestamp. Without the lock, under sufficient
        contention, multiple workers can observe ``digest not in cache``
        before any of them writes. With the lock, exactly one succeeds.
        """
        from engine_api import _tv_seen_register

        digest = "race-digest-" + "a" * 50  # deterministic, fixed-length
        now = time.time()

        # Barrier ensures all threads start the call simultaneously.
        barrier = threading.Barrier(64)
        results = []
        results_lock = threading.Lock()

        def worker():
            barrier.wait()
            ok = _tv_seen_register(digest, now)
            with results_lock:
                results.append(ok)

        with ThreadPoolExecutor(max_workers=64) as ex:
            futures = [ex.submit(worker) for _ in range(64)]
            for f in as_completed(futures):
                f.result()  # surface any worker exception

        true_count = sum(1 for r in results if r)
        false_count = sum(1 for r in results if not r)

        # The invariant: exactly one worker may successfully register;
        # the other 63 must see the digest already present and return False.
        assert true_count == 1, (
            f"Expected exactly 1 accept under contention; got {true_count}. "
            f"This indicates the check-then-set is racing — the lock has "
            f"either been removed or made ineffective."
        )
        assert false_count == 63

    def test_distinct_digests_64_concurrent_workers_all_accepted(self):
        """The flip side: 64 workers each registering a DISTINCT digest
        must all succeed. The lock should serialise but not block valid
        traffic — a regression that, say, replaced the function body
        with ``return False`` would pass the same-digest test but fail
        this one.
        """
        from engine_api import _tv_seen_register

        now = time.time()
        barrier = threading.Barrier(64)
        results = {}
        results_lock = threading.Lock()

        def worker(worker_id: int):
            digest = f"distinct-digest-{worker_id:04d}-" + "b" * 40
            barrier.wait()
            ok = _tv_seen_register(digest, now)
            with results_lock:
                results[worker_id] = ok

        with ThreadPoolExecutor(max_workers=64) as ex:
            futures = [ex.submit(worker, i) for i in range(64)]
            for f in as_completed(futures):
                f.result()

        accepts = [v for v in results.values() if v]
        assert len(accepts) == 64, (
            f"Expected all 64 distinct digests to register; got {len(accepts)}. "
            f"The lock should serialise, not block."
        )

    def test_lock_is_released_after_each_call(self):
        """A leaked lock acquisition would deadlock the next caller.
        After a sequence of registers, the lock must be releasable
        (i.e., not held by some earlier call that crashed).
        """
        from engine_api import _TV_SEEN_NONCES_LOCK, _tv_seen_register

        now = time.time()
        for i in range(10):
            _tv_seen_register(f"sequential-digest-{i}", now + i)

        # If the lock were leaked, this acquire would block / fail.
        acquired = _TV_SEEN_NONCES_LOCK.acquire(timeout=1.0)
        assert acquired, "lock appears to have been leaked by _tv_seen_register"
        _TV_SEEN_NONCES_LOCK.release()
