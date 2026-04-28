"""Integration test: login → trigger re-extraction → poll until done.

Usage:
    python tests/test_reextract.py

Requires:
    - AI-KB backend running at http://localhost:8000
"""
from __future__ import annotations

import sys
import time
import httpx

BASE = "http://localhost:8000/v1"
EMAIL = "admin@neargo.ai"
PASSWORD = "NearGo2026!"
ORG_ID = "36f377e9-4e49-4b83-b7b0-681b973aa734"  # NearGo Admin org

POLL_INTERVAL = 3      # seconds between trace polls
TIMEOUT = 600          # seconds total


def step(msg: str) -> None:
    print(f"\n{'='*60}\n{msg}\n{'='*60}")


def login(client: httpx.Client) -> str:
    step("1. Login")
    r = client.post(f"{BASE}/auth/login", json={"email": EMAIL, "password": PASSWORD})
    r.raise_for_status()
    token = r.json()["access_token"]
    print(f"  OK  token: {token[:40]}...")
    return token


def pick_doc(client: httpx.Client) -> str:
    step(f"2. List docs in org {ORG_ID}")
    r = client.get(f"{BASE}/orgs/{ORG_ID}/docs")
    r.raise_for_status()
    items = r.json() if isinstance(r.json(), list) else r.json().get("items", [])
    if not items:
        print("  No docs found — aborting.")
        sys.exit(1)
    doc = items[0]
    doc_id = doc.get("id") or doc.get("doc_id")
    title = doc.get("title") or doc.get("name") or "(untitled)"
    print(f"  Using doc: {doc_id}  title={title!r}")
    return doc_id


def trigger_reparse(client: httpx.Client, doc_id: str) -> float:
    step(f"3. Trigger reparse  doc={doc_id}")
    t_trigger = time.monotonic()
    r = client.post(f"{BASE}/docs/{doc_id}/reparse")
    print(f"  status: {r.status_code}")
    if r.status_code not in (200, 202):
        print(f"  body: {r.text}")
        r.raise_for_status()
    else:
        print(f"  body: {r.json()}")
    return t_trigger


def poll_trace(client: httpx.Client, doc_id: str, t_trigger: float) -> None:
    step(f"4. Polling traces  doc={doc_id}")
    deadline = time.monotonic() + TIMEOUT
    terminal = {"completed", "failed", "error", "done", "success", "cancelled"}
    last_step = None
    status = "unknown"
    data: dict = {}
    seen_running = False   # must see "running/processing" before accepting terminal

    while time.monotonic() < deadline:
        try:
            r = client.get(f"{BASE}/docs/{doc_id}/traces/latest", timeout=10)
        except httpx.TimeoutException:
            print(f"  [{time.strftime('%H:%M:%S')}] (request timed out, backend may be busy)")
            time.sleep(POLL_INTERVAL)
            continue

        if r.status_code == 404:
            print(f"  [{time.strftime('%H:%M:%S')}] (trace not yet created, waiting...)")
            time.sleep(POLL_INTERVAL)
            continue
        r.raise_for_status()
        data = r.json()

        status = (
            data.get("status")
            or data.get("state")
            or data.get("pipeline_status")
            or "unknown"
        )
        current_step = data.get("current_step") or data.get("stage") or data.get("step") or ""
        action = (data.get("current_action") or "")[:70]
        elapsed = data.get("elapsed_seconds") or data.get("elapsed") or data.get("duration_s") or "?"

        # Track: once we see running, a subsequent terminal state is real.
        if status.lower() in ("running", "processing", "queued"):
            seen_running = True

        if current_step != last_step:
            print(f"  [{time.strftime('%H:%M:%S')}] status={status}  step={current_step}  action={action}")
            last_step = current_step

        if status.lower() in terminal:
            # If we've seen running/processing at least once, this is a genuine completion.
            # If not, the "failed" may be a stale pre-reparse entry — wait for the
            # new pipeline to start writing its own trace.
            elapsed_since_trigger = time.monotonic() - t_trigger
            if seen_running or elapsed_since_trigger > 15:
                print(f"\n  FINAL STATUS: {status}  (elapsed_since_trigger={elapsed_since_trigger:.0f}s)")
                _print_summary(data)
                return
            else:
                print(f"  [{time.strftime('%H:%M:%S')}] status={status} but waiting for new pipeline to start...")

        time.sleep(POLL_INTERVAL)

    print(f"\n  TIMEOUT after {TIMEOUT}s — last status: {status}")
    _print_summary(data)


def _print_summary(data: dict) -> None:
    keys = [
        "trace_id", "file_name", "duration_s",
        "entities_count", "triples_count", "chunks_count",
        "error",
    ]
    for k in keys:
        if k in data and data[k] not in (None, "", 0, {}):
            print(f"  {k}: {data[k]}")
    g = data.get("graph", {})
    if g:
        print(f"  graph.status: {g.get('status')}  triples_written: {g.get('triples_written', 0)}"
              f"  chunks_ok: {g.get('chunks_ok', 0)}/{g.get('chunks_total', 0)}")
    ch = data.get("chunk", {})
    if ch.get("total"):
        print(f"  chunk.total: {ch['total']}  by_level: {ch.get('by_level', {})}")
    em = data.get("embed", {})
    if em.get("vectors_indexed"):
        print(f"  embed.vectors_indexed: {em['vectors_indexed']}  dur: {em.get('duration_s', 0):.1f}s")


def main() -> None:
    print(f"Target: {BASE}")
    print(f"Org:    {ORG_ID}")
    with httpx.Client(timeout=30) as client:
        token = login(client)
        client.headers["Authorization"] = f"Bearer {token}"
        doc_id = pick_doc(client)
        t_trigger = trigger_reparse(client, doc_id)
        poll_trace(client, doc_id, t_trigger)
    print("\nDone.")


if __name__ == "__main__":
    main()
