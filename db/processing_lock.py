# db/processing_lock.py — Distributed Processing Lock using Supabase
#
# PURPOSE:
#   Prevents the same customer session from being processed simultaneously
#   across multiple workers. Replaces the in-memory _processing_sessions set.
#
# HOW IT WORKS:
#   acquire_lock() -> INSERT a row with session_id as PRIMARY KEY
#                    If insert succeeds -> lock acquired (safe to process)
#                    If insert fails (duplicate key) -> already locked -> skip
#
#   release_lock() -> DELETE the row when processing is done
#
# CRASH SAFETY:
#   A background cleanup removes stale locks older than 2 minutes.
#   This handles the case where the server crashed mid-pipeline
#   and release_lock() never ran.

from datetime import datetime, timezone, timedelta
from typing import Optional
from supabase import create_client, Client  # type: ignore[import]
from config import SUPABASE_URL, SUPABASE_SERVICE_KEY

_supabase: Optional[Client] = None

def _get_client() -> Client:
    global _supabase
    if _supabase is None:
        _supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase


async def acquire_lock(session_id: str, tenant_id: str) -> bool:
    """
    Try to acquire a processing lock for this session.

    Returns True  -> lock acquired, safe to process this message.
    Returns False -> session already being processed, skip this message.

    Uses Supabase INSERT — the PRIMARY KEY constraint on session_id
    makes this atomic. Two workers cannot both acquire the lock
    because only one INSERT can succeed for the same session_id.
    """
    try:
        _get_client().table("processing_locks").insert({
            "session_id": session_id,
            "tenant_id":  tenant_id,
            "locked_at":  datetime.now(timezone.utc).isoformat(),
        }).execute()
        print(f"[LOCK] Acquired for {session_id}")
        return True

    except Exception:
        # INSERT failed — duplicate key = session already being processed
        print(f"[LOCK] Already locked - skipping {session_id}")
        return False


async def release_lock(session_id: str) -> None:
    """
    Release the processing lock after pipeline completes.
    Called in the finally block — always runs even if pipeline crashes.
    """
    try:
        _get_client().table("processing_locks") \
            .delete() \
            .eq("session_id", session_id) \
            .execute()
        print(f"[LOCK] Released for {session_id}")
    except Exception as e:
        print(f"[LOCK] Release failed: {e}")
        # Not critical — stale lock will be cleaned up by cleanup_stale_locks()


async def cleanup_stale_locks() -> None:
    """
    Deletes locks older than 2 minutes.

    Called at the start of every incoming message (before acquire_lock).
    Handles the edge case where server crashed mid-pipeline and
    release_lock() never ran — without this, that session would be
    permanently locked until server restart.

    2 minutes is safely above the longest possible pipeline run
    (worst case: 4 LLM calls * 30s timeout = 120s).
    """
    try:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(minutes=2)
        ).isoformat()

        _get_client().table("processing_locks") \
            .delete() \
            .lt("locked_at", cutoff) \
            .execute()

    except Exception as e:
        print(f"[LOCK] Stale cleanup failed: {e}")
