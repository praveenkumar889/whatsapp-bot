# db/workflow_state.py — Workflow State Manager
#
# PURPOSE:
#   Manages the WORKFLOW_PENDING state for incomplete orders.
#   When a customer provides partial order info (product but no quantity,
#   or quantity but no product), we cache what we have and wait for the rest.
#
# AC 1: Flags sessions as WORKFLOW_PENDING, caches gathered variables
# AC 3: 20-minute expiry window — auto-merges or resets
#
# FIVE FUNCTIONS:
#   get_pending_state()  → check if session has active pending order
#   save_pending_state() → cache partial order + set 20min expiry
#   merge_state()        → combine cached + new entities
#   complete_state()     → mark order as COMPLETED when all fields collected
#   expire_state()       → cleanup expired states

import json
from datetime import datetime, timezone, timedelta
from typing import Optional
from supabase import create_client, Client  # type: ignore[import]
from config import SUPABASE_URL, SUPABASE_SERVICE_KEY
from models.schemas import EntityResult

# ── Supabase client — lazy singleton ──────────────────────────────────────────
_supabase: Optional[Client] = None

def _get_client() -> Client:
    global _supabase
    if _supabase is None:
        _supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase


def _now_ist() -> datetime:
    """Returns current time in IST timezone."""
    return datetime.now(timezone(timedelta(hours=5, minutes=30)))


async def get_pending_state(tenant_id: str, session_id: str) -> Optional[dict]:
    """
    Checks if this customer has an active WORKFLOW_PENDING state.

    WHY:
        When Praveen says "I want to order prawns pickle" and the bot asks
        "How many units?", the next message "500gm" needs to be merged with
        the cached product_name from the previous turn.
        
        Without this, "500gm" alone has no context and the bot asks
        "Which product?" again — frustrating the customer.

    CHECKS:
        1. Status must be WORKFLOW_PENDING (not COMPLETED or EXPIRED)
        2. expires_at must be in the future (within 20-minute window)
        
    Returns:
        dict → cached state with product_name, quantity_value, quantity_unit etc.
        None → no active pending state (fresh start)
    """
    try:
        now_utc = datetime.now(timezone.utc).isoformat()

        result = _get_client().table("workflow_sessions") \
            .select("*") \
            .eq("tenant_id", tenant_id) \
            .eq("session_id", session_id) \
            .eq("status", "WORKFLOW_PENDING") \
            .gt("expires_at", now_utc) \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()

        if result.data:
            state = result.data[0]
            print(f"[STATE] Active WORKFLOW_PENDING found — product={state.get('product_name')} qty={state.get('quantity_value')}")
            return state

        print(f"[STATE] No active pending state for {session_id}")
        return None

    except Exception as e:
        print(f"[STATE] get_pending_state failed: {e}")
        return None


async def save_pending_state(
    tenant_id:     str,
    session_id:    str,
    product_name:  Optional[str],
    quantity_value: Optional[int],
    quantity_unit:  Optional[str],
    delivery_date:  str,
    missing_fields: list,
    items:         list = None,  # Full items list for multi-product orders
) -> bool:
    """
    Saves or updates the WORKFLOW_PENDING state for this customer session.

    MULTI-PRODUCT:
        Stores the full items list as JSON in items_json column.
        On follow-up, merge_state() restores all items and fills in
        the missing quantity for the right item.

    AC 1: Flags session as WORKFLOW_PENDING + caches gathered variables.
    AC 3: Sets expires_at = now + 20 minutes.
    """
    try:
        now_utc    = datetime.now(timezone.utc)
        expires_at = (now_utc + timedelta(minutes=20)).isoformat()

        existing = _get_client().table("workflow_sessions") \
            .select("id") \
            .eq("tenant_id", tenant_id) \
            .eq("session_id", session_id) \
            .eq("status", "WORKFLOW_PENDING") \
            .limit(1) \
            .execute()

        # Serialize items list to JSON for storage
        items_json_str = None
        if items:
            items_json_str = json.dumps([
                {
                    "product_name":   i.product_name,
                    "quantity_value": i.quantity_value,
                    "quantity_unit":  i.quantity_unit,
                }
                for i in items
            ])

        row = {
            "tenant_id":      tenant_id,
            "session_id":     session_id,
            "status":         "WORKFLOW_PENDING",
            "product_name":   product_name,
            "quantity_value": quantity_value,
            "quantity_unit":  quantity_unit,
            "delivery_date":  delivery_date,
            "missing_fields": json.dumps(missing_fields),
            "items_json":     items_json_str,
            "expires_at":     expires_at,
            "updated_at":     now_utc.isoformat(),
        }

        if existing.data:
            _get_client().table("workflow_sessions") \
                .update(row) \
                .eq("id", existing.data[0]["id"]) \
                .execute()
            print(f"[STATE] WORKFLOW_PENDING updated — items={len(items) if items else 1} missing={missing_fields}")
        else:
            _get_client().table("workflow_sessions") \
                .insert(row) \
                .execute()
            print(f"[STATE] WORKFLOW_PENDING created — items={len(items) if items else 1} missing={missing_fields} expires={expires_at}")

        return True

    except Exception as e:
        print(f"[STATE] save_pending_state failed: {e}")
        return False


def merge_state(cached_state: dict, new_entities: EntityResult) -> EntityResult:
    """
    Merges cached workflow state with new entities from the latest follow-up message.

    MULTI-PRODUCT FIX:
        Restores the full cached items list from items_json.
        The follow-up answer (e.g. "3 units") fills in the FIRST missing quantity
        across all cached items — NOT just items[0].

    Example:
        Cached: [{flood lights, qty=10}, {street lights, qty=None}]
        Follow-up: "3 units"
        Merged: [{flood lights, qty=10}, {street lights, qty=3}] ✅
    """
    from models.schemas import OrderItem

    # Restore full items list from cached items_json if available
    cached_items = []
    items_json_str = cached_state.get("items_json")
    if items_json_str:
        try:
            raw_items = json.loads(items_json_str)
            cached_items = [
                OrderItem(
                    product_name   = i.get("product_name"),
                    quantity_value = i.get("quantity_value"),
                    quantity_unit  = i.get("quantity_unit"),
                )
                for i in raw_items
            ]
        except Exception as e:
            print(f"[STATE] Failed to restore items_json: {e}")

    # Fall back to single-item from flat columns if no items_json
    if not cached_items:
        cached_items = [OrderItem(
            product_name   = cached_state.get("product_name"),
            quantity_value = cached_state.get("quantity_value"),
            quantity_unit  = cached_state.get("quantity_unit"),
        )]

    new_items = new_entities.items

    # Single follow-up answer (e.g. "2 units", "2 units each", "I want 2")
    # The entity extractor already handled "each" semantics via LLM context.
    # If it returned N items matching our N cached items — use them directly.
    # If it returned 1 item — fill the first missing field in cached items.
    if len(new_items) == 1:
        new_item     = new_items[0]
        merged_items = []
        filled       = False

        for cached_item in cached_items:
            if not filled:
                # Fill missing quantity
                if cached_item.quantity_value is None and new_item.quantity_value is not None:
                    merged_items.append(OrderItem(
                        product_name   = cached_item.product_name or new_item.product_name,
                        quantity_value = new_item.quantity_value,
                        quantity_unit  = new_item.quantity_unit or cached_item.quantity_unit,
                    ))
                    filled = True
                    continue
                # Fill missing product
                elif cached_item.product_name is None and new_item.product_name is not None:
                    merged_items.append(OrderItem(
                        product_name   = new_item.product_name,
                        quantity_value = cached_item.quantity_value or new_item.quantity_value,
                        quantity_unit  = cached_item.quantity_unit or new_item.quantity_unit,
                    ))
                    filled = True
                    continue
            merged_items.append(cached_item)

        if not filled:
            merged_items = cached_items

    elif len(new_items) == len(cached_items):
        # Entity extractor returned same number of items as cached — direct merge
        # Happens when LLM correctly expanded "2 units each" → N items with qty=2
        merged_items = []
        for new_item, cached_item in zip(new_items, cached_items):
            merged_items.append(OrderItem(
                product_name   = new_item.product_name   or cached_item.product_name,
                quantity_value = new_item.quantity_value if new_item.quantity_value is not None
                                 else cached_item.quantity_value,
                quantity_unit  = new_item.quantity_unit  or cached_item.quantity_unit,
            ))
        print(f"[STATE] Direct N-to-N merge — {len(merged_items)} items")

    else:
        # Customer re-stated all items differently — use new extraction as-is
        merged_items = new_items

    # Recalculate missing across all merged items
    missing = []
    for i, item in enumerate(merged_items):
        prefix = f"item_{i+1}_" if len(merged_items) > 1 else ""
        if not item.product_name:
            missing.append(f"{prefix}product_name")
        if item.quantity_value is None:
            missing.append(f"{prefix}quantity")

    merged_delivery = new_entities.delivery_date or cached_state.get("delivery_date")

    print(f"[STATE] Merged — {len(merged_items)} item(s) missing={missing}")
    for i, item in enumerate(merged_items):
        print(f"  item[{i}]: product={item.product_name} qty={item.quantity_value} {item.quantity_unit}")

    return EntityResult(
        items             = merged_items,
        delivery_date     = merged_delivery or new_entities.delivery_date,
        invoice_number    = new_entities.invoice_number,
        payment_reference = new_entities.payment_reference,
        missing_entities  = missing,
        raw_text          = new_entities.raw_text,
        tenant_id         = new_entities.tenant_id,
    )


async def complete_state(tenant_id: str, session_id: str) -> bool:
    """
    Marks the workflow state as COMPLETED when all required fields are collected.

    Called when both product_name and quantity are present after merging.
    The customer will get the order confirmation reply.

    Returns:
        True → marked complete
        False → failed (logged, pipeline continues)
    """
    try:
        _get_client().table("workflow_sessions") \
            .update({
                "status":     "COMPLETED",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }) \
            .eq("tenant_id", tenant_id) \
            .eq("session_id", session_id) \
            .eq("status", "WORKFLOW_PENDING") \
            .execute()

        print(f"[STATE] WORKFLOW_PENDING -> COMPLETED for {session_id}")
        return True

    except Exception as e:
        print(f"[STATE] complete_state failed: {e}")
        return False


async def expire_state(tenant_id: str, session_id: str) -> bool:
    """
    Marks expired WORKFLOW_PENDING states as EXPIRED.

    Called when get_pending_state() finds the expires_at is in the past.
    The customer will get a fresh start message instead of a confused reply.

    AC 3: Cleanly expires state if 20-minute window exceeded.

    Returns:
        True → marked expired
        False → failed (logged, pipeline continues)
    """
    try:
        now_utc = datetime.now(timezone.utc).isoformat()

        _get_client().table("workflow_sessions") \
            .update({
                "status":     "EXPIRED",
                "updated_at": now_utc,
            }) \
            .eq("tenant_id", tenant_id) \
            .eq("session_id", session_id) \
            .eq("status", "WORKFLOW_PENDING") \
            .lt("expires_at", now_utc) \
            .execute()

        print(f"[STATE] Expired pending state for {session_id}")
        return True

    except Exception as e:
        print(f"[STATE] expire_state failed: {e}")
        return False