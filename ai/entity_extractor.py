# ai/entity_extractor.py — Entity Extraction Engine with Smart Context

import json
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from openai import AzureOpenAI

from models.schemas import EntityResult, OrderItem
from config import (
    AZURE_AI_ENDPOINT,
    AZURE_AI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_AI_API_VERSION,
)

_client = AzureOpenAI(
    azure_endpoint = AZURE_AI_ENDPOINT,
    api_key        = AZURE_AI_API_KEY,
    api_version    = AZURE_AI_API_VERSION,
)

def _default_delivery_date() -> str:
    ist = timezone(timedelta(hours=5, minutes=30))
    return (datetime.now(ist) + timedelta(days=5)).strftime("%Y-%m-%d")


# Keywords that signal a NEW order is starting — ignore previous order context
NEW_ORDER_TRIGGERS = [
    "i want to order",
    "i want to place",
    "place an order",
    "new order",
    "i want to buy",
    "i need to order",
    "can i order",
]

def _is_new_order(message: str) -> bool:
    """
    Detects if the customer is starting a BRAND NEW order.

    WHY THIS MATTERS:
        "I want to order 2kg" → new order → ignore history context
        "Mutton pickle"       → follow-up → use history context (qty from prev msg)
        "2 kgs"               → follow-up → use history context (product from prev msg)

    If this is a new order, we should NOT inherit product/quantity from
    a previous conversation turn — the customer wants something different.
    """
    msg_lower = message.lower().strip()
    return any(trigger in msg_lower for trigger in NEW_ORDER_TRIGGERS)


def _get_relevant_history(session_history: List[dict], current_message: str) -> List[dict]:
    """
    Returns only the RELEVANT portion of session history for entity extraction.

    THE PROBLEM WE'RE SOLVING:
        Jyothika ordered mutton pickle in turn 1-4.
        Then she says "Hello" (greeting).
        Then she says "I want 2 kg" (new order — not mutton pickle).

        Without this filter, the entity extractor reads all 10 turns,
        finds "mutton pickle" in old history, and assumes she wants mutton pickle again.

    THE SOLUTION:
        When a new order is detected ("I want to order...", "I want 2kg"),
        we find the LAST new order trigger in history and only return
        messages AFTER that point. This gives the AI only the current
        order conversation, not old ones.

        For follow-up messages ("Mutton pickle", "2 kgs"), we return
        the recent history (last 6 turns) so the AI can combine entities.

    Args:
        session_history: Full conversation history from DB.
        current_message: The current message being processed.

    Returns:
        Filtered history relevant to the current order context.
    """
    if not session_history:
        return []

    if _is_new_order(current_message):
        # This is a brand new order — don't inherit old context
        # Return empty history so entity extractor only uses current message
        return []

    # For follow-up messages, find where the current order conversation started
    # Look for the last "new order trigger" in history
    last_new_order_index = -1
    for i, msg in enumerate(session_history):
        if msg.get("role") == "user":
            content = msg.get("content", "").lower()
            if any(trigger in content for trigger in NEW_ORDER_TRIGGERS):
                last_new_order_index = i

    if last_new_order_index >= 0:
        # Return only messages from the last new order trigger onwards
        relevant = session_history[last_new_order_index:]
        return relevant
    else:
        # No clear new order start found — return last 6 turns only
        return session_history[-6:] if len(session_history) > 6 else session_history


ENTITY_SYSTEM_PROMPT = """
You are an entity extraction AI for an order tracking platform.

Extract ALL products and quantities from the customer message.
The customer may order ONE or MULTIPLE products in a single message.

Output a JSON array of items. Each item has:
  product_name    — specific product name (string or null)
  quantity_value  — integer amount (integer or null)
  quantity_unit   — unit string (string or null) e.g. "units", "kg", "pieces"

RULES:
1. Reply ONLY with valid JSON array. No explanation, no markdown.
2. Each item MUST have exactly: "product_name", "quantity_value", "quantity_unit".
3. If a field is unknown, set it to null.
4. quantity_value must be integer only — no units in this field.
   "10 units of flood lights" → quantity_value: 10, quantity_unit: "units"
   "5 pieces of gate lights"  → quantity_value: 5,  quantity_unit: "pieces"
5. product_name should be the specific item name only — no quantity, no units.
6. Extract EVERY product mentioned — do not skip any.
7. If quantity is missing for a product, set quantity_value to null.
8. If product is generic (e.g. "lights" alone with no type) → product_name: null.
9. ONLY extract from messages shown — do not invent data.

Single product example:
  "I want to order 5 LED flood lights"
  → [{"product_name": "LED flood light", "quantity_value": 5, "quantity_unit": "units"}]

Multiple products example:
  "I want 10 flood lights and 5 gate lights"
  → [
      {"product_name": "flood lights", "quantity_value": 10, "quantity_unit": "units"},
      {"product_name": "gate lights",  "quantity_value": 5,  "quantity_unit": "units"}
    ]

Missing quantity example:
  "I want 10 flood lights and garden lights"
  → [
      {"product_name": "flood lights",  "quantity_value": 10, "quantity_unit": "units"},
      {"product_name": "garden lights", "quantity_value": null, "quantity_unit": null}
    ]

Follow-up example (customer answering bot's question):
  History: bot asked "how many flood lights?"
  Message: "15 units"
  → [{"product_name": null, "quantity_value": 15, "quantity_unit": "units"}]

Output format — always an array:
[{"product_name": "flood lights", "quantity_value": 10, "quantity_unit": "units"}]
"""


def _build_error_result(customer_message: str, tenant_id: str) -> EntityResult:
    """Returns a safe fallback EntityResult on parse/extraction failure."""
    return EntityResult(
        items             = [OrderItem(product_name=None, quantity_value=None, quantity_unit=None)],
        delivery_date     = _default_delivery_date(),
        invoice_number    = None,
        payment_reference = None,
        missing_entities  = ["product_name", "quantity"],
        raw_text          = customer_message,
        tenant_id         = tenant_id,
    )


async def extract_entities(
    customer_message: str,
    tenant_id:        str,
    session_history:  List[dict] = None,
    force_new_order:  bool = False,
    cached_items:     List = None,  # Pending items from WORKFLOW_PENDING state
) -> EntityResult:
    """
    Extracts ALL product+quantity pairs from message + relevant session history.

    CACHED ITEMS CONTEXT:
        When cached_items is provided (from WORKFLOW_PENDING), the LLM knows
        which products are already in context. This allows it to correctly
        interpret messages like "I want 2 units each" or "I want 2 units"
        when multiple products are pending — without any hardcoded keywords.

    MULTI-PRODUCT:
        Returns EntityResult with items list — one OrderItem per product.
        "10 flood lights and 5 gate lights" → 2 items
        "I want flood lights" → 1 item (qty=None, will ask)
    """
    raw = ""
    try:
        if force_new_order:
            relevant_history = []
        else:
            relevant_history = _get_relevant_history(
                session_history or [],
                current_message = customer_message,
            )

        # Build dynamic system prompt — include pending products context if available
        if cached_items:
            pending_context = "\n".join([
                f"  - {item.product_name or 'Unknown product'} (qty: {item.quantity_value or 'not specified'})"
                for item in cached_items
            ])
            system_prompt = ENTITY_SYSTEM_PROMPT + f"""

IMPORTANT CONTEXT — Products currently pending in this order:
{pending_context}

If the customer provides a quantity without specifying products (e.g. "I want 2 units",
"2 units", "I want 2 of each"), extract quantity for EACH pending product above.
Return one item per pending product, each with the provided quantity.

Example:
  Pending: [Aeris Gate Light, Villa Gate Light]
  Customer: "I want 2 units"
  → [{{"product_name": "Aeris Gate Light", "quantity_value": 2, "quantity_unit": "units"}},
     {{"product_name": "Villa Gate Light",  "quantity_value": 2, "quantity_unit": "units"}}]
"""
        else:
            system_prompt = ENTITY_SYSTEM_PROMPT

        messages = [{"role": "system", "content": system_prompt}]
        if relevant_history:
            messages.extend(relevant_history)
        messages.append({
            "role":    "user",
            "content": f"[tenant: {tenant_id}]\n{customer_message}"
        })

        response = _client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 500,   # increased for multi-product responses
            temperature = 0,
            messages    = messages,
        )

        raw    = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        # Handle both array and single-object responses
        if isinstance(parsed, dict):
            parsed = [parsed]

        items = []
        for p in parsed:
            qty_val = p.get("quantity_value")
            if qty_val is not None:
                try:
                    qty_val = int(qty_val)
                except (ValueError, TypeError):
                    qty_val = None

            # Only keep product_name if it looks like a valid SKU
            # If customer sent a product name (e.g. "solar lights", "flood lights")
            # set product_name=None so _generate_follow_up asks for SKU instead
            raw_product = p.get("product_name")
            from db.product_store import _is_sku
            product_name = raw_product if (raw_product and _is_sku(raw_product)) else None

            if raw_product and not product_name:
                print(f"[ENTITY] '{raw_product}' is not a SKU → setting product_name=None")

            items.append(OrderItem(
                product_name   = product_name,
                quantity_value = qty_val,
                quantity_unit  = p.get("quantity_unit"),
            ))

        if not items:
            items = [OrderItem(product_name=None, quantity_value=None, quantity_unit=None)]

        # Build missing_entities — flattened across all items
        missing_entities = []
        for i, item in enumerate(items):
            prefix = f"item_{i+1}_" if len(items) > 1 else ""
            if not item.product_name:
                missing_entities.append(f"{prefix}product_name")
            if item.quantity_value is None:
                missing_entities.append(f"{prefix}quantity")

        delivery_date = _default_delivery_date()

        print(
            f"[ENTITY] force_new={force_new_order} history={len(relevant_history)} turns "
            f"items={len(items)} missing={missing_entities}"
        )
        for i, item in enumerate(items):
            print(f"  item[{i}]: product={item.product_name} qty={item.quantity_value} {item.quantity_unit}")

        return EntityResult(
            items             = items,
            delivery_date     = delivery_date,
            invoice_number    = None,
            payment_reference = None,
            missing_entities  = missing_entities,
            raw_text          = customer_message,
            tenant_id         = tenant_id,
        )

    except json.JSONDecodeError as e:
        print(f"[ENTITY] JSON parse error: {e} | raw='{raw}'")
        return _build_error_result(customer_message, tenant_id)
    except Exception as e:
        print(f"[ENTITY ERROR] {e}")
        return _build_error_result(customer_message, tenant_id)