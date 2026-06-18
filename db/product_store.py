# db/product_store.py — Product Price Lookup + Order Storage
#
# PRICE LOOKUP ROUTING (3 routes, in order):
#   1. SKU detected      → check product_cache DB → if miss → call Products API → save to cache
#   2. Full product name → check product_cache by name → get real API price
#   3. Both fail         → return None (Supabase products table REMOVED)
#
# NOTE: Supabase `products` table has been dropped.
#       All prices now come exclusively from colleague's Products API
#       cached in the product_cache table.

import re
import uuid
import httpx
from datetime import datetime, timezone
from typing import Optional
from supabase import create_client, Client  # type: ignore[import]
from config import SUPABASE_URL, SUPABASE_SERVICE_KEY, PRODUCTS_API_URL

_supabase: Optional[Client] = None

def _get_client() -> Client:
    global _supabase
    if _supabase is None:
        _supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase


def _generate_order_id() -> str:
    """Generates unique order ID: INV#XXXXX"""
    return f"INV#{uuid.uuid4().hex[:5].upper()}"


# ── SKU Detection ─────────────────────────────────────────────────────────────

def _is_sku(text: str) -> bool:
    """
    Detects if a string looks like a product SKU.

    Rules (all must pass):
    1. Length: 4–15 characters
    2. No spaces
    3. Must contain at least one digit — rejects English words (WANT, GIVE etc.)
    4. Must contain at least one letter — rejects pure numbers
    5. Only alphanumeric + dash + parentheses allowed

    Valid:   10C-2012 ✅  12M-2014B ✅  SLR-W50Y ✅  LOS06Y(M1) ✅  ALT20C ✅
    Invalid: WANT ❌  GIVE ❌  "gate lights" ❌
    """
    text = text.strip()
    if len(text) > 15 or len(text) < 4:
        return False
    if ' ' in text:
        return False
    if not any(c.isdigit() for c in text):
        return False
    if not any(c.isalpha() for c in text):
        return False
    allowed = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-()')
    if not all(c in allowed for c in text):
        return False
    return True


def _extract_skus_from_text(text: str) -> list:
    """
    Extracts ALL SKUs from a message text.

    IMPORTANT: Only strips leading punctuation (.,!?:;\"') — NOT parentheses.
    Parentheses are part of valid SKUs like LOS06Y(M1).

    Returns list of unique SKUs found, preserving order.
    """
    detected = []
    seen     = set()
    # Strip only non-SKU punctuation from word boundaries
    # Do NOT strip ( or ) as they are valid SKU characters
    strip_chars = ".,!?:;\"\' "
    for word in text.upper().split():
        clean = word.strip(strip_chars)
        if clean and _is_sku(clean) and clean not in seen:
            detected.append(clean)
            seen.add(clean)
    return detected


# ── Products API ──────────────────────────────────────────────────────────────

async def _fetch_from_products_api(skus: list) -> list:
    """
    Calls colleague's Products API with one or more SKUs.

    POST {PRODUCTS_API_URL}
    Body: {"skus": ["24C-2055", "LOS06Y(M1)"]}

    Returns list of product dicts. Empty list if API blocked or fails.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                PRODUCTS_API_URL,
                json={"skus": [s.upper() for s in skus]},
                headers={"Content-Type": "application/json"},
            )

        if response.status_code == 403:
            print(f"[PRODUCTS API] 403 Host not whitelisted")
            return []

        if response.status_code != 200:
            print(f"[PRODUCTS API] HTTP {response.status_code}")
            return []

        data     = response.json()
        products = data.get("products", [])

        results = []
        for p in products:
            price = p.get("price")
            name  = p.get("name")
            sku   = p.get("sku")
            if not price or not name:
                continue
            results.append({
                "product_name":  name,
                "list_price":    float(price),
                "floor_price":   float(price) * 0.85,
                "sku":           sku,
                "image_url":     p.get("image_url"),
                "product_url":   p.get("url"),
                "discount_pct":  p.get("discount_percentage", 0),
                "regular_price": p.get("regular_price", price),
                "categories":    p.get("categories", []),
                "features":      p.get("features", []),
                "specs":         p.get("specs", []),
                "use_cases":     p.get("use_cases", []),
                "review_count":  p.get("review_count", 0),
                "warranties":    p.get("warranties", []),
                "policies":      p.get("policies", []),
                "faqs":          p.get("faqs", []),
            })
            print(f"[PRODUCTS API] {sku} -> '{name}' @ Rs.{price}")

        return results

    except Exception as e:
        print(f"[PRODUCTS API] Error: {e}")
        return []


async def get_product_price(
    tenant_id:    str,
    product_name: str,
) -> Optional[dict]:
    """
    Looks up price for a product.

    SKU-ONLY: Only SKU codes are supported (e.g. 10C-2012, ALT20C).
    If product_name is not a SKU, returns None.
    Caller (_generate_follow_up) will ask customer to provide SKU.
    """

    # SKU only — reject non-SKU product names immediately
    if not _is_sku(product_name):
        print(f"[PRODUCT] '{product_name}' is not a SKU — returning None")
        return None

    # SKU → check DB cache first, then call API
    print(f"[PRODUCT] SKU: '{product_name}' -> checking DB cache")
    try:
        from db.session_store import get_product_api_response, save_product_api_response
        cached = await get_product_api_response(tenant_id, product_name)
        if cached:
            print(f"[PRODUCT] Loaded from DB cache — SKU={product_name}")
            return cached[0]
    except Exception as e:
        print(f"[PRODUCT] DB cache check failed: {e}")

    # Not in cache — call API
    print(f"[PRODUCT] Not in cache — calling Products API for SKU={product_name}")
    results = await _fetch_from_products_api([product_name])
    if results:
        try:
            from db.session_store import save_product_api_response
            await save_product_api_response(tenant_id, product_name, results)
        except Exception as e:
            print(f"[PRODUCT] Cache save failed: {e}")
        return results[0]

    print(f"[PRODUCT] API returned nothing for SKU '{product_name}'")
    return None


async def create_order(
    tenant_id:   str,
    session_id:  str,
    sender_name: str,
    items:       list,
) -> Optional[dict]:
    """
    Creates a confirmed order with one or more line items.
    Inserts header into orders + line items into order_items.
    """
    try:
        order_id       = _generate_order_id()
        total_price    = round(sum(
            float(item["unit_price"]) * int(item["quantity_value"])
            for item in items
        ), 2)
        total_with_gst = round(total_price * 1.18, 2)

        order_row = {
            "order_id":       order_id,
            "tenant_id":      tenant_id,
            "session_id":     session_id,
            "sender_name":    sender_name,
            "product_name":   items[0]["product_name"] if len(items) == 1 else f"{len(items)} products",
            "quantity_value": items[0]["quantity_value"] if len(items) == 1 else None,
            "quantity_unit":  items[0]["quantity_unit"]  if len(items) == 1 else None,
            "unit_price":     items[0]["unit_price"]     if len(items) == 1 else None,
            "total_price":    total_price,
            "total_with_gst": total_with_gst,
            "items_count":    len(items),
            "status":         "CONFIRMED",
        }
        print(f"[ORDER] Inserting header {order_id} - {len(items)} item(s)")
        result = _get_client().table("orders").insert(order_row).execute()

        if not result.data:
            print(f"[ORDER] Header insert returned no data")
            return None

        item_rows = []
        for item in items:
            item_total = round(float(item["unit_price"]) * int(item["quantity_value"]), 2)
            item_rows.append({
                "order_id":       order_id,
                "tenant_id":      tenant_id,
                "product_name":   item["product_name"],
                "quantity_value": item["quantity_value"],
                "quantity_unit":  item.get("quantity_unit"),
                "unit_price":     item["unit_price"],
                "total_price":    item_total,
            })
        print(f"[ORDER] Inserting {len(item_rows)} order_items rows")
        _get_client().table("order_items").insert(item_rows).execute()

        order = result.data[0]
        order["items"] = item_rows

        print(f"[ORDER] Created {order_id} - total=Rs.{total_price} + GST=Rs.{total_with_gst}")
        for item in items:
            print(f"  -> {item['product_name']} x {item['quantity_value']} @ Rs.{item['unit_price']}")

        return order

    except Exception as e:
        print(f"[ORDER] Create order failed: {e}")
        return None


async def get_order_by_id(order_id: str, tenant_id: str) -> Optional[dict]:
    """Fetches a specific order by order_id."""
    try:
        result = _get_client().table("orders") \
            .select("*") \
            .eq("order_id", order_id) \
            .eq("tenant_id", tenant_id) \
            .limit(1) \
            .execute()
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"[ORDER] Fetch by ID failed: {e}")
        return None