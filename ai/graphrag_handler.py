# ai/graphrag_handler.py — GraphRAG API integration and product list handling
#
# Extracted from main.py to keep the orchestrator lightweight.
# Contains: call_graphrag_api, _send_structured_product_list, _coerce_pythonic_dict
# All imports must be explicit — no globals from main.py.

import ast
import json
import re
import time
import httpx
from typing import Optional

from openai import AzureOpenAI
from config import (
    GRAPHRAG_API_URL,
    AZURE_AI_ENDPOINT, AZURE_AI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT, AZURE_AI_API_VERSION,
)
from adapter.whatsapp_adapter import send_whatsapp_reply, send_whatsapp_image
from ai.product_followup import _try_resolve_product_followup
from db.session_store import (
    save_graphrag_product_selection,
    save_product_api_responses_batch,
    save_tenant_offers,
    save_last_discussed_product,
    save_outbound_message,
    get_graphrag_product_selection,
    get_negotiation_state,
)

_client = AzureOpenAI(
    azure_endpoint = AZURE_AI_ENDPOINT,
    api_key        = AZURE_AI_API_KEY,
    api_version    = AZURE_AI_API_VERSION,
    timeout        = 30.0,
    max_retries    = 0,
)


async def _send_structured_product_list(incoming, products: list) -> str:
    """
    Builds and sends the full product-list response: caches products,
    saves the selection for follow-up picking, sends image cards for the
    first 3 products, and returns the numbered text summary.

    Extracted so it can be reused both for the initial GraphRAG response
    AND for a successful retry response — fixes a bug where a successful
    retry with real products silently fell through to returning the
    ORIGINAL error text instead of ever rendering the retried products.
    """
    print(f"[GRAPHRAG] Got {len(products)} products from structured response")

    if len(products) == 1:
        try:
            from db.session_store import save_last_discussed_product
            pname = products[0].get("name") or products[0].get("product_name")
            if pname:
                await save_last_discussed_product(incoming.tenant_id, incoming.session_id, pname)
        except Exception as e:
            print(f"[GRAPHRAG] Failed to save single product context: {e}")

    try:
        _t_cache_save_start = time.monotonic()
        batch_items = []
        for p in products:
            sku = p.get("sku")
            if sku:
                cached_item = [{
                    "product_name":               p.get("name"),
                    "list_price":                 float(p.get("price_num", 0)),
                    "sku":                        sku,
                    "image_url":                  p.get("image_url"),
                    "installation_url":           p.get("installation_url"),
                    "product_url":                p.get("url"),
                    "discount_pct":               p.get("discount_percentage", 0),
                    "regular_price":              p.get("regular_price", p.get("price_num", 0)),
                    "features":                   [],
                    "specs":                      [],
                    "review_count":               p.get("review_count", 0),
                    "rating":                     p.get("rating", 0),
                    "policies":                   [],
                    "faqs":                       [],
                    "warranties":                 [],
                    "warranty":                   p.get("warranty", ""),
                    "replacement_exchange_policy": p.get("replacement_exchange_policy", ""),
                    "feature_descriptions":       p.get("feature_descriptions", ""),
                    "global_offers":              p.get("global_offers", ""),
                }]
                batch_items.append({"sku": sku, "api_response": cached_item})

        from db.session_store import save_product_api_responses_batch
        await save_product_api_responses_batch(incoming.tenant_id, batch_items)
        print(f"[TIMING] Product cache batch save ({len(batch_items)} products): {time.monotonic() - _t_cache_save_start:.2f}s")
    except Exception as e:
        print(f"[GRAPHRAG] Cache save failed (non-critical): {e}")

    try:
        await save_graphrag_product_selection(
            tenant_id  = incoming.tenant_id,
            session_id = incoming.session_id,
            products   = products,
        )
        print(f"[GRAPHRAG] Product selection saved to workflow_sessions")
    except Exception as e:
        print(f"[GRAPHRAG] Selection save failed (non-critical): {e}")

    try:
        _go = next((p.get("global_offers") for p in products if p.get("global_offers")), None)
        if _go:
            await save_tenant_offers(tenant_id=incoming.tenant_id, offers_text=_go)
    except Exception as e:
        print(f"[GRAPHRAG] tenant_offers save failed (non-critical): {e}")

    MAX_IMAGE_PRODUCTS = 3
    for i, p in enumerate(products, 1):
        if i > MAX_IMAGE_PRODUCTS:
            break

        img_url   = p.get("image_url")
        name      = p.get("name", "Product")
        price     = p.get("price_num", 0)
        reg_price = p.get("regular_price", price)
        discount  = p.get("discount_percentage", 0)
        rating    = p.get("rating", 0)
        reviews   = p.get("review_count", 0)

        caption = f"{i}. {name}\nRs.{float(price):,.0f}"
        if discount:
            caption += f" (Save {discount}% off Rs.{float(str(reg_price).replace(',','')):,.0f})"
        if rating:
            caption += f"\n⭐ {rating} ({reviews} reviews)"

        if img_url:
            img_wamid = await send_whatsapp_image(incoming.session_id, img_url, caption)
            if img_wamid:
                print(f"[GRAPHRAG] Image sent for product {i}: {name} — wamid={img_wamid}")
                await save_outbound_message(
                    tenant_id     = incoming.tenant_id,
                    session_id    = incoming.session_id,
                    message_id    = img_wamid,
                    text          = caption,
                    media_url     = img_url,
                    original_type = "image",
                    region        = incoming.region,
                )
        else:
            reply_wamid = await send_whatsapp_reply(incoming.session_id, caption)
            if reply_wamid:
                print(f"[GRAPHRAG] No image for product {i}: {name} — sent text card wamid={reply_wamid}")
                await save_outbound_message(
                    tenant_id  = incoming.tenant_id,
                    session_id = incoming.session_id,
                    message_id = reply_wamid,
                    text       = caption,
                    region        = incoming.region,
                )

    lines = [f"Here are the options for you, {incoming.sender_name}! 💡\n"]
    for i, p in enumerate(products, 1):
        name      = p.get("name", "Product")
        price     = p.get("price_num", 0)
        reg_price = p.get("regular_price", price)
        discount  = p.get("discount_percentage", 0)
        if i <= MAX_IMAGE_PRODUCTS:
            entry = f"*{i}.* {name} — Rs.{float(price):,.0f}"
            if discount:
                entry += f" (Save {discount}% off Rs.{float(str(reg_price).replace(',','')):,.0f})"
            lines.append(entry)
        else:
            lines.append(f"*{i}.* {name} — Rs.{float(price):,.0f}")

    lines.append(
        f"\nReply with the product *number* or *name* to know more or place an order."
    )

    summary_text = "\n".join(lines)
    if len(summary_text) > 4096:
        summary_text = summary_text[:4090] + "\n…"

    return summary_text


def _coerce_pythonic_dict(value):
    """
    GraphRAG is expected to return structured shapes (list of product dicts,
    or a {"status": "needs_clarification", ...} dict) as real JSON.

    In production we've seen it instead return that SAME dict already
    stringified on GraphRAG's side (Python's str(dict) — single quotes,
    not valid JSON) inside response_text. Because that arrives as a plain
    str, `isinstance(response_text, dict)` below is False, every structured
    check is skipped, and the literal Python dict text gets sent to the
    customer verbatim.

    This safely converts a string that LOOKS like a Python dict literal
    back into a real dict so the existing needs_clarification / product-list
    handling below can catch it. Anything that isn't a clean dict literal
    is returned unchanged — never raises, never guesses.
    """
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, SyntaxError):
                pass
    return value


async def call_graphrag_api(incoming, session_history: list = None) -> str:
    """
    Calls the Hybrid RAG Agent API for ALL product-related queries.

    HANDLES:
        - Product browsing by category: "I want garden lights"
        - Product follow-up questions: "is it aluminum?", "what's the warranty?"
        - Ordering: "I want 2 units of Reva"
        - Picking from a list: "1", "the second one", "I want Romy"

    RESPONSE FORMAT from GraphRAG API:
        {
          "status": "routed_to_knowledge_base",
          "response_text": [
            {"sku": "12C-2080", "name": "Reva LED Garden Bollard",
             "price_num": 2653, "image_url": "...", "rating": 4.87,
             "review_count": 55, "feature_descriptions": "...", ...},
            ...
          ]
        }

    FLOW:
        1. Check if customer is following up on a previously shown product list
           (PRODUCT_SELECTION in DB) — resolve "1", "Reva", "is it aluminum?"
        2. Call GraphRAG API → get list of matching products
        3. Save each product to product_cache (24hr TTL)
        4. Send product image + caption for each product
        5. Send numbered text list so customer can pick
        6. Save all products to workflow_sessions PRODUCT_SELECTION (20min)
           so next message like "1" or "is it waterproof?" can be resolved
    """
    try:
        # ── Pre-check: is this a follow-up about a previously shown product? ──
        # If customer already saw a numbered list and is asking "is it aluminum?"
        # or "tell me more about Romy" — resolve that before calling GraphRAG.
        if session_history:
            follow_up_reply = await _try_resolve_product_followup(incoming, session_history)
            if follow_up_reply == "__ALREADY_HANDLED__":
                # Image/link/installation already sent directly to WhatsApp —
                # return empty string so the outer pipeline sends nothing more,
                # but does NOT fall through to GraphRAG.
                return ""
            if follow_up_reply:
                return follow_up_reply

        # ── Send original query to GraphRAG ──────────────────────────────────
        # GraphRAG uses Neo4j semantic search which understands natural language.
        # We send the customer's original message as-is — no stripping, no cleaning.
        # "i want to order outdoor lights?" → GraphRAG receives exactly this.
        graphrag_text = incoming.text

        # Only handle quote-reply prefix — strip [Quoting:...] to get actual message
        if graphrag_text.startswith("[Quoting:") and "\n" in graphrag_text:
            actual_msg = graphrag_text.split("\n", 1)[1].strip()
            if actual_msg:
                print(f"[GRAPHRAG] Quote-reply — using actual message: '{actual_msg[:60]}'")
                graphrag_text = actual_msg

        # ── Build payload matching messages table schema ───────────────────
        payload = {
            "id":                  incoming.message_id,
            "tenant_id":           incoming.tenant_id,
            "message_id":          incoming.message_id,
            "session_id":          incoming.session_id,
            "channel":             incoming.channel,
            "timestamp_unix":      incoming.timestamp,
            "region":              incoming.region,
            "original_type":       incoming.original_type,
            "text":                graphrag_text,
            "intent":              "FAQ_KNOWLEDGE",
            "confidence":          0.95,
            "product_name":        None,
            "quantity_value":      None,
            "quantity_unit":       None,
            "delivery_date":       None,
            "missing_entities":    [],
            "reply_text":          None,
            "replied_at":          None,
            "sender_name":         incoming.sender_name,
            "sender_phone_number": incoming.sender_phone,
            "trace_id":            incoming.trace_id,
            "received_at":         incoming.received_at,
            "direction":           "inbound",
            "invoice_number":      None,
            "payment_reference":   None,
        }

        print(f"[GRAPHRAG] Calling {GRAPHRAG_API_URL} for: '{graphrag_text[:60]}'")

        # GraphRAG uses LangChain + Neo4j — can take 40-60 seconds
        graphrag_timeout = httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=graphrag_timeout) as client:
            response = await client.post(
                GRAPHRAG_API_URL,
                json    = payload,
                headers = {"Content-Type": "application/json"},
            )

        if response.status_code == 403:
            print(f"[GRAPHRAG] 403 — host not whitelisted")
            support = getattr(incoming, 'support_email', None) or incoming.biz_name
            return (
                f"Thanks for your interest, {incoming.sender_name}! 😊\n\n"
                f"I'm having trouble fetching product information right now.\n"
                f"Please contact *{support}* for assistance."
            )

        if response.status_code != 200:
            print(f"[GRAPHRAG] HTTP {response.status_code}")
            support = getattr(incoming, 'support_email', None) or incoming.biz_name
            return (
                f"I'm having trouble fetching product information right now, "
                f"{incoming.sender_name}. 🔧\n\n"
                f"Please try again shortly or contact *{support}*"
            )

        data = response.json()
        print(f"[GRAPHRAG] Response received — keys: {list(data.keys()) if isinstance(data, dict) else 'list'}")

        # Store raw response on incoming so pipeline can save it to DB
        try:
            incoming._graphrag_raw = json.dumps(data, ensure_ascii=False)
        except Exception:
            incoming._graphrag_raw = str(data)

        response_text = data.get("response_text", [])
        response_text = _coerce_pythonic_dict(response_text)

        # ── Clarification request response ──────────────────────────────────
        # GraphRAG can return a THIRD response shape: a dict with
        # "status": "needs_clarification" and "available_collections" — this
        # happens when a query (e.g. "outdoor lights") matches products
        # spanning multiple distinct collections and GraphRAG wants the
        # customer to narrow down which one they mean.
        #
        # BUG FIXED: previously this dict fell through to str(response_text)
        # and got sent to the customer VERBATIM as raw Python dict syntax
        # (e.g. "{'status': 'needs_clarification', 'message': ...}") —
        # confirmed in production screenshots. Now it's rendered as a
        # clean, friendly numbered list instead.
        if isinstance(response_text, dict) and response_text.get("status") == "needs_clarification":
            collections = response_text.get("available_collections", [])
            clarify_msg = response_text.get(
                "message",
                "Could you let me know which category you're interested in?"
            )
            print(f"[GRAPHRAG] Needs clarification — {len(collections)} collections offered")

            lines = [f"Hi {incoming.sender_name}! {clarify_msg}"]
            if collections:
                lines.append("")
                for i, c in enumerate(collections, 1):
                    lines.append(f"*{i}.* {c}")
                lines.append("")
                lines.append("Just reply with the collection name and I'll show you the options! 💡")

            return "\n".join(lines)

        # ── Structured product list response ──────────────────────────────
        if isinstance(response_text, list) and response_text and isinstance(response_text[0], dict):
            return await _send_structured_product_list(incoming, response_text)

        # ── Plain text / string response ───────────────────────────────────
        # CRITICAL: response_text can be an empty list [] when GraphRAG finds
        # zero matching products. An empty list is falsy in Python, so the old
        # `if response_text else str(data)` fallback incorrectly stringified
        # the ENTIRE raw API payload (status, tenant_id, message_id, etc.) and
        # sent that directly to the customer as a WhatsApp message. Fixed:
        # explicitly check for the empty-list case and reply with a clean,
        # friendly message instead of ever exposing raw API internals.
        if isinstance(response_text, list) and len(response_text) == 0:
            print(f"[GRAPHRAG] Empty product list — no matches found")
            return (
                f"Sorry {incoming.sender_name}, I couldn't find any products matching that. "
                f"Could you try describing it differently, or browse all products at {incoming.website or incoming.biz_name}? 💡"
            )

        reply_str = str(response_text).strip() if response_text else str(data)
        print(f"[GRAPHRAG] Plain text reply — {len(reply_str)} chars")

        # If GraphRAG returned a short error message (≤100 chars), retry once
        # with an even simpler query — just the last 1-2 words as keywords
        if len(reply_str) <= 100 and ("error" in reply_str.lower() or "sorry" in reply_str.lower()):
            print(f"[GRAPHRAG] API error detected — retrying with simplified query")
            words = [w for w in graphrag_text.split() if len(w) > 3]
            simple_query = " ".join(words[-2:]) if words else graphrag_text
            if simple_query and simple_query != graphrag_text:
                print(f"[GRAPHRAG] Retry query: '{simple_query}'")
                payload["text"] = simple_query
                try:
                    async with httpx.AsyncClient(timeout=graphrag_timeout) as retry_client:
                        retry_resp = await retry_client.post(
                            GRAPHRAG_API_URL,
                            json    = payload,
                            headers = {"Content-Type": "application/json"},
                        )
                    if retry_resp.status_code == 200:
                        retry_data = retry_resp.json()
                        retry_text = retry_data.get("response_text", [])
                        retry_text = _coerce_pythonic_dict(retry_text)
                        if isinstance(retry_text, list) and retry_text and isinstance(retry_text[0], dict):
                            print(f"[GRAPHRAG] Retry succeeded — {len(retry_text)} products")
                            # BUG FIX: previously this only set response_text with a comment
                            # "fall through to handling below" — but no such handling existed
                            # after this point, so the retry's real products were silently
                            # discarded and the ORIGINAL error text was returned instead.
                            return await _send_structured_product_list(incoming, retry_text)
                        elif isinstance(retry_text, dict) and retry_text.get("status") == "needs_clarification":
                            collections = retry_text.get("available_collections", [])
                            clarify_msg = retry_text.get(
                                "message",
                                "Could you let me know which category you're interested in?"
                            )
                            lines = [f"Hi {incoming.sender_name}! {clarify_msg}"]
                            if collections:
                                lines.append("")
                                for i, c in enumerate(collections, 1):
                                    lines.append(f"*{i}.* {c}")
                                lines.append("")
                                lines.append("Just reply with the collection name and I'll show you the options! 💡")
                            return "\n".join(lines)
                        elif isinstance(retry_text, str) and len(retry_text) > 100:
                            reply_str = retry_text
                except Exception as retry_err:
                    print(f"[GRAPHRAG] Retry failed: {retry_err}")

            # If we still have the original short error/sorry text (retry didn't
            # produce usable products or a longer message), never expose GraphRAG's
            # raw error string to the customer — replace with a friendly message.
            if len(reply_str) <= 100 and ("error" in reply_str.lower() or "sorry" in reply_str.lower()):
                print(f"[GRAPHRAG] Retry did not resolve the error — sending friendly fallback")
                return (
                    f"Sorry {incoming.sender_name}, I'm having trouble finding that right now. "
                    f"Could you try rephrasing, or browse all products at {incoming.website or incoming.biz_name}? 💡"
                )

        if len(reply_str) <= 4096:
            return reply_str

        # Split long plain text reply at line boundaries
        chunks  = []
        lines   = reply_str.split("\n")
        current = ""
        for line in lines:
            candidate = current + "\n" + line if current else line
            if len(candidate) > 3800:
                if current:
                    chunks.append(current.strip())
                if len(line) > 3800:
                    while len(line) > 3800:
                        chunks.append(line[:3800])
                        line = line[3800:]
                    current = line
                else:
                    current = line
            else:
                current = candidate
        if current.strip():
            chunks.append(current.strip())
        if not chunks:
            chunks = [reply_str[i:i+3800] for i in range(0, len(reply_str), 3800)]

        print(f"[GRAPHRAG] Split into {len(chunks)} message(s)")
        return "\n\n⟨MSG_SPLIT⟩\n\n".join(chunks)

    except Exception as e:
        import traceback
        print(f"[GRAPHRAG] Error: {type(e).__name__}: {e}")
        print(f"[GRAPHRAG] Traceback: {traceback.format_exc()[-300:]}")
        support = getattr(incoming, 'support_email', None) or incoming.biz_name
        website = getattr(incoming, 'website', None) or ""
        return (
            f"Thanks for your interest in our products, {incoming.sender_name}! 💡\n\n"
            f"Our product search is temporarily unavailable. Meanwhile:\n\n"
            + (f"• Browse all products at *{website}*\n" if website else "")
            + f"\nNeed help? Contact *{support}*"
        )