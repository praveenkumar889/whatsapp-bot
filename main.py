# ═════════════════════════════════════════════════════════════════════════════
# main.py — FastAPI Application Entry Point
#
# WHAT THIS FILE IS:
#   The brain of the entire WhatsApp AI system.
#   Every customer message flows through this file from receipt to reply.
#
# ARCHITECTURE:
#   This file is the ORCHESTRATOR — it calls other modules but does not
#   contain business logic itself. Each step delegates to a specialist:
#
#   adapter/whatsapp_adapter.py  → parse raw Meta JSON → IncomingMessage
#   ai/intent_router.py          → classify what customer wants
#   db/session_store.py          → all Supabase DB operations
#
# PIPELINE (9 steps — fully GPT-driven, zero hardcoded keyword lists):
#   Step 1  — Parse webhook          → translate Meta JSON to IncomingMessage
#   Step 2  — Resolve tenant_id      → find which business owns this number
#   Step 3  — Deduplicate            → skip if already processed
#   Step 3.5— Rate limit             → skip if same session already processing
#   Step 4  — Fetch session history  → last 10 messages for AI context
#   Step 5  — Save message           → Save-First rule
#   Step 6  — Classify intent        → FAQ_KNOWLEDGE | GREETING | ESCALATION | UNKNOWN
#   Step 7  — Update intent in DB    → store classification result
#   Step 8  — Route to handler       → correct handler based on intent
#   Step 9  — Send reply + store     → POST to Meta + audit trail
#
# ALL PRODUCT QUERIES (Step 8):
#   Every product-related message (browse, info, order) routes to call_graphrag_api()
#   which calls the Hybrid RAG Agent API (GraphRAG + Neo4j).
#   The API handles natural language search, product details, and ordering guidance.
# ═════════════════════════════════════════════════════════════════════════════

import asyncio
import ast
import json
import re
import time
import httpx
from typing import Optional
from datetime import datetime, timezone, timedelta
from openai import AzureOpenAI
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse

from config import (
    VERIFY_TOKEN, ACCESS_TOKEN, PHONE_NUMBER_ID,
    AZURE_AI_ENDPOINT, AZURE_AI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT, AZURE_AI_API_VERSION,
    GRAPHRAG_API_URL,
)
from adapter.whatsapp_adapter import parse_webhook
from ai.intent_router import classify_intent
from ai.negotiator import (
    is_negotiation_request,
    handle_negotiation,
)
from db.session_store import (
    resolve_tenant_id,
    is_duplicate,
    get_session_history,
    save_message,
    update_intent,
    update_reply,
    save_product_api_response,
    get_product_api_response,
    get_cached_product_by_name,
    save_graphrag_product_selection,
    get_graphrag_product_selection,
    save_outbound_message,
    save_negotiation_state,
    get_negotiation_state,
    clear_negotiation_state,
    save_tenant_offers,
    get_tenant_offers,
)
from db.processing_lock import acquire_lock, release_lock, cleanup_stale_locks

app = FastAPI(title="Order Tracking AI — WhatsApp Webhook")

# ── Concurrency guard ─────────────────────────────────────────────────────────
# Limits simultaneous background tasks to 50.
# WHY: Without this, 500 users at once = 500 concurrent DB connections.
#      Supabase free tier maxes at ~60 connections → all queries fail.
#      Tasks above the limit queue and WAIT — they are NOT dropped.
#      Customer still gets a reply, just a few seconds later.
_pipeline_semaphore = asyncio.Semaphore(50)

# ── Shared Azure OpenAI client ─────────────────────────────────────────────────
# timeout=30s: Azure default is 600s — a hung LLM call holds the session lock
# for 10 minutes, blocking all messages from that user.
# max_retries=0: we handle failures ourselves with friendly fallback replies.
_ai_client = AzureOpenAI(
    azure_endpoint = AZURE_AI_ENDPOINT,
    api_key        = AZURE_AI_API_KEY,
    api_version    = AZURE_AI_API_VERSION,
    timeout        = 30.0,
    max_retries    = 0,
)

# ── Periodic lock cleanup ─────────────────────────────────────────────────────
# Runs every 60s as a background task instead of on every request.
# Previously cleanup_stale_locks() added DB latency to every single message.
async def _periodic_lock_cleanup():
    while True:
        try:
            await asyncio.sleep(60)
            await cleanup_stale_locks()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[CLEANUP] Periodic cleanup error: {e}")

@app.on_event("startup")
async def startup():
    asyncio.create_task(_periodic_lock_cleanup())
    print("[STARTUP] Periodic lock cleanup task started")


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/webhook")
async def verify_webhook(request: Request):
    """
    Meta webhook verification handshake.
    Runs once during initial setup when you click "Verify and Save"
    in Meta Developer Portal. Never called again in normal operation.
    """
    params    = dict(request.query_params)
    mode      = params.get("hub.mode")
    token     = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("[WEBHOOK] Verified by Meta")
        return PlainTextResponse(content=challenge)
    return PlainTextResponse(content="Forbidden", status_code=403)


@app.post("/webhook")
async def receive_message(request: Request):
    """
    Receives every WhatsApp message from Meta Cloud API.
    Returns HTTP 200 IMMEDIATELY to Meta (within milliseconds),
    then processes the message as a background task.

    WHY BACKGROUND TASK:
        Meta requires a response within 5 seconds.
        Our pipeline (DB + LLM calls + GraphRAG) can take 2-10 seconds.
        If we process synchronously and Meta times out → it retries →
        customer gets duplicate replies.
    """
    data = await request.json()
    async def _guarded():
        async with _pipeline_semaphore:
            await process_message(data)
    asyncio.create_task(_guarded())
    return JSONResponse(content={"status": "ok"})


# ══════════════════════════════════════════════════════════════════════════════
# CORE PIPELINE — process_message()
# ══════════════════════════════════════════════════════════════════════════════

async def process_message(data: dict):
    """
    The full 9-step message processing pipeline.
    Called as a background task for every inbound WhatsApp message.
    Never raises — customer always gets a reply even if individual steps fail.
    """
    _t_pipeline_start = time.monotonic()

    # ── Step 1: Parse webhook ──────────────────────────────────────────────
    # Translates raw Meta JSON → clean IncomingMessage object.
    # Returns None for delivery/read receipts, stickers — skip those.
    incoming = await parse_webhook(data)
    if not incoming:
        print("[PIPELINE] Skipped — not a text message or parse failed")
        return

    # ── Step 2: Resolve full tenant profile from DB ───────────────────────
    # Fetches ALL tenant fields from tenants table via phone_number_id.
    # ZERO HARDCODING: biz_name, website, support_email, timezone all from DB.
    # New client = insert one row, zero code changes.
    tenant_info = await resolve_tenant_id(incoming.phone_number_id)
    if tenant_info is None:
        print(f"[PIPELINE] Unknown phone_number_id={incoming.phone_number_id} — skipping")
        return

    incoming.tenant_id     = tenant_info["tenant_id"]
    incoming.biz_name      = tenant_info.get("biz_name")      or incoming.biz_name
    incoming.timezone      = tenant_info.get("timezone")      or incoming.timezone
    incoming.region        = tenant_info.get("region")        or incoming.region
    incoming.language      = tenant_info.get("language")      or incoming.language
    incoming.tagline       = tenant_info.get("tagline")
    incoming.city          = tenant_info.get("city")
    incoming.support_email = tenant_info.get("support_email")
    incoming.website       = tenant_info.get("website")
    incoming.upi_id        = tenant_info.get("upi_id")
    incoming.account_name  = tenant_info.get("account_name")
    # GST rate from tenant config (default 18% for LED lighting / standard goods)
    incoming.gst_rate      = float(tenant_info.get("gst_rate") or 18) / 100

    print(f"\n{'─'*60}")
    print(f"[{incoming.trace_id}] {incoming.sender_name} ({incoming.sender_phone})")
    print(f"[TENANT]   {incoming.tenant_id}")
    print(f"[MESSAGE]  {incoming.text}")

    # ── Step 2.5: Resolve quoted message caption ───────────────────────────
    # When a customer swipes a message to quote-reply it, Meta sends a
    # "context" object with the wamid of the quoted message.
    # We look that wamid up in our messages table to get the bot's reply text
    # (which contains the product caption like "1. Reva LED Garden Bollard\nRs.2,653")
    # Then we prepend it to incoming.text so the pipeline has full context:
    #   incoming.text = "[Quoting: 5. Perumal 6W LED Divine Light — Rs.789]\nI want to buy this"
    # This lets the intent router and GraphRAG resolver understand what product
    # the customer is referring to without any special-case logic.
    if incoming.quoted_message_id:
        try:
            from db.session_store import get_reply_by_message_id
            quoted_text = await get_reply_by_message_id(
                tenant_id  = incoming.tenant_id,
                message_id = incoming.quoted_message_id,
            )
            if quoted_text:
                # Trim to first 200 chars — captions can be long, we just need product name
                quoted_preview = quoted_text.strip()[:200]
                incoming.quoted_caption = quoted_preview
                # Prepend quoted context to message text so ALL downstream handlers
                # (intent router, GraphRAG, follow-up resolver) see the full picture
                incoming.text = f"[Quoting: {quoted_preview}]\n{incoming.text}"
                print(f"[ADAPTER] Quoted caption resolved — prepended to message text")
            else:
                print(f"[ADAPTER] Quoted message not found in DB — processing without context")
        except Exception as e:
            print(f"[ADAPTER] Quoted message lookup failed (non-critical): {e}")

    # ── Step 3: Deduplicate ────────────────────────────────────────────────
    # Meta sometimes retries webhook delivery if server was slow.
    # Without this check → same message processed twice → duplicate replies.
    if await is_duplicate(incoming.message_id):
        print(f"[PIPELINE] Duplicate — skipping {incoming.message_id}")
        return

    # ── Step 3.5: Distributed processing lock ─────────────────────────────
    # Prevents same session being processed simultaneously across workers.
    # INSERT into processing_locks table — PRIMARY KEY prevents duplicates.
    # cleanup_stale_locks() runs as a background task every 60s (see startup).
    if not await acquire_lock(incoming.session_id, incoming.tenant_id):
        print(f"[PIPELINE] Session {incoming.session_id} already processing — skipping")
        return

    try:
        # ── Step 4: Fetch session history ──────────────────────────────────
        # Last 10 messages for this customer from DB.
        # WhatsApp has NO history API — every webhook is isolated.
        # History gives AI context: "1" after a numbered list = pick option 1.
        # Format: [{"role": "user", "content": "..."}, {"role": "assistant", ...}]
        session_history = await get_session_history(
            tenant_id  = incoming.tenant_id,
            session_id = incoming.session_id,
            limit      = 10,
        )

        # ── Step 5: Save to DB (Save-First rule) ──────────────────────────
        # Inserts the message into messages table BEFORE processing.
        # If AI call or reply fails, message is still in DB for debugging.
        await save_message(incoming)

        # ── Step 6: Classify intent ────────────────────────────────────────
        # Sends customer message + session history to Azure OpenAI GPT-4.1.
        # Returns: FAQ_KNOWLEDGE | HUMAN_ESCALATION | GREETING | UNKNOWN
        # History context lets AI understand follow-up messages like "1" or "Reva".
        _t_intent_start = time.monotonic()
        result = await classify_intent(incoming.text, session_history)
        print(f"[TIMING] classify_intent: {time.monotonic() - _t_intent_start:.2f}s")
        print(f"[INTENT]   {result.intent}  confidence={result.confidence_score}")

        # ── Step 7: Update intent in DB ────────────────────────────────────
        # Message was saved in Step 5 with intent=NULL.
        # Now fill in the classification result.
        await update_intent(incoming.message_id, result.intent, result.confidence_score)

        # ── Step 8: Route to correct handler ──────────────────────────────
        # FAQ_KNOWLEDGE    → GraphRAG API (all product queries, browsing, ordering)
        # HUMAN_ESCALATION → empathy reply + support contact
        # GREETING         → time-aware greeting from DB timezone
        # UNKNOWN          → helpful fallback with capability list
        #
        # NOTE: All product-related messages (browse, info, order, follow-up)
        #       are handled by call_graphrag_api(). The GraphRAG API handles
        #       natural language search, product details, and ordering guidance.

        # ── PRE-ROUTE GUARD: awaiting_invoice_confirmation ─────────────────
        # When the bot has shown an order summary and is waiting for "Confirm",
        # any message that is NOT a confirmation (e.g. "I want 5890", "can I
        # get a lower price?", "I need a discount") must re-enter the negotiation
        # handler — not fall through to UNKNOWN or HUMAN_ESCALATION.
        #
        # Without this guard those messages hit intent routing:
        #   "I want 5890" → UNKNOWN → "I didn't quite understand"  ← confirmed bug
        #   "I need a discount" → HUMAN_ESCALATION → "team will contact you"
        _pre_neg_state = await get_negotiation_state(incoming.tenant_id, incoming.session_id)
        _awaiting_conf = (
            _pre_neg_state is not None
            and _pre_neg_state.get("awaiting_invoice_confirmation", False)
            and _pre_neg_state.get("quantity")
            and _pre_neg_state.get("last_offer_price")
        )
        if _awaiting_conf:
            _actual_confirm = await _is_invoice_confirmation_request(incoming, session_history)
            if not _actual_confirm:
                # Message is not a confirmation — treat as continued negotiation
                print(f"[NEG GUARD] Message while awaiting confirmation — re-entering negotiation")
                _ng_product   = _pre_neg_state.get("product_name", "")
                _ng_price_num = float(_pre_neg_state.get("price_num", 0))
                _ng_reg_price = float(_pre_neg_state.get("regular_price") or _ng_price_num)
                _ng_disc_pct  = int(_pre_neg_state.get("graphrag_discount_pct") or 0)
                if _ng_product and _ng_price_num > 0:
                    _resumed = {**_pre_neg_state, "awaiting_invoice_confirmation": False}
                    _ng_go = _pre_neg_state.get("global_offers")
                    if not _ng_go:
                        try:
                            _ng_to = await get_tenant_offers(incoming.tenant_id)
                            _ng_go = _ng_to.get("offers_text") if _ng_to else None
                        except Exception: _ng_go = None
                    _ng_result = await handle_negotiation(
                        incoming              = incoming,
                        product_name          = _ng_product,
                        price_num             = _ng_price_num,
                        regular_price         = _ng_reg_price,
                        graphrag_discount_pct = _ng_disc_pct,
                        session_history       = session_history,
                        negotiation_state     = _resumed,
                        global_offers         = _ng_go,
                    )
                    await save_negotiation_state(
                        incoming.tenant_id, incoming.session_id, _ng_result["state"]
                    )
                    if _ng_result["order_ready"] and _ng_result["agreed_price"]:
                        _a = _ng_result["agreed_price"]
                        _q = _ng_result["quantity"]
                        _sub  = round(_a * _q, 2)
                        _gst  = round(_sub * incoming.gst_rate, 2)
                        _tot  = round(_sub * (1 + incoming.gst_rate), 2)
                        await save_negotiation_state(
                            incoming.tenant_id, incoming.session_id,
                            {**_ng_result["state"],
                             "awaiting_invoice_confirmation": True,
                             "last_offer_price": _a, "quantity": _q}
                        )
                        reply = "\n".join([
                            f"Here's your updated order summary, {incoming.sender_name}! 🎉",
                            "",
                            f"• *Product:* {_ng_product}",
                            f"• *Quantity:* {_q} units",
                            f"• *Price per unit:* Rs.{_a:,.0f}",
                            f"• *Subtotal:* Rs.{_sub:,.0f}",
                            f"• *GST ({int(incoming.gst_rate*100)}%):* Rs.{_gst:,.2f}",
                            f"• *Total Payable:* Rs.{_tot:,.2f}",
                            "",
                            "Reply *Confirm* to place your order and receive your invoice! 🎉",
                        ])
                    else:
                        reply = _ng_result["reply"]
                    await update_intent(incoming.message_id, "WORKFLOW_ACTION", 0.95)
                    sent = await send_whatsapp_reply(incoming.session_id, reply)
                    if sent:
                        await save_outbound_message(
                            tenant_id  = incoming.tenant_id,
                            session_id = incoming.session_id,
                            message_id = sent,
                            text       = reply,
                    region        = incoming.region,
                        )
                        await update_reply(
                            incoming.message_id, reply,
                            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                            None,
                        )
                    return

        # ── Fast-path confirmation check ───────────────────────────────────
        # _is_invoice_confirmation_request() relies on session_history which
        # is fetched before the outbound order summary is saved — so "Confirm"
        # would fail the LLM check. Fast-path asks the LLM directly whether
        # the message is a confirmation, independently of session history.
        _fast_neg = _pre_neg_state  # reuse — already fetched above
        _is_fast_confirm = False
        if _fast_neg is not None and _fast_neg.get("awaiting_invoice_confirmation", False):
            try:
                _fc_resp = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: _ai_client.chat.completions.create(
                        model       = AZURE_OPENAI_DEPLOYMENT,
                        max_tokens  = 5,
                        temperature = 0,
                        messages    = [
                            {"role": "system", "content": (
                                "The bot just showed an order summary and asked the customer to confirm. "
                                "Is the customer's message a confirmation to place the order?\n"
                                "YES examples: 'confirm', 'proceed', 'yes', 'ok', 'sure', 'do it', "
                                "'ok confirm', 'yes confirm', 'yes proceed', 'ok proceed', "
                                "'ok then proceed with the order', 'proceed with the order', "
                                "'go ahead', 'go ahead with the order', 'yes go ahead', 'sure proceed'\n"
                                "NO examples: 'can I get cheaper', 'any more discount', 'add more units'\n"
                                "Reply ONLY 'YES' or 'NO'."
                            )},
                            {"role": "user", "content": incoming.text},
                        ],
                    )
                )
                _is_fast_confirm = "YES" in _fc_resp.choices[0].message.content.strip().upper()
            except Exception as _fce:
                print(f"[FAST CONFIRM] LLM check failed: {_fce}")

        if await _is_invoice_inquiry(incoming.text) or _is_fast_confirm or await _is_invoice_confirmation_request(incoming, session_history):
            # Guard: check if there is an active negotiation state first.
            # If yes, customer saying "proceed" should finalize the NEGOTIATED order
            # (not fetch an old order from DB).
            # If no negotiation state, check for existing order in DB as before.
            neg_state_check = await get_negotiation_state(
                incoming.tenant_id, incoming.session_id
            )

            if neg_state_check and neg_state_check.get("quantity") and neg_state_check.get("last_offer_price"):
                product_name    = neg_state_check.get("product_name")
                # Prefer auto_offer_unit_price when available — it is the actual
                # discounted price shown to the customer (e.g. 8% off = Rs.2440.76).
                # last_offer_price is now the list price (negotiation_baseline) after
                # the Bug3 fix and should NOT be used as the invoice price for
                # auto-tier orders. Only fall back to last_offer_price when no
                # auto_offer_unit_price exists (i.e. a pure negotiated deal).
                _auto_price     = neg_state_check.get("auto_offer_unit_price")
                agreed_price    = float(_auto_price) if _auto_price else float(neg_state_check.get("last_offer_price", 0))
                quantity        = int(neg_state_check.get("quantity", 0))
                total_price     = round(agreed_price * quantity, 2)
                total_with_gst  = round(total_price * (1 + incoming.gst_rate), 2)
                gst_amount      = round(total_price * incoming.gst_rate, 2)
                awaiting_conf   = neg_state_check.get("awaiting_invoice_confirmation", False)

                if product_name and agreed_price > 0 and quantity > 0:
                    if awaiting_conf:
                        # Customer confirmed — create order and generate invoice
                        print(f"[INVOICE] Confirmation received — creating negotiated order")
                        try:
                            from db.product_store import create_order
                            items = [{
                                "product_name":   product_name,
                                "quantity_value": quantity,
                                "quantity_unit":  "units",
                                "unit_price":     agreed_price,
                                "total_price":    total_price,
                            }]
                            new_order = await create_order(
                                tenant_id   = incoming.tenant_id,
                                session_id  = incoming.session_id,
                                sender_name = incoming.sender_name,
                                items       = items,
                            )
                            if new_order:
                                await clear_negotiation_state(incoming.tenant_id, incoming.session_id)
                                print(f"[INVOICE] Negotiated order created: {new_order.get('order_id')}")
                        except Exception as e:
                            print(f"[INVOICE] Negotiated order creation failed: {e}")
                            new_order = None
                        reply = await handle_invoice_request(incoming, negotiated_order=new_order)
                    else:
                        # First time — show order summary, set flag, wait for confirmation
                        updated_state = {**neg_state_check, "awaiting_invoice_confirmation": True}
                        await save_negotiation_state(incoming.tenant_id, incoming.session_id, updated_state)
                        print(f"[INVOICE] Showing order summary — awaiting confirmation")
                        lines = [
                            f"Here's your order summary, {incoming.sender_name}! Please review:",
                            "",
                            f"• *Product:* {product_name}",
                            f"• *Quantity:* {quantity} units",
                            f"• *Price per unit:* Rs.{agreed_price:,.0f}",
                            f"• *Subtotal:* Rs.{total_price:,.0f}",
                            f"• *GST ({int(incoming.gst_rate*100)}%):* Rs.{gst_amount:,.2f}",
                            f"• *Total Payable:* Rs.{total_with_gst:,.2f}",
                            "",
                            "Reply *Confirm* to place your order and receive your invoice! 🎉",
                        ]
                        reply = "\n".join(lines)
                else:
                    # No agreed price yet — route normally
                    reply = await call_graphrag_api(incoming, session_history)
            else:
                # No active negotiation — check for existing order in DB
                from db.session_store import get_last_order_from_orders
                existing_order = await get_last_order_from_orders(
                    incoming.tenant_id, incoming.session_id
                )
                if existing_order:
                    reply = await handle_invoice_request(incoming)
                else:
                    # No order in DB — route to product follow-up to ask quantity
                    print(f"[INVOICE] Skipped — no existing order found, routing to product follow-up")
                    reply = await call_graphrag_api(incoming, session_history)
        elif result.intent in ("FAQ_KNOWLEDGE", "WORKFLOW_ACTION") or result.confidence_score < 0.50:
            if result.confidence_score < 0.50:
                _unk_neg = await get_negotiation_state(incoming.tenant_id, incoming.session_id)
                if _unk_neg and _unk_neg.get("product_name"):
                    print(f"[UNKNOWN] Active negotiation — redirecting")
                    reply = await call_graphrag_api(incoming, session_history)
                else:
                    reply = await handle_unknown(incoming)
            else:
                reply = await call_graphrag_api(incoming, session_history)
                # Auto-generate invoice in background after product selection / ordering actions
                asyncio.create_task(_ensure_invoice_generated(incoming))

                # If this is an order confirmation, append the invoice generation confirmation prompt
                if await _is_order_confirmation_reply(reply):
                    prompt = await _generate_confirmation_prompt(reply, incoming)
                    if prompt:
                        reply = f"{reply}\n\n{prompt}"
        elif result.intent == "HUMAN_ESCALATION":
            _esc_neg_state = await get_negotiation_state(incoming.tenant_id, incoming.session_id)
            if _esc_neg_state and _esc_neg_state.get("product_name"):
                print(f"[ESCALATION] Active negotiation — redirecting to negotiation handler")
                reply = await call_graphrag_api(incoming, session_history)
            else:
                reply = await handle_escalation(incoming)
        elif result.intent == "GREETING":
            reply = await handle_greeting(incoming)
        else:
            reply = await handle_unknown(incoming)

        # ── Step 9: Send reply + store in DB ──────────────────────────────
        # POST to Meta Graph API. Split messages >3800 chars (GraphRAG can
        # return long product lists). Store reply for audit trail + SLA tracking.
        MSG_SPLIT = "\n\n⟨MSG_SPLIT⟩\n\n"
        success = False
        if not reply or not reply.strip():
            # Empty reply means the handler already sent everything directly
            # (e.g. installation image + link) — nothing more to send here.
            print(f"[PIPELINE] Empty reply — handler already sent message(s) directly, skipping duplicate send")
            success = True
        elif MSG_SPLIT in reply:
            chunks  = reply.split(MSG_SPLIT)
            for i, chunk in enumerate(chunks):
                chunk = chunk.strip()
                if not chunk:
                    continue
                sent_wamid = await send_whatsapp_reply(incoming.session_id, chunk)
                if sent_wamid:
                    success = True
                    print(f"[WHATSAPP] Message chunk {i+1}/{len(chunks)} sent — wamid={sent_wamid}")
                    await save_outbound_message(
                        tenant_id  = incoming.tenant_id,
                        session_id = incoming.session_id,
                        message_id = sent_wamid,
                        text       = chunk,
                    region        = incoming.region,
                    )
                else:
                    print(f"[WHATSAPP] Chunk {i+1}/{len(chunks)} failed")
        else:
            sent_wamid = await send_whatsapp_reply(incoming.session_id, reply)
            if sent_wamid:
                success = True
                await save_outbound_message(
                    tenant_id  = incoming.tenant_id,
                    session_id = incoming.session_id,
                    message_id = sent_wamid,
                    text       = reply,
                    region        = incoming.region,
                )

        if success:
            replied_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            graphrag_raw = getattr(incoming, '_graphrag_raw', None)
            stored_reply_text = reply if reply and reply.strip() else "[handled directly — image/link sent]"
            await update_reply(incoming.message_id, stored_reply_text, replied_at, graphrag_raw)

        print(f"[TIMING] TOTAL pipeline time: {time.monotonic() - _t_pipeline_start:.2f}s")

    finally:
        # Always release the session lock — even if pipeline crashes.
        await release_lock(incoming.session_id)


# ══════════════════════════════════════════════════════════════════════════════
# GRAPHRAG HANDLER — all product queries route here
# ══════════════════════════════════════════════════════════════════════════════

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
        f"\nReply with the product name to know more or place an order."
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
        import json as _json
        try:
            incoming._graphrag_raw = _json.dumps(data, ensure_ascii=False)
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


async def _parse_followup_message(incoming, selection: list, session_history: list = None) -> dict:
    """
    Uses LLM to parse the follow-up message to identify if they are:
    - selecting a product by name (selected_product_name) — NAME ONLY, no numeric index selection
    - specifying quantity/unit (quantity, quantity_unit)
    - requesting comparison (is_comparison)
    - requesting images (asks_for_image)
    - performing a new category search / broad search (is_new_search)
    Zero hardcoding.
    """
    product_names = [p.get("product_name") or p.get("name") or "" for p in selection]
    try:
        recent_history = session_history[-4:] if session_history else []
        response = _ai_client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 200,
            temperature = 0,
            messages    = [
                {"role": "system", "content": (
                    "You are a precise data extraction assistant.\n"
                    "Analyze the customer's message in the context of the list of products currently displayed to them:\n"
                    + "\n".join(f"- {name}" for name in product_names)
                    + "\n\nExtract the following properties and return ONLY a valid JSON object:\n"
                    "- selected_product_name: the specific product name from the list the user is explicitly mentioning by NAME "
                    "(e.g., 'tell me about Reva' → 'Reva', 'I want the Saraswathi divine light' → 'Saraswathi divine light'). "
                    "Set to null if they are not mentioning a specific product by name. "
                    "IMPORTANT: Products are selected by NAME ONLY — a bare number alone (e.g. '5', '12', '57') is NEVER "
                    "a product selection, it is always a quantity or unrelated number. Do not try to match bare numbers to list positions.\n"
                    "- quantity: integer or null (e.g. '1 unit' → 1, 'order 5' → 5, or a bare number like '12' when no product name is given → 12)\n"
                    "- quantity_unit: string or null (e.g. 'units', 'pieces', 'kg')\n"
                    "- is_comparison: boolean (true ONLY if the user explicitly asks to COMPARE two or more "
                    "products side by side — e.g. 'compare this with X', 'compare Romy and Electra', "
                    "'what is the difference between X and Y', 'compare these two'. "
                    "The customer wants a side-by-side table of features. "
                    "Set false for recommendation/suggestion questions.)\n"
                    "- is_recommendation: boolean (true if the user asks for a recommendation or wants help "
                    "picking ONE product — e.g. 'which is better', 'which should I choose', "
                    "'suggest me one', 'recommend one for me', 'which fits my budget', "
                    "'which is best for low budget and high durability', 'which one would you recommend', "
                    "'which is cheaper', 'which one is more durable', 'suggest me one based on budget'. "
                    "The customer wants ONE product picked for them, not a side-by-side table. "
                    "Set false if the customer explicitly asked to compare products side by side.)\n"
                    "- asks_for_image: boolean (true if user asks to see/get/share a picture, image, photo, visual, "
                    "installation guide, installation steps, installation link, or asks 'where is the link', "
                    "'send me the link/guide', 'can you share it', or any request implying they want the actual "
                    "image or link resent — even if they already received one before)\n"
                    "- is_new_search: boolean (true if the user is requesting to browse or know details about a general category or product type "
                    "e.g., 'I want to know the details about garden lights', 'show me gate lights', 'solar lights', rather than asking "
                    "a follow-up question or selecting a specific item from the list shown above).\n"
                    "- is_offer_inquiry: boolean — Set TRUE when customer asks to SEE available offers or discounts. "
                    "This includes asking about offers for a SPECIFIC product by name:\n"
                    "  TRUE examples: 'any offers?', 'is there any offer?', 'any discount available?', "
                    "'what are the offers?', 'any scheme?', 'any deals?', "
                    "'is there any offers for Olivia Stem?', 'any offers for this product?', "
                    "'any offers for Sandy?', 'is there a discount on Romy?', "
                    "'what discount do I get?', 'any special offer for this?'\n"
                    "  FALSE examples: 'can I get for Rs.2000', 'give me 10% off', "
                    "'can we go with 2000 each', 'I want it for 1500', 'my budget is 3000'\n"
                    "Key rule: if it has 'offer', 'discount', 'scheme', 'deal' → TRUE. "
                    "If it has a specific Rs. amount as a counter-price → FALSE.\n\n"
                    "CRITICAL: If the assistant's last message asked 'How many units...' or 'how many' and the user replies with a number, "
                    "that number is ALWAYS the quantity.\n"
                    "CRITICAL: A bare number on its own (e.g. customer just types '12') is ALWAYS a quantity, NEVER a product selection. "
                    "Products can only be selected by typing their name.\n"
                    "Reply ONLY with the JSON object. No other text."
                )},
                *recent_history,
                {"role": "user", "content": incoming.text},
            ],
        )
        content = response.choices[0].message.content.strip()
        # Clean up code fence formatting if any
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        return json.loads(content)
    except Exception as e:
        print(f"[FOLLOW-UP] LLM parser failed: {e}")
        return {
            "selected_product_name": None,
            "quantity": None,
            "quantity_unit": None,
            "is_comparison": False,
            "is_recommendation": False,
            "is_offer_inquiry": False,
            "asks_for_image": False,
            "is_new_search": False
        }


async def _get_active_product_context(
    incoming,
    selection: list,
    session_history: list,
) -> list:
    """
    Scans recent session history to find which specific products were
    being discussed, and returns those as the active comparison context.

    Used when customer asks "which is better for budget?" or "suggest me one"
    without naming products — we first check if specific products were recently
    in focus. Only returns [] when truly nothing was discussed (fresh browse).

    Zero hardcoding — fully LLM-driven.
    """
    if not session_history:
        return []

    product_names = [
        p.get("product_name") or p.get("name") or ""
        for p in selection
        if p.get("product_name") or p.get("name")
    ]
    if not product_names:
        return []

    try:
        response = _ai_client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 120,
            temperature = 0,
            messages    = [
                {"role": "system", "content": (
                    "From the conversation history, identify which specific products "
                    "from the list below were most recently discussed, compared, or "
                    "shown in detail (product brief, features, warranty, comparison).\n\n"
                    "Return ONLY a JSON array of the exact product names from the list. "
                    "Return [] if no specific products were discussed "
                    "(customer just got the category list without picking any).\n\n"
                    "Available products:\n"
                    + "\n".join(f"- {name}" for name in product_names)
                    + "\n\nReturn ONLY valid JSON array. No explanation, no markdown."
                )},
                *session_history[-6:],
                {"role": "user", "content": "Which products from the list were recently being discussed?"},
            ],
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = "\n".join(
                l for l in raw.splitlines()
                if not l.strip().startswith("```")
            ).strip()
        names = json.loads(raw)
        if not isinstance(names, list):
            return []

        result = []
        for name in names:
            name_lower = name.lower().strip()
            for p in selection:
                pname = (p.get("product_name") or p.get("name") or "").lower()
                if name_lower[:12] in pname or pname[:12] in name_lower:
                    if p not in result:
                        result.append(p)
                    break

        print(f"[FOLLOW-UP] Active context: {[p.get('product_name') or p.get('name') for p in result]}")
        return result

    except Exception as e:
        print(f"[FOLLOW-UP] Active context extraction failed: {e}")
        return []


async def _handle_comparison(
    incoming,
    compared: list,
    session_history: list,
    show_recommendation: bool = False,
) -> str:
    """
    show_recommendation=False → comparison only (side-by-side, no recommendation)
    show_recommendation=True  → recommendation only (pick one, no table)
    """
    products_data = []
    for p in compared:
        pname = p.get("product_name") or p.get("name")
        cached = await get_cached_product_by_name(incoming.tenant_id, pname)
        if cached:
            products_data.append(cached)
        else:
            products_data.append(p)

    if show_recommendation:
        system_prompt = f"""You are a helpful WhatsApp assistant for {incoming.biz_name}.
The customer wants a recommendation — pick the BEST product for them based on their criteria.
Do NOT show a side-by-side comparison table.

FORMAT:
- Start with: "*[Product Name]* is the best choice for you, {incoming.sender_name}, because..."
- Give 2-3 short bullet points explaining why it fits their criteria
- End with: "Would you like to order it or know more details?"

RULES:
- Be direct — pick ONE product only
- Do NOT list all products side by side
- Do NOT add a Comparison section

PRODUCTS TO CONSIDER:
{json.dumps(products_data, indent=2)}
"""
    else:
        system_prompt = f"""You are a helpful WhatsApp assistant for {incoming.biz_name}.
The customer wants a side-by-side comparison.

FORMAT RULES — CRITICAL:
- NEVER use markdown tables (no | pipes, no --- dashes). WhatsApp does not render tables.
- Use this exact structure for each product:

*1. [Product Name]*
  • Price: Rs.X,XXX (Y% off Rs.Z,ZZZ)
  • [Key feature 1]
  • [Key feature 2]
  • [Key feature 3]
  • Best for: [use case]

RULES:
- Address the customer as {incoming.sender_name}
- Use *bold* only for product names
- Max 5 bullet points per product
- Do NOT add a Recommendation section
- End with: "Let me know which one you'd like or if you need more details!"

PRODUCTS TO COMPARE:
{json.dumps(products_data, indent=2)}
"""

    try:
        response = _ai_client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 600,
            temperature = 0.3,
            messages    = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": incoming.text},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[FOLLOW-UP] Comparison/recommendation LLM failed: {e}")
        return "I had trouble with that right now. Which product would you like to know more about?"


async def _try_resolve_product_followup(incoming, session_history: list):
    """
    Checks if the customer's message is a follow-up about a product they already
    saw in a previous GraphRAG result (PRODUCT_SELECTION in workflow_sessions).

    RESOLVES TWO CASES:
        1. Name match / comparison: "tell me about Romy", "compare Romy and Reva"
           → word-score customer message against product names in selection,
             or routes to _handle_comparison for multi-product comparisons
        2. Pure follow-up: "is it aluminum?", "what's the warranty?", "1 unit"
           → scan last bot messages to find which product was last discussed

    Returns:
        str  → LLM answer using product data from cache
        None → not a product follow-up, let call_graphrag_api() handle it
    """
    selection = await get_graphrag_product_selection(incoming.tenant_id, incoming.session_id)
    if not selection:
        from db.session_store import get_last_discussed_product
        last_prod = await get_last_discussed_product(incoming.tenant_id, incoming.session_id)
        if last_prod:
            selection = [{
                "product_name": last_prod,
                "name": last_prod,
            }]
            print(f"[FOLLOW-UP] No active selection found - loaded last discussed product from DB: {last_prod}")
        else:
            return None

    # ── Standard follow-up parsing ────────────────────────────────────────────
    # ── Negotiation check ────────────────────────────────────────────────────
    # If customer asks for discount OR has active negotiation state, handle it.
    # Runs BEFORE standard follow-up parsing.
    # New-search guard: if customer asks for a new product category, clear
    # any stale negotiation state and route to GraphRAG instead.
    neg_state = await get_negotiation_state(incoming.tenant_id, incoming.session_id)

    # ── DEDICATED OFFER INQUIRY PRE-CHECK ────────────────────────────────────
    # Runs BEFORE parse and is_negotiation_request.
    # "Currently is there any offers?" / "is there any offers?" →
    # is_negotiation_request returns True (sees "offers" as discount).
    # This focused YES/NO check catches it first and blocks the negotiation path.
    _is_offer_inq = False
    try:
        _oiq = _ai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT, max_tokens=5, temperature=0,
            messages=[
                {"role": "system", "content": (
                    "Does this message ask to SEE available store offers, discounts or schemes?\n"
                    "Is the customer asking to SEE or KNOW ABOUT available store offers/discounts?\n"
                    "YES: 'any offers?', 'currently is there any offers?', 'any offers for this?', "
                    "'any offers for Olivia Stem?', 'is there any offer?', 'what are the offers?', "
                    "'any deals?', 'any scheme?', 'what discounts do you have?'\n"
                    "NO — these are price counter-offers or negotiation requests:\n"
                    "'can I get for Rs.2000', 'give me 10% off', 'I want it for 1500', "
                    "'can I get any additional discount?', 'can you give more discount?', "
                    "'I want extra discount', 'can we go with 2000 each'\n"
                    "KEY: 'can I GET discount' = NO (negotiation). 'are there any offers?' = YES (inquiry).\n"
                    "Reply ONLY 'YES' or 'NO'."
                )},
                {"role": "user", "content": incoming.text},
            ],
        )
        if "YES" in _oiq.choices[0].message.content.strip().upper():
            _is_offer_inq = True
            print(f"[OFFER INQUIRY] Pre-check YES: '{incoming.text}'")
    except Exception as _oiqe:
        print(f"[OFFER INQUIRY] Pre-check failed: {_oiqe}")

    # ── Always parse the follow-up BEFORE the negotiation check ─────────────
    _t_parse_early = time.monotonic()
    quick_parsed = await _parse_followup_message(incoming, selection, session_history)
    print(f"[TIMING] early _parse_followup_message: {time.monotonic() - _t_parse_early:.2f}s")

    # Merge pre-check with parser
    _is_offer_inq = _is_offer_inq or quick_parsed.get("is_offer_inquiry", False)

    # is_comparison/recommendation/offer_inquiry = never a price negotiation.
    if (quick_parsed.get("is_comparison", False)
            or quick_parsed.get("is_recommendation", False)
            or _is_offer_inq):
        print(f"[FOLLOW-UP] is_comparison/recommendation/offer_inquiry — bypassing negotiation")
        neg_state = None
    elif neg_state:
        if quick_parsed.get("is_new_search", False):
            print(f"[NEGOTIATOR] New search — clearing stale negotiation state")
            await clear_negotiation_state(incoming.tenant_id, incoming.session_id)
            neg_state = None
        elif neg_state.get("product_name"):
            saved_product = (neg_state.get("product_name") or "").lower().strip()
            current_products = [
                (p.get("product_name") or p.get("name") or "").lower().strip()
                for p in selection
            ]
            product_still_active = any(
                saved_product[:10] in cp or cp[:10] in saved_product
                for cp in current_products
                if cp
            )
            if current_products and not product_still_active:
                try:
                    prod_check = _ai_client.chat.completions.create(
                        model       = AZURE_OPENAI_DEPLOYMENT,
                        max_tokens  = 5,
                        temperature = 0,
                        messages    = [
                            {"role": "system", "content": (
                                f"Customer was negotiating: '{saved_product}'.\n"
                                f"Current products shown: {', '.join(current_products[:3])}.\n"
                                "Is the customer now asking about a DIFFERENT product?\n"
                                "Reply ONLY 'YES' or 'NO'."
                            )},
                            {"role": "user", "content": incoming.text},
                        ],
                    )
                    is_new_product = "YES" in prod_check.choices[0].message.content.strip().upper()
                except Exception:
                    is_new_product = False

                if is_new_product:
                    print(f"[NEGOTIATOR] Product changed from '{saved_product}' — clearing stale negotiation state")
                    await clear_negotiation_state(incoming.tenant_id, incoming.session_id)
                    neg_state = None

    _t_neg_check_start = time.monotonic()
    _is_neg_req = False if _is_offer_inq else await is_negotiation_request(incoming.text, session_history)
    print(f"[TIMING] is_negotiation_request: {time.monotonic() - _t_neg_check_start:.2f}s")
    if neg_state or _is_neg_req:
        # Resolve which product is being negotiated — priority order:
        # 1. Active negotiation state (already has product_name)
        # 2. Last discussed product from DB
        # 3. Fallback to first in selection
        product_name = (neg_state or {}).get("product_name")

        if not product_name:
            try:
                from db.session_store import get_last_discussed_product
                product_name = await get_last_discussed_product(
                    incoming.tenant_id, incoming.session_id
                )
                if product_name:
                    print(f"[NEGOTIATOR] Using last discussed product: {product_name}")
            except Exception as e:
                print(f"[NEGOTIATOR] get_last_discussed_product failed: {e}")

        if not product_name and getattr(incoming, 'quoted_caption', None):
            for p in selection:
                pname = p.get("product_name") or p.get("name") or ""
                first_word = pname.lower().split()[0] if pname else ""
                if first_word and len(first_word) > 3 and first_word in incoming.quoted_caption.lower():
                    product_name = pname
                    print(f"[NEGOTIATOR] Resolved from quoted caption: {product_name}")
                    break

        if not product_name and selection:
            product_name = selection[0].get("product_name") or selection[0].get("name")
            print(f"[NEGOTIATOR] Fallback to first in selection: {product_name}")

        if product_name:
            cached = await get_cached_product_by_name(incoming.tenant_id, product_name)
            if cached:
                price_num      = float(cached.get("list_price") or 0)
                regular_price  = float(cached.get("regular_price") or price_num)
                discount_pct   = int(cached.get("discount_pct") or 0)

                # IMPORTANT: price_num must always be the TRUE LIST PRICE from the
                # product cache. Never overwrite it with auto_offer_unit_price.
                # auto_offer_unit_price is only used as the negotiation STARTING POINT
                # inside handle_negotiation — it must NOT corrupt price_num itself,
                # otherwise every subsequent quantity change re-discounts an already-
                # discounted price (compounding discount bug) and "regular price" in
                # the order summary shows the discounted price instead of the real one.

                if price_num > 0:
                    current_state = neg_state or {
                        "rounds":            0,
                        "quantity":          None,
                        "last_offer_price":  None,
                        "floor_price":       None,
                        "product_name":      product_name,
                        "price_num":         price_num,
                        "awaiting_quantity": False,
                    }

                    result = await handle_negotiation(
                        incoming               = incoming,
                        product_name           = product_name,
                        price_num              = price_num,
                        regular_price          = regular_price,
                        graphrag_discount_pct  = discount_pct,
                        session_history        = session_history,
                        negotiation_state      = current_state,
                        global_offers          = (
                            (cached.get("global_offers") if cached else None)
                            or (current_state or {}).get("global_offers")
                            or (lambda _t: _t.get("offers_text") if _t else None)(
                                await get_tenant_offers(incoming.tenant_id)
                            )
                        ),
                    )

                    await save_negotiation_state(
                        incoming.tenant_id, incoming.session_id, result["state"]
                    )

                    if result["order_ready"] and result["agreed_price"]:
                        # Guard: if already awaiting confirmation, don't show summary again
                        # This prevents duplicate order summaries when customer keeps negotiating
                        already_awaiting = neg_state and neg_state.get("awaiting_invoice_confirmation", False)
                        if already_awaiting:
                            old_agreed = float(neg_state.get("last_offer_price", 0))
                            new_agreed = float(result["agreed_price"])
                            # Only re-show if price actually changed
                            if abs(old_agreed - new_agreed) < 1.0:
                                print(f"[NEGOTIATOR] Already awaiting confirmation at Rs.{old_agreed} — skipping duplicate summary")
                                return f"You've already confirmed Rs.{old_agreed:,.0f}/unit, {incoming.sender_name}. Please reply *Confirm* to place your order! 🎉"

                        # Do NOT create order yet — show summary first and wait for Confirm
                        agreed  = result["agreed_price"]
                        qty     = result["quantity"]
                        sub     = round(agreed * qty, 2)
                        gst     = round(sub * incoming.gst_rate, 2)
                        total   = round(sub * (1 + incoming.gst_rate), 2)
                        updated = {
                            **result["state"],
                            "awaiting_invoice_confirmation": True,
                            "last_offer_price": agreed,
                            "quantity": qty,
                        }
                        await save_negotiation_state(
                            incoming.tenant_id, incoming.session_id, updated
                        )
                        print(f"[NEGOTIATOR] Showing order summary before invoice")
                        lines = [
                            f"Great news, {incoming.sender_name}! Here's your order summary:",
                            "",
                            f"• *Product:* {product_name}",
                            f"• *Quantity:* {qty} units",
                            f"• *Price per unit:* Rs.{agreed:,.0f}",
                            f"• *Subtotal:* Rs.{sub:,.0f}",
                            f"• *GST ({int(incoming.gst_rate*100)}%):* Rs.{gst:,.2f}",
                            f"• *Total Payable:* Rs.{total:,.2f}",
                            "",
                            "Reply *Confirm* to place your order and receive your invoice! 🎉",
                        ]
                        return "\n".join(lines)

                    if result["escalate"]:
                        await clear_negotiation_state(incoming.tenant_id, incoming.session_id)

                    import json as _j
                    incoming._graphrag_raw = _j.dumps({
                        "handler": "negotiation",
                        "product": product_name,
                        "rounds": result["state"].get("rounds"),
                        "agreed_price": result.get("agreed_price"),
                        "order_ready": result.get("order_ready"),
                    })
                    return result["reply"]

    # ── Standard follow-up parsing ────────────────────────────────────────────
    # quick_parsed is always set above (moved out of the neg_state block)
    # so we always reuse it here — zero duplicate LLM calls.
    if quick_parsed is not None:
        parsed = quick_parsed
        print(f"[FOLLOW-UP] Reusing quick_parsed (skipped duplicate LLM call): {parsed}")
    else:
        _t_parse_start = time.monotonic()
        parsed = await _parse_followup_message(incoming, selection, session_history)
        print(f"[TIMING] _parse_followup_message: {time.monotonic() - _t_parse_start:.2f}s")
        print(f"[FOLLOW-UP] LLM parsed: {parsed}")
    
    # ── Check if user wants to start a new search ────────────────────────────
    # ── Guard: if message matches a product in the current selection list,
    # it is a SELECTION not a new search — even if LLM says is_new_search=True.
    # Happens when bot displays "Outdoor LED Gate Lamp Lights" and customer
    # replies with exactly that text. LLM classifies it as a category search
    # but it is actually selecting item from the list the bot showed.
    _msg_lower = incoming.text.lower().strip()
    _selection_names = [
        (p.get("product_name") or p.get("name") or "").lower().strip()
        for p in selection
    ]
    _matches_selection = any(
        _msg_lower == name or
        (_msg_lower in name and len(_msg_lower) > 6) or
        (name in _msg_lower and len(name) > 6)
        for name in _selection_names if name
    )
    if _matches_selection and parsed.get("is_new_search", False):
        print(f"[FOLLOW-UP] is_new_search overridden — message matches selection list item: '{incoming.text}'")
        parsed["is_new_search"] = False
        # Also set selected_product_name if not already set
        if not parsed.get("selected_product_name"):
            for p in selection:
                pname = (p.get("product_name") or p.get("name") or "").lower().strip()
                if _msg_lower == pname or (_msg_lower in pname and len(_msg_lower) > 6) or (pname in _msg_lower and len(pname) > 6):
                    parsed["selected_product_name"] = p.get("product_name") or p.get("name")
                    print(f"[FOLLOW-UP] Auto-resolved selected_product_name: '{parsed['selected_product_name']}'")
                    break

    if parsed.get("is_new_search", False):
        print(f"[FOLLOW-UP] LLM parser identified category search/new search — routing to GraphRAG")

        # QUERY ENRICHMENT — LLM-driven, zero hardcoded word lists.
        # Only enriches when query has purely vague references (no product info).
        # "related products for this" → enrich with last product ✅
        # "outdoor lights" → skip enrichment (already specific) ✅
        # Two-step: first check if purely vague, then rewrite only if YES.
        selected_product = parsed.get("selected_product_name")
        try:
            from db.session_store import get_last_discussed_product
            last_product = await get_last_discussed_product(
                incoming.tenant_id, incoming.session_id
            )
            # Only enrich when LLM resolved to same last product (vague ref)
            # AND query has no specific product/category info
            should_enrich = (
                last_product
                and selected_product
                and selected_product.lower() == last_product.lower()
            )
            if should_enrich and incoming.text:
                check_resp = _ai_client.chat.completions.create(
                    model       = AZURE_OPENAI_DEPLOYMENT,
                    max_tokens  = 5,
                    temperature = 0,
                    messages    = [
                        {"role": "system", "content": (
                            "Does this query contain ONLY vague pronouns/references "
                            "with NO specific product name, category, or type? "
                            "Answer YES only if it has zero product info. "
                            "Answer NO if it has ANY product keyword (even generic like 'lights').\n"
                            "Reply ONLY 'YES' or 'NO'."
                        )},
                        {"role": "user", "content": incoming.text},
                    ],
                )
                is_vague = "YES" in check_resp.choices[0].message.content.strip().upper()
                if is_vague:
                    enrich_resp = _ai_client.chat.completions.create(
                        model       = AZURE_OPENAI_DEPLOYMENT,
                        max_tokens  = 80,
                        temperature = 0,
                        messages    = [
                            {"role": "system", "content": (
                                f"Rewrite this query replacing vague references with: {last_product}\n"
                                "Reply with ONLY the rewritten query."
                            )},
                            {"role": "user", "content": incoming.text},
                        ],
                    )
                    enriched = enrich_resp.choices[0].message.content.strip()
                    if enriched and enriched != incoming.text:
                        print(f"[FOLLOW-UP] Query enriched: '{incoming.text[:50]}' → '{enriched[:80]}'")
                        incoming.text = enriched
                else:
                    print(f"[FOLLOW-UP] New category search — skipping enrichment")
        except Exception as e:
            print(f"[FOLLOW-UP] Enrichment failed (non-critical): {e}")

        return None

    # NOTE: numeric list-index selection (picking "57" to mean item #57)
    # has been REMOVED entirely. It was unreliable on long product lists
    # (90+ items) and collided with quantity parsing ("57" meaning 57 units).
    # Customers must now select products by NAME only.
    is_comparison     = parsed.get("is_comparison", False)
    is_recommendation = parsed.get("is_recommendation", False)
    is_offer_inquiry  = _is_offer_inq or parsed.get("is_offer_inquiry", False)
    asks_for_image    = parsed.get("asks_for_image", False)

    matched_product = None

    # ── Case 0: Offer inquiry ─────────────────────────────────────────────────
    # Two-layer detection — layer 2 specifically handles "offers for [product name]"
    # which layer 1 sometimes misses due to product context.
    if not is_offer_inquiry:
        try:
            _oi = _ai_client.chat.completions.create(
                model       = AZURE_OPENAI_DEPLOYMENT,
                max_tokens  = 5,
                temperature = 0,
                messages    = [
                    {"role": "system", "content": (
                        "Does this message ask to SEE available store offers, discounts or schemes?\n"
                        "YES: 'any offers?', 'any offers for this?', 'any offers for Olivia Stem?', "
                        "'is there any offers for Sandy?', 'any discount?', 'what are the offers?', "
                        "'any deal?', 'any scheme?', 'is there a discount on this product?'\n"
                        "NO: 'can I get for Rs.2000', 'give me 10% off', 'I want it for 1500', "
                        "'my budget is 3000', 'can we go with 2000 each'\n"
                        "Rule: messages with 'offer'/'discount'/'scheme'/'deal' → YES. "
                        "Messages with a specific Rs. amount as counter → NO.\n"
                        "Reply ONLY 'YES' or 'NO'."
                    )},
                    {"role": "user", "content": incoming.text},
                ],
            )
            if "YES" in _oi.choices[0].message.content.strip().upper():
                is_offer_inquiry = True
                print(f"[OFFER INQUIRY] Detected via layer-2: '{incoming.text}'")
        except Exception as _e:
            print(f"[OFFER INQUIRY] Layer-2 check failed: {_e}")

    if is_offer_inquiry:
        _offers_text = None
        _price_num   = None
        _prod_name   = None

        # ── Priority 1: Use last-discussed product for price calculation ─────
        # Customer asked about Romy → "any offers?" → calculate for Romy, not
        # whatever random product is first in the selection list.
        try:
            from db.session_store import get_last_discussed_product as _gldp
            _last_prod = await _gldp(incoming.tenant_id, incoming.session_id)
            if _last_prod:
                _lcp = await get_cached_product_by_name(incoming.tenant_id, _last_prod)
                if _lcp:
                    _lgo = _lcp.get("global_offers")
                    if _lgo and str(_lgo).strip():
                        _offers_text = str(_lgo).strip()
                        _price_num   = float(_lcp.get("list_price") or 0)
                        _prod_name   = _last_prod
                        print(f"[OFFER INQUIRY] Using last-discussed: '{_prod_name}' @ Rs.{_price_num:,.0f}")
        except Exception as _lde:
            print(f"[OFFER INQUIRY] last_discussed_product lookup failed: {_lde}")

        # ── Priority 2: Check message for named product ───────────────────────
        if not _offers_text:
            # Extract product name from the message (e.g. "any offers for Romy 12W?")
            try:
                _pm = _ai_client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT, max_tokens=30, temperature=0,
                    messages=[
                        {"role": "system", "content": (
                            "Extract the specific product name from this message if one is mentioned.\n"
                            "Reply with ONLY the product name, or 'NONE' if no product is named."
                        )},
                        {"role": "user", "content": incoming.text},
                    ],
                )
                _named = _pm.choices[0].message.content.strip()
                if _named and _named.upper() != "NONE":
                    for p in selection:
                        pname = p.get("product_name") or p.get("name") or ""
                        if _named.lower()[:8] in pname.lower() or pname.lower()[:8] in _named.lower():
                            _cp2 = await get_cached_product_by_name(incoming.tenant_id, pname)
                            _go2 = (_cp2 or p).get("global_offers")
                            if _go2 and str(_go2).strip():
                                _offers_text = str(_go2).strip()
                                _price_num   = float((_cp2 or p).get("list_price") or p.get("price_num") or 0)
                                _prod_name   = pname
                                break
            except Exception:
                pass

        # ── Priority 3: First product in selection with cached global_offers ──
        if not _offers_text:
            for p in selection[:5]:
                pname = p.get("product_name") or p.get("name")
                if pname:
                    _cp = await get_cached_product_by_name(incoming.tenant_id, pname)
                    _go = (_cp or p).get("global_offers")
                    if _go and str(_go).strip():
                        _offers_text = str(_go).strip()
                        _price_num   = float((_cp or p).get("list_price") or p.get("price_num") or 0)
                        _prod_name   = pname
                        break

        # ── Priority 4: tenant_offers table ──────────────────────────────────
        if not _offers_text:
            try:
                _to = await get_tenant_offers(incoming.tenant_id)
                if _to:
                    _offers_text = _to.get("offers_text")
            except Exception:
                pass

        if _offers_text:
            _tier_ctx = ""
            try:
                from ai.negotiator import parse_global_offer_tiers as _pt
                _tiers = _pt(_offers_text)
                if _tiers and _price_num and _price_num > 0:
                    _lines = []
                    for _mv, _dp in _tiers:
                        _dp_price  = round(_price_num * (1 - _dp / 100), 2)
                        _min_units = max(1, int(_mv / _price_num) + (1 if _mv % _price_num else 0))
                        _lines.append(
                            f"  Rs.{_mv:,}+ order → {_dp}% off → "
                            f"Rs.{_dp_price:,.0f}/unit (≈{_min_units}+ units)"
                        )
                    _tier_ctx = (
                        f"\n\nCalculated prices for {_prod_name} (Rs.{_price_num:,.0f}/unit):\n"
                        + "\n".join(_lines)
                    )
            except Exception:
                pass

            try:
                _fmt = _ai_client.chat.completions.create(
                    model       = AZURE_OPENAI_DEPLOYMENT,
                    max_tokens  = 350,
                    temperature = 0.3,
                    messages    = [
                        {"role": "system", "content": (
                            f"You are a sales assistant for {incoming.biz_name}.\n"
                            f"Customer {incoming.sender_name} wants to see available offers.\n"
                            "Show the offers as a clean WhatsApp message with calculated prices.\n"
                            "FORMAT:\n"
                            "- One bullet per tier with emoji\n"
                            "- *bold* for % and thresholds\n"
                            "- Show actual price per unit at each tier\n"
                            "- Also mention free shipping and return policy\n"
                            "- Do NOT invent any numbers — use only what is provided\n"
                            "- End with: 'If you'd like an extra discount, tell me how many "
                            "units you need and I'll work out the best price for you!'\n\n"
                            f"STORE OFFERS:\n{_offers_text}{_tier_ctx}"
                        )},
                        {"role": "user", "content": incoming.text},
                    ],
                )
                return _fmt.choices[0].message.content.strip()
            except Exception as _fe:
                return (
                    f"Here are the current offers, {incoming.sender_name}! 🎉\n\n"
                    + _offers_text + _tier_ctx
                    + "\n\nIf you'd like an extra discount, tell me how many units you need!"
                )
        else:
            return (
                f"I'll check the latest offers for you, {incoming.sender_name}! "
                f"Browse our products and I'll confirm the best available price."
            )

    if quick_parsed.get("is_comparison", False) or quick_parsed.get("is_recommendation", False):
        neg_state = None

    # ── Case 1: Comparison OR recommendation ───────────────────────────────
    if is_comparison or is_recommendation:
        compared_names = []
        if parsed.get("selected_product_name"):
            compared_names.append(parsed["selected_product_name"])
        compared = []
        if compared_names:
            for name in compared_names:
                name_lower = name.lower().strip()
                for p in selection:
                    pname = (p.get("product_name") or p.get("name") or "").lower()
                    if name_lower in pname or pname in name_lower:
                        compared.append(p)
                        break

        # ── Level 2: Pronoun resolution ("compare THIS with X") ──────────
        # Detect pronoun via LLM, then inject last-discussed product.
        _has_pronoun = False
        if len(compared) <= 1:
            try:
                pronoun_resp = _ai_client.chat.completions.create(
                    model       = AZURE_OPENAI_DEPLOYMENT,
                    max_tokens  = 5,
                    temperature = 0,
                    messages    = [
                        {"role": "system", "content": (
                            "Does this message contain a pronoun or vague reference "
                            "(like 'this', 'it', 'the current one', 'this product') "
                            "that refers to a product already being discussed?\n"
                            "Reply ONLY 'YES' or 'NO'."
                        )},
                        {"role": "user", "content": incoming.text},
                    ],
                )
                _has_pronoun = "YES" in pronoun_resp.choices[0].message.content.strip().upper()
            except Exception:
                _has_pronoun = False

        if _has_pronoun and len(compared) <= 1:
            # ── Resolve "this" from session history first (most reliable) ─
            # The DB lookup (last_discussed_product) has a timing gap: the save
            # happens at the END of the previous pipeline run, but both messages
            # can arrive within the same second. Session history is set at the
            # START of this pipeline run so it's guaranteed to be current.
            # Use _get_active_product_context to scan bot's recent messages
            # (e.g. the Villa brief) and find which product "this" refers to.
            try:
                _context_for_pronoun = await _get_active_product_context(
                    incoming, selection, session_history
                )
                for _cp in _context_for_pronoun:
                    _cp_name = (_cp.get("product_name") or _cp.get("name") or "").lower()
                    already_in = any(
                        _cp_name[:12] in (p.get("product_name") or p.get("name") or "").lower()
                        for p in compared
                    )
                    if not already_in:
                        compared.insert(0, _cp)
                        print(f"[FOLLOW-UP] Pronoun 'this' resolved via session history: '{_cp.get('product_name') or _cp.get('name')}'")
                        break  # Only need the single most recently discussed product
            except Exception as e:
                print(f"[FOLLOW-UP] Pronoun history resolution failed: {e}")

            # ── Fallback: DB lookup if history resolution didn't find anything ─
            if len(compared) <= 1:
                try:
                    from db.session_store import get_last_discussed_product
                    _last = await get_last_discussed_product(incoming.tenant_id, incoming.session_id)
                    if _last:
                        _last_lower = _last.lower().strip()
                        _last_p = None
                        for p in selection:
                            pname = (p.get("product_name") or p.get("name") or "").lower()
                            if _last_lower[:12] in pname or pname[:12] in _last_lower:
                                _last_p = p
                                break
                        if _last_p is None:
                            try:
                                _cached = await get_cached_product_by_name(incoming.tenant_id, _last)
                                _last_p = _cached if _cached else {"product_name": _last, "name": _last}
                            except Exception:
                                _last_p = {"product_name": _last, "name": _last}
                        already_in = any(
                            _last_lower[:12] in (p.get("product_name") or p.get("name") or "").lower()
                            for p in compared
                        )
                        if not already_in:
                            compared.insert(0, _last_p)
                            print(f"[FOLLOW-UP] Pronoun resolved via DB fallback: '{_last}'")
                except Exception as e:
                    print(f"[FOLLOW-UP] Pronoun DB fallback failed: {e}")

        # ── Level 3: Active context from session history ──────────────────
        # "suggest me one with low budget" after discussing Romy →
        # use recently-discussed products, not full 18-product list.
        if len(compared) < 2:
            context_products = await _get_active_product_context(
                incoming, selection, session_history
            )
            # Merge: add context products not already in compared
            for cp in context_products:
                cp_name = (cp.get("product_name") or cp.get("name") or "").lower()
                if not any(
                    cp_name[:12] in (p.get("product_name") or p.get("name") or "").lower()
                    for p in compared
                ):
                    compared.append(cp)

        # ── Level 4: Full selection fallback ─────────────────────────────
        # Only when nothing specific was discussed — e.g. customer just
        # received the category list and immediately asks "which is best?"
        if len(compared) < 2:
            compared = selection
            print(f"[FOLLOW-UP] No context found — using full selection ({len(compared)} products)")

        print(f"[FOLLOW-UP] Comparison set: {[c.get('product_name') or c.get('name') for c in compared[:5]]}")

        # Send images if requested
        if asks_for_image:
            for p in compared:
                pname = p.get("product_name") or p.get("name")
                cached = await get_cached_product_by_name(incoming.tenant_id, pname)
                img = (cached or p).get("image_url")
                if img:
                    price = float((cached or p).get("list_price") or (cached or p).get("price_num", 0) or 0)
                    caption = f"{(cached or p).get('product_name') or pname}\nRs.{price:,.0f}"
                    img_wamid = await send_whatsapp_image(incoming.session_id, img, caption)
                    if img_wamid:
                        await save_outbound_message(
                            tenant_id     = incoming.tenant_id,
                            session_id    = incoming.session_id,
                            message_id    = img_wamid,
                            text          = caption,
                            media_url     = img,
                            original_type = "image",
                    region        = incoming.region,
                        )

        import json as _j
        incoming._graphrag_raw = _j.dumps({
            "handler": "comparison" if is_comparison else "recommendation",
            "products": [p.get("product_name") or p.get("name") for p in compared[:5]],
        })
        return await _handle_comparison(
            incoming, compared, session_history,
            show_recommendation=is_recommendation,
        )

    # ── Case 2: Name match ──────────────────────────────────────────────────
    # Check if LLM parsed a specific product name first
    if not matched_product and parsed.get("selected_product_name"):
        tgt_name = parsed["selected_product_name"].lower().strip()
        for p in selection:
            pname = (p.get("product_name") or p.get("name") or "").lower()
            if tgt_name in pname or pname in tgt_name:
                matched_product = p
                print(f"[FOLLOW-UP] Name match via LLM parser: '{tgt_name}' -> {pname}")
                break

    # Fallback to word-score name matching
    if not matched_product:
        import re
        msg_lower = incoming.text.lower().strip()
        msg_words = set(re.findall(r'\b[a-z]+\b', msg_lower))
        best_score = 0
        for p in selection:
            pname  = (p.get("product_name") or p.get("name") or "").lower()
            pwords = set(re.findall(r'\b[a-z]+\b', pname))
            # Only count words >3 chars — skip "led", "12w", "the", "and"
            score = sum(1 for w in pwords if len(w) > 3 and w in msg_words)
            if score > best_score:
                best_score      = score
                matched_product = p

        if matched_product and best_score > 0:
            print(f"[FOLLOW-UP] Name match (score={best_score}): '{msg_lower}' -> {matched_product.get('product_name')}")
        else:
            matched_product = None

    # ── Deterministic bare-number resolution ───────────────────────────────────
    # A bare number means exactly ONE of three things, decided purely from the
    # bot's single most recent message — never guessed, never scanned across
    # multiple turns:
    #
    #   (a) Bot's last message was the freshly-shown product LIST itself
    #       → number is a 1-based LIST POSITION → map to that product by name,
    #         then ask "how many units?" (number is NEVER reused as quantity)
    #   (b) Bot's last message was an explicit quantity question
    #       → number is the QUANTITY for the product already in context
    #   (c) Anything else (order summary, product Q&A, installation reply, etc.)
    #       → ambiguous → ask the customer to reply with the product name
    if not matched_product:
        bare_number_only = re.fullmatch(r"\s*\d{1,4}\s*", incoming.text.strip()) is not None
        if bare_number_only and not parsed.get("selected_product_name"):

            last_bot_msg = ""
            if session_history:
                assistant_msgs = [m["content"] for m in session_history if m.get("role") == "assistant"]
                if assistant_msgs:
                    last_bot_msg = assistant_msgs[-1].lower()

            # Unique marker text that ONLY appears on a freshly-shown product list —
            # guarantees this number is the customer's first reply to THAT exact list.
            bot_just_showed_list = "reply with the product name to know more or place an order" in last_bot_msg

            bot_asked_quantity = (
                "how many units" in last_bot_msg
                or "how many would you like" in last_bot_msg
            )

            extracted_number = int(incoming.text.strip())

            if bot_just_showed_list:
                # (a) Map number -> product by 1-based position in the SAME list
                # that was just shown. This is deterministic: position N in the
                # list the bot displayed maps directly to selection[N-1].
                if 1 <= extracted_number <= len(selection):
                    matched_product = selection[extracted_number - 1]
                    print(f"[FOLLOW-UP] List-position pick: '{extracted_number}' -> {matched_product.get('product_name') or matched_product.get('name')} (list size={len(selection)})")
                    # Force quantity to remain unset — never reuse this number as quantity.
                    parsed["quantity"] = None
                    parsed["_number_was_list_position"] = True  # threaded downstream to suppress quantity inference
                else:
                    print(f"[FOLLOW-UP] '{extracted_number}' out of range for list size={len(selection)} — asking for product name")
                    return (
                        f"Hi {incoming.sender_name}! That number isn't in the list (1-{len(selection)}). "
                        f"Could you please reply with the *product name* instead? 😊"
                    )

            elif bot_asked_quantity:
                # (b) Legitimate quantity context — let existing downstream logic
                # (Case 3 / quantity injection) handle it normally.
                print(f"[FOLLOW-UP] Bot asked quantity — '{extracted_number}' treated as QUANTITY, falling through")

            else:
                # (c) Ambiguous — bot's last message was neither a list nor a
                # quantity question. Do not guess; ask for the product name.
                print(f"[FOLLOW-UP] Bare number '{extracted_number}' with no list/quantity context — asking for product name instead of guessing")
                return (
                    f"Hi {incoming.sender_name}! Could you please reply with the *product name* "
                    f"you'd like to know more about or order? 😊"
                )

    # ── New-search guard before Case 3 ──────────────────────────────────────
    # PERFORMANCE: removed a redundant second LLM call here. _parse_followup_message
    # (called above at the top of this function) already classifies is_new_search
    # using the same product list context. If it said False, we trust that result
    # instead of re-asking the same NEW_SEARCH/FOLLOW_UP question a second time —
    # this was adding a full extra sequential round-trip to every follow-up.
    if not matched_product and False:  # disabled: redundant with parsed["is_new_search"] above
        product_names_in_selection = [
            p.get("product_name", p.get("name", "")) for p in selection
            if p.get("product_name") or p.get("name")
        ]
        try:
            guard_response = _ai_client.chat.completions.create(
                model       = AZURE_OPENAI_DEPLOYMENT,
                max_tokens  = 5,
                temperature = 0,
                messages    = [
                    {"role": "system", "content": (
                        "You classify a customer message as NEW_SEARCH or FOLLOW_UP.\n\n"
                        "NEW_SEARCH — customer is asking about a different product category "
                        "or type that is NOT related to the products listed below.\n"
                        "FOLLOW_UP — customer is asking a follow-up question (feature, price, "
                        "warranty, quantity, delivery) about one of the products listed below, "
                        "or their message is short and context-dependent.\n\n"
                        f"Current products shown to customer:\n"
                        + "\n".join(f"- {n}" for n in product_names_in_selection)
                        + "\n\nReply with ONLY one word: NEW_SEARCH or FOLLOW_UP"
                    )},
                    {"role": "user", "content": incoming.text},
                ],
            )
            classification = guard_response.choices[0].message.content.strip().upper()
            if "NEW_SEARCH" in classification:
                print(f"[FOLLOW-UP] LLM guard: NEW_SEARCH — routing to GraphRAG")
                return None
            print(f"[FOLLOW-UP] LLM guard: FOLLOW_UP — continuing to Case 3")
        except Exception as e:
            print(f"[FOLLOW-UP] LLM guard failed ({e}) — defaulting to FOLLOW_UP")

    # ── Case 3: Pure follow-up — scan bot history ───────────────────────────
    if not matched_product and session_history:
        recent_bot_msgs = [
            m["content"] for m in session_history[-6:]
            if m.get("role") == "assistant"
        ]
        combined_bot_text = " ".join(recent_bot_msgs).lower()
        for p in selection:
            pname      = (p.get("product_name") or p.get("name") or "").lower()
            first_word = pname.split()[0] if pname else ""
            if first_word and len(first_word) > 3 and first_word in combined_bot_text:
                matched_product = p
                print(f"[FOLLOW-UP] Bot history match: '{first_word}' -> {pname}")
                break

    if not matched_product:
        return None

    product_name = matched_product.get("product_name") or matched_product.get("name")
    
    # Save as the last discussed product in the database so context is retained
    try:
        from db.session_store import save_last_discussed_product
        await save_last_discussed_product(incoming.tenant_id, incoming.session_id, product_name)
    except Exception as e:
        print(f"[FOLLOW-UP] Failed to save last discussed product: {e}")

    cached_product = await get_cached_product_by_name(incoming.tenant_id, product_name)

    if not cached_product:
        print(f"[FOLLOW-UP] product_cache miss for '{product_name}' — falling through to GraphRAG")
        return None

    # ── Send image only if explicitly requested ───────────────────────────
    if asks_for_image:
        # Use LLM to decide: is this an installation/steps request or a product image request?
        # Zero hardcoding — LLM reads the actual message and decides.
        try:
            img_intent_resp = _ai_client.chat.completions.create(
                model       = AZURE_OPENAI_DEPLOYMENT,
                max_tokens  = 5,
                temperature = 0,
                messages    = [
                    {"role": "system", "content": (
                        "Classify this customer message into one of two types:\n"
                        "INSTALLATION — customer is asking for installation steps, how to install, "
                        "fitting guide, setup instructions, mounting guide, or how to fit/assemble the product.\n"
                        "PRODUCT_IMAGE — customer is asking to see the product image, photo, picture, or visual.\n"
                        "Reply ONLY with one word: INSTALLATION or PRODUCT_IMAGE"
                    )},
                    {"role": "user", "content": incoming.text},
                ],
            )
            img_intent = img_intent_resp.choices[0].message.content.strip().upper()
        except Exception:
            img_intent = "PRODUCT_IMAGE"
        print(f"[FOLLOW-UP] Image intent: {img_intent}")

        inst_url = (cached_product.get("installation_url") or matched_product.get("installation_url") or "").replace("http://", "https://")
        img_url  = (cached_product.get("image_url") or matched_product.get("image_url") or "").replace("http://", "https://")

        if "INSTALLATION" in img_intent and inst_url:
            # Send installation image
            caption = f"Installation guide — {cached_product.get('product_name') or product_name}"
            inst_wamid = await send_whatsapp_image(incoming.session_id, inst_url, caption)
            if inst_wamid:
                print(f"[FOLLOW-UP] Installation image sent for '{product_name}' — wamid={inst_wamid}")
                await save_outbound_message(
                    tenant_id     = incoming.tenant_id,
                    session_id    = incoming.session_id,
                    message_id    = inst_wamid,
                    text          = caption,
                    media_url     = inst_url,
                    original_type = "image",
                    region        = incoming.region,
                )
            # Also send text with the installation link
            link_text = (
                f"Here is the installation guide for *{cached_product.get('product_name') or product_name}*:\n\n"
                f"🔗 {inst_url}\n\n"
                f"To order, just tell me how many units you'd like!"
            )
            link_wamid = await send_whatsapp_reply(incoming.session_id, link_text)
            if link_wamid:
                await save_outbound_message(
                    tenant_id  = incoming.tenant_id,
                    session_id = incoming.session_id,
                    message_id = link_wamid,
                    text       = link_text,
                    region        = incoming.region,
                )
            return "__ALREADY_HANDLED__"  # Sentinel: image+link already sent, skip GraphRAG + LLM reply

        elif img_url:
            # Product image only
            price   = float(cached_product.get("list_price") or matched_product.get("list_price", 0) or 0)
            caption = f"{cached_product.get('product_name') or product_name}\nRs.{price:,.0f}"
            img_wamid = await send_whatsapp_image(incoming.session_id, img_url, caption)
            if img_wamid:
                print(f"[FOLLOW-UP] Product image sent for '{product_name}' — wamid={img_wamid}")
                await save_outbound_message(
                    tenant_id     = incoming.tenant_id,
                    session_id    = incoming.session_id,
                    message_id    = img_wamid,
                    text          = caption,
                    media_url     = img_url,
                    original_type = "image",
                    region        = incoming.region,
                )

    product_context = {
        "name":                       cached_product.get("product_name"),
        "sku":                        cached_product.get("sku"),
        "price":                      f"Rs.{float(cached_product.get('list_price') or 0):,.0f}",
        "list_price":                 float(cached_product.get("list_price") or 0),
        "discount_pct":               cached_product.get("discount_pct", 0),
        "list_price_num":             float(cached_product.get("list_price") or 0),
        "regular_price":              f"Rs.{float(cached_product.get('regular_price') or 0):,.0f}",
        "discount":                   f"{cached_product.get('discount_pct', 0)}% off",
        "rating":                     cached_product.get("rating", 0),
        "review_count":               cached_product.get("review_count", 0),
        "features":                   cached_product.get("features", []),
        "feature_descriptions":       cached_product.get("feature_descriptions", ""),
        "specs":                      cached_product.get("specs", []),
        "warranties":                 cached_product.get("warranties", []),
        "warranty":                   cached_product.get("warranty", ""),
        "replacement_exchange_policy": cached_product.get("replacement_exchange_policy", ""),
        "installation_url":           cached_product.get("installation_url", ""),
        "global_offers":              cached_product.get("global_offers", ""),
        "delivery_policy":            [
            pol.get("content", "") for pol in cached_product.get("policies", [])
        ],
        "faqs": [
            {"q": f.get("question"), "a": f.get("answer")}
            for f in cached_product.get("faqs", [])
        ],
        "product_url": cached_product.get("product_url"),
    }

    # Inject parsed quantity if present
    number_was_list_position = parsed.get("_number_was_list_position", False)
    if number_was_list_position:
        parsed_qty = None
        product_context["customer_just_selected_by_number"] = True
        print(f"[FOLLOW-UP] Number was used for list-position selection — suppressing quantity inference for this turn")
    else:
        parsed_qty = parsed.get("quantity")
    parsed_unit = parsed.get("quantity_unit") or "units"

    if parsed_qty is not None:
        product_context["parsed_order_quantity"] = parsed_qty
        product_context["parsed_order_unit"]     = parsed_unit

        # ── Auto-apply global offer tier to order ─────────────────────────────
        # If order value qualifies for a tier, apply it automatically and show it
        # in the order summary. The customer does NOT need to negotiate for this.
        # If they want MORE than the auto-applied tier → 5% negotiation path.
        try:
            from ai.negotiator import parse_global_offer_tiers as _pt, get_applicable_tier as _gat, get_next_tier as _gnt
            _price  = float(product_context.get("list_price") or 0)
            _go_str = product_context.get("global_offers") or ""
            if not _go_str:
                # Fallback: tenant_offers table
                _to2 = await get_tenant_offers(incoming.tenant_id)
                _go_str = _to2.get("offers_text", "") if _to2 else ""
            if _price > 0 and _go_str:
                _tiers      = _pt(_go_str)
                _order_val  = _price * int(parsed_qty)
                _, _disc    = _gat(_order_val, _tiers)
                _next_t     = _gnt(_order_val, _tiers)
                if _disc > 0:
                    _disc_price = round(_price * (1 - _disc / 100), 2)
                    _disc_total = round(_disc_price * int(parsed_qty), 2)
                    product_context["auto_offer_applied"]    = True
                    product_context["auto_offer_disc_pct"]   = _disc
                    product_context["auto_offer_unit_price"] = _disc_price
                    product_context["auto_offer_total"]      = _disc_total
                    if _next_t:
                        _u2next = max(1, int((_next_t[0] - _order_val) / _price) + 1)
                        product_context["auto_offer_upsell"] = (
                            f"Order {_u2next} more unit(s) to reach Rs.{_next_t[0]:,} "
                            f"and unlock {_next_t[1]}% off!"
                        )
                    print(f"[OFFER] Auto-applied {_disc}% to {product_name} x {parsed_qty}")
        except Exception as _aoe:
            print(f"[OFFER] Auto-apply failed: {_aoe}")

    recent_history = session_history[-6:] if session_history else []

    try:
        _t_final_start = time.monotonic()
        response = _ai_client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 400,
            temperature = 0.3,
            messages    = [
                {"role": "system", "content": f"""You are a helpful WhatsApp assistant for {incoming.biz_name}.

The customer is interacting about a specific product from a list they were shown.
You have the COMPLETE product data for that product.
Use the conversation history to understand context.

DETECT THE CUSTOMER'S INTENT and respond accordingly:

CRITICAL RULE — NUMBER REUSE: If PRODUCT DATA contains 'customer_just_selected_by_number': true,
the customer's message was a BARE NUMBER used ONLY to pick this product from a numbered list
(e.g. typing "50" to select item 50). That number is NOT a quantity, even though it appears
in the raw message below. In this case you MUST NOT generate an order summary or infer any
quantity — treat this exactly like INTENT A2 and ask "How many units of [Product Name] would
you like?" instead.

INTENT A1 — ORDER WITH QUANTITY:
  Customer is specifying they want to buy/order and they specified the quantity (or 'parsed_order_quantity' is present in the PRODUCT DATA).
  Examples: "I want 1 unit", "I'll take 2", "order 5", "3 pieces", "send me 4", or 'parsed_order_quantity' is present.
  Do NOT apply this intent if 'customer_just_selected_by_number' is true — see CRITICAL RULE above.
  → Generate a clear ORDER SUMMARY:
     • Product: [name]
     • Quantity: [parsed_order_quantity]
     • Regular price: [list_price]/unit (already [discount_pct]% off original)
     IF 'auto_offer_applied' is true in PRODUCT DATA:
       • Store offer [auto_offer_disc_pct]% OFF automatically applied: *Rs.[auto_offer_unit_price]/unit*
       • *Total: Rs.[auto_offer_total]* 🎉
       • Add: "This is the best automatic discount for your order value."
       IF 'auto_offer_upsell' exists: mention it naturally.
     ELSE:
       • Unit Price: [list_price]
       • Total: [quantity × list_price]
  → End with: "Please confirm and we'll process your order! 🎉"
  → NEVER ask "how many" again.

INTENT A2 — ORDER WITHOUT QUANTITY:
  Customer is saying they want to buy, order, purchase, or take the product, but they have NOT specified how many units they want.
  Examples: "I want to order this product", "I want to buy this", "please place an order", "I'll take it".
  → Respond by asking: "How many units of [Product Name] do you want to process with this product?"
  → Do NOT generate any order summary.

INTENT B — PRODUCT QUESTION:
  Customer is asking about features, specs, delivery, warranty, etc.
  → Answer ONLY using the product data — do not invent information.
  → Be concise, max 8 lines.
  → End with: "To order, just tell me how many units you'd like!"

INTENT C — INSTALLATION QUESTION:
  Customer asks about installation, setup, fitting, mounting, or how to install,
  but this turn did NOT trigger a fresh image/link send (no installation_url available,
  or it's a vague follow-up). Do NOT claim anything was already sent unless the customer's
  own message or recent history shows the bot just sent it in this exchange.
  → If installation_url exists in product data, say: "Let me get that installation guide for you — please give me one moment, or reply 'send installation guide' and I'll share it right away."
  → If installation_url does NOT exist: "I don't have an installation guide image for this product yet — please contact our team for installation help."
  → Then briefly describe any installation tips from feature_descriptions if available.
  → End with: "To order, just tell me how many units you'd like!"

RULES:
- Address the customer as {incoming.sender_name}
- NEVER ask "which product?" — the product is already known from context
- Use WhatsApp formatting (• bullets, *bold* for key info)
- If answer not in product data: "I don't have that info, please contact our team"
- NEVER include raw URLs or markdown links like [text](url) in your reply — images are sent separately by the system
- NEVER mention installation_url, image_url or any URL from product data in your text reply
- For warranty questions: read from the "warranty" field and state it clearly in plain text
- NEVER claim you "already sent" an image, link, or guide unless it was sent earlier in THIS visible conversation history — if unsure, offer to send it now instead of claiming it was sent

PRODUCT DATA:
{json.dumps(product_context, indent=2)}
"""},
                *recent_history,
                {"role": "user", "content": incoming.text},
            ],
        )
        reply = response.choices[0].message.content.strip()
        print(f"[TIMING] Final answer LLM call: {time.monotonic() - _t_final_start:.2f}s")
        print(f"[FOLLOW-UP] LLM answered for product '{product_name}'")

        import json as _j
        incoming._graphrag_raw = _j.dumps({
            "handler": "product_followup",
            "product": product_name,
            "quantity": parsed_qty,
        })

        # Save pending order to DB if quantity is specified.
        # CRITICAL: only save _fresh_neg when there is NO active negotiation state
        # with a quantity. If neg_state exists (customer is mid-order, e.g. "add 2
        # more units"), _parse_followup_message extracts the raw number (2) and
        # _fresh_neg would overwrite quantity=1 with quantity=2 — then
        # detect_quantity_change sees current=2 and "add 2" → returns 4 instead of 3.
        # When neg_state exists, handle_negotiation owns the quantity via
        # detect_quantity_change. We must not interfere here.
        _active_neg = await get_negotiation_state(incoming.tenant_id, incoming.session_id)
        _has_active_qty = _active_neg and _active_neg.get("quantity")
        if parsed_qty is not None and not _has_active_qty:
            try:
                from db.session_store import save_pending_order
                await save_pending_order(
                    tenant_id      = incoming.tenant_id,
                    session_id     = incoming.session_id,
                    product_name   = product_name,
                    quantity_value = int(parsed_qty),
                    quantity_unit  = parsed_unit,
                )
                print(f"[ORDER] Saved pending order to DB: {product_name} x {parsed_qty}")
                # Clear stale state then save fresh state. Only runs for NEW orders
                # (no existing neg_state quantity) — never for quantity updates.
                await clear_negotiation_state(incoming.tenant_id, incoming.session_id)
                _fresh_neg = {
                    "product_name":      product_name,
                    "price_num":         float(product_context.get("list_price") or 0),
                    "quantity":          int(parsed_qty),
                    "rounds":            0,
                    "awaiting_quantity": False,
                }
                if product_context.get("auto_offer_applied") and product_context.get("auto_offer_unit_price"):
                    _fresh_neg["auto_offer_unit_price"] = product_context["auto_offer_unit_price"]
                    _fresh_neg["auto_offer_disc_pct"]   = product_context.get("auto_offer_disc_pct", 0)
                await save_negotiation_state(incoming.tenant_id, incoming.session_id, _fresh_neg)
                print(f"[OFFER] Fresh neg_state qty={parsed_qty} auto={_fresh_neg.get('auto_offer_unit_price')}")
            except Exception as e:
                print(f"[ORDER] Failed to save pending order: {e}")
        elif parsed_qty is not None and _has_active_qty:
            print(f"[OFFER] Skipping _fresh_neg save — active neg_state qty={_active_neg.get('quantity')} exists. handle_negotiation owns quantity updates.")

        return reply

    except Exception as e:
        print(f"[FOLLOW-UP] LLM failed: {e} — falling through to GraphRAG")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# INTENT HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

async def handle_escalation(incoming) -> str:
    """
    Handles HUMAN_ESCALATION intent — customer is upset or needs a human.
    GPT reads the specific complaint and acknowledges it empathetically.
    """
    try:
        response = _ai_client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 150,
            temperature = 0.7,
            messages    = [
                {"role": "system", "content": f"""You are a warm, empathetic WhatsApp assistant for {incoming.biz_name}.
The customer is upset or needs human assistance.

Generate a short, empathetic reply that:
- Addresses the customer as {incoming.sender_name}
- Acknowledges their SPECIFIC concern (read their message carefully)
- Assures them a team member will help shortly
- Is warm and professional (max 3 lines)
- Uses 1-2 appropriate emojis

Reply ONLY with the message text — no JSON, no explanation.
"""},
                {"role": "user", "content": incoming.text},
            ],
        )
        reply = response.choices[0].message.content.strip()
        print(f"[ESCALATION] GPT empathy reply generated")
        return reply
    except Exception as e:
        print(f"[ESCALATION] GPT failed: {e} — using fallback")

    support = getattr(incoming, 'support_email', None) or incoming.biz_name
    return (
        f"I understand your concern, {incoming.sender_name}. 🙏\n"
        f"I'm connecting you with a team member right now. "
        f"Someone will respond to you shortly.\n\n"
        f"You can also reach us at *{support}*"
    )


async def handle_greeting(incoming) -> str:
    """
    Handles GREETING intent — any casual greeting, thanks, or farewell.

    TIMEZONE: Uses incoming.timezone from tenants table in DB.
    Each business gets the correct local time automatically.
    Zero hardcoding — timezone comes from DB.
    """
    # Get actual time using tenant's timezone from DB
    try:
        from zoneinfo import ZoneInfo
        tenant_tz = ZoneInfo(incoming.timezone or "UTC")
    except Exception as e:
        print(f"[GREETING] Invalid timezone '{incoming.timezone}': {e} — using UTC")
        tenant_tz = timezone.utc

    now  = datetime.now(tenant_tz)
    hour = now.hour

    if hour < 12:
        time_of_day   = "morning"
        time_greeting = "Good morning"
    elif hour < 17:
        time_of_day   = "afternoon"
        time_greeting = "Good afternoon"
    else:
        time_of_day   = "evening"
        time_greeting = "Good evening"

    print(f"[GREETING] tenant_tz={incoming.timezone} hour={hour} → {time_greeting}")

    try:
        response = _ai_client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 200,
            temperature = 0.7,
            messages    = [
                {"role": "system", "content": f"""You are a friendly WhatsApp assistant for {incoming.biz_name}.
The customer sent a greeting or casual message.

ACTUAL CURRENT TIME: {time_of_day} — it is currently {time_greeting} for this customer.
This is the real server time. Use THIS for any time-based greeting.

CRITICAL RULE: NEVER use the time the customer mentioned in their message.
Always use the ACTUAL CURRENT TIME above.

Classify the message and reply warmly:
- THANK_YOU    → thanking or saying goodbye
- HOW_ARE_YOU  → asking how you are
- OKAY         → acknowledging (ok, noted, sure, got it)
- INTRO        → introducing themselves
- GENERAL      → any other greeting

Reply ONLY with valid JSON — no explanation, no markdown:
{{"type": "GENERAL", "reply": "Your reply here"}}

Reply rules:
- Address the customer as {incoming.sender_name}
- Be warm, natural, concise (max 4 lines)
- Use 1-2 appropriate emojis
- THANK_YOU   → say you're welcome warmly and wish them well
- HOW_ARE_YOU → say you're doing great, ask how you can help today
- OKAY        → acknowledge positively, ask what they need help with
- INTRO       → acknowledge their name warmly, ask how you can help
- GENERAL     → ALWAYS start with "{time_greeting}" (the ACTUAL time)
                 then offer help: browse products, place order, connect with team
"""},
                {"role": "user", "content": f"[ACTUAL_SERVER_TIME: {time_of_day} / {time_greeting}]\n{incoming.text}"},
            ],
        )

        raw    = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        reply  = parsed.get("reply", "")

        if reply:
            print(f"[GREETING] GPT type={parsed.get('type')} time={time_of_day} reply generated")
            return reply

    except Exception as e:
        print(f"[GREETING] GPT failed: {e} — using fallback")

    return (
        f"{time_greeting}, {incoming.sender_name}! 👋\n\n"
        f"How can I help you today?\n"
        f"• 💡 Browse or order products\n"
        f"• 🙋 Connect with our team"
    )


async def handle_unknown(incoming) -> str:
    """
    Handles UNKNOWN intent — unclear or out-of-scope messages.
    GPT generates a contextual helpful response instead of a generic one.
    """
    try:
        response = _ai_client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 150,
            temperature = 0.7,
            messages    = [
                {"role": "system", "content": f"""You are a friendly WhatsApp assistant for {incoming.biz_name}.
The customer sent an unclear or out-of-scope message.

Generate a helpful reply that:
- Addresses the customer as {incoming.sender_name}
- Gently acknowledges you didn't quite understand
- Lists what you CAN help with:
  • 💡 Browse or search for products
  • 📦 Place an order
  • 🙋 Connect with our support team
- Asks what they need help with
- Is warm and friendly (max 4 lines)

Reply ONLY with the message text — no JSON, no explanation.
"""},
                {"role": "user", "content": incoming.text},
            ],
        )
        reply = response.choices[0].message.content.strip()
        print(f"[UNKNOWN] GPT reply generated")
        return reply
    except Exception as e:
        print(f"[UNKNOWN] GPT failed: {e} — using fallback")

    support = getattr(incoming, 'support_email', None) or incoming.biz_name
    return (
        f"Hi {incoming.sender_name}! 👋\n\n"
        f"I can help you with:\n"
        f"• 💡 Browsing or searching for products\n"
        f"• 📦 Placing an order\n"
        f"• 🙋 Connecting with our support team\n\n"
        f"What would you like help with today?"
    )


async def _is_invoice_inquiry(message: str) -> bool:
    """
    Uses LLM to determine if the customer's message is asking for their invoice,
    bill, receipt, or payment document. Zero hardcoding.
    """
    try:
        response = _ai_client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 5,
            temperature = 0,
            messages    = [
                {"role": "system", "content": (
                    "Determine if the user is explicitly asking for their invoice, receipt, "
                    "bill, or payment document for their order (e.g. 'where is my invoice', "
                    "'send invoice', 'invoice please', 'show bill').\n"
                    "Reply ONLY 'YES' or 'NO'."
                )},
                {"role": "user", "content": message},
            ],
        )
        content = response.choices[0].message.content.strip().upper()
        return "YES" in content
    except Exception as e:
        print(f"[INVOICE] Inquiry check failed: {e}")
        return False


async def _is_order_confirmation_reply(reply_text: str) -> bool:
    """
    Uses LLM to determine if the assistant's reply text indicates that an order has been
    confirmed, placed, processed, or scheduled. Zero hardcoding.
    """
    try:
        response = _ai_client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 5,
            temperature = 0,
            messages    = [
                {"role": "system", "content": (
                    "Determine if the assistant's reply text indicates that an order has been "
                    "confirmed, placed, processed, or scheduled (e.g., 'Thank you for confirming', "
                    "'Your order is now being processed', 'order confirmed', 'will now be processed').\n"
                    "Reply ONLY 'YES' or 'NO'."
                )},
                {"role": "user", "content": reply_text},
            ],
        )
        content = response.choices[0].message.content.strip().upper()
        return "YES" in content
    except Exception as e:
        print(f"[INVOICE] Confirmation reply check failed: {e}")
        return False


async def _generate_confirmation_prompt(reply_text: str, incoming) -> str:
    """
    Dynamically generates a short line asking the user to reply with 'Confirm' or 'Proceed'
    to automatically generate and receive their tax invoice. Zero hardcoding.
    """
    try:
        response = _ai_client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 100,
            temperature = 0.5,
            messages    = [
                {"role": "system", "content": (
                    f"You are a WhatsApp assistant for {incoming.biz_name}.\n"
                    "The customer's order has just been confirmed. "
                    "Write a short line (max 1 line) asking them to reply with 'Confirm' or 'Proceed' "
                    "so we can automatically generate and send their tax invoice.\n"
                    "Make it natural and warm, and use emojis if appropriate.\n"
                    "Example: Reply 'Proceed' or 'Confirm' to get your invoice right away! 📄"
                )},
                {"role": "user", "content": reply_text},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[INVOICE] Failed to generate confirmation prompt: {e}")
        return "Reply 'Proceed' or 'Confirm' to automatically generate and receive your tax invoice! 📄"


async def _is_invoice_confirmation_request(incoming, session_history: list) -> bool:
    """
    Uses LLM to determine if the customer's message is a confirmation (e.g., 'Proceed', 'Confirm')
    in response to the assistant's previous message asking them to confirm to generate their invoice.
    Zero hardcoding.
    """
    if not session_history:
        return False

    recent_bot_msgs = [
        m["content"] for m in session_history[-4:]
        if m.get("role") == "assistant"
    ]
    if not recent_bot_msgs:
        return False

    last_bot_msg = recent_bot_msgs[-1]

    try:
        response = _ai_client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 5,
            temperature = 0,
            messages    = [
                {"role": "system", "content": (
                    "Determine if the user is replying with confirmation (like 'Proceed', 'Confirm', "
                    "'Yes proceed', 'do it', 'sure') to the assistant's previous message asking them "
                    "to confirm or proceed to generate/receive their invoice.\n"
                    "Reply ONLY 'YES' or 'NO'."
                )},
                {"role": "user", "content": f"Assistant: {last_bot_msg}\nUser: {incoming.text}"},
            ],
        )
        content = response.choices[0].message.content.strip().upper()
        return "YES" in content
    except Exception as e:
        print(f"[INVOICE] Confirmation check failed: {e}")
        return False


async def handle_invoice_request(incoming, negotiated_order: dict = None) -> str:
    """
    Generates invoice PDF. When negotiated_order is passed (from confirmed
    negotiation), uses it directly — avoids stale pending order overriding
    the correct negotiated quantity/price.
    """
    print(f"[INVOICE] Handling invoice request for session {incoming.session_id}")
    from db.session_store import (
        get_last_order_from_orders,
        update_order_invoice_url,
        get_pending_order,
        delete_pending_order,
        get_cached_product_by_name,
    )
    from db.product_store import create_order
    from utils.invoice import generate_and_upload_invoice

    if negotiated_order:
        # Use the just-created negotiated order directly (correct qty + price).
        # Delete stale pending order which has old qty/price from before negotiation.
        order = negotiated_order
        print(f"[INVOICE] Using negotiated order: {order.get('order_id')}")
        try:
            pending = await get_pending_order(incoming.tenant_id, incoming.session_id)
            if pending:
                await delete_pending_order(incoming.tenant_id, incoming.session_id)
                print(f"[INVOICE] Deleted stale pending order")
        except Exception:
            pass
    else:
        # Non-negotiated path: commit pending order if exists
        try:
            pending = await get_pending_order(incoming.tenant_id, incoming.session_id)
            if pending:
                print(f"[INVOICE] Committing pending order: {pending}")
                product_name = pending["product_name"]
                qty_val  = pending["quantity_value"]
                qty_unit = pending["quantity_unit"] or "units"
                cached_product = await get_cached_product_by_name(incoming.tenant_id, product_name)
                if cached_product:
                    unit_price = float(cached_product.get("list_price") or 0)
                    items = [{
                        "product_name":   product_name,
                        "quantity_value": qty_val,
                        "quantity_unit":  qty_unit,
                        "unit_price":     unit_price,
                        "total_price":    qty_val * unit_price,
                    }]
                    new_order = await create_order(
                        tenant_id   = incoming.tenant_id,
                        session_id  = incoming.session_id,
                        sender_name = incoming.sender_name,
                        items       = items,
                    )
                    if new_order:
                        print(f"[INVOICE] Order committed: {new_order.get('order_id')}")
                        await delete_pending_order(incoming.tenant_id, incoming.session_id)
        except Exception as commit_err:
            print(f"[INVOICE] Error committing pending order: {commit_err}")

        order = await get_last_order_from_orders(incoming.tenant_id, incoming.session_id)
    if not order:
        return (
            f"I couldn't find any recent orders for you, {incoming.sender_name}. 🤔\n\n"
            f"If you'd like to place a new order, just let me know what you need!"
        )

    # Get invoice_url if already exists
    invoice_url = order.get("invoice_url")
    if not invoice_url:
        print(f"[INVOICE] Invoice URL missing for order {order.get('order_id')} — generating now...")
        invoice_url = await generate_and_upload_invoice(
            order         = order,
            biz_name      = incoming.biz_name,
            tagline       = incoming.tagline,
            city          = incoming.city,
            support_email = incoming.support_email,
            website       = incoming.website,
            upi_id        = incoming.upi_id,
            account_name  = incoming.account_name,
        )
        if invoice_url:
            await update_order_invoice_url(order["order_id"], incoming.tenant_id, invoice_url)
            print(f"[INVOICE] Invoice URL updated in DB: {invoice_url}")

    if invoice_url:
        return (
            f"Here is your tax invoice for order *{order.get('order_id')}*, {incoming.sender_name}! 📄\n\n"
            f"🔗 *Download Invoice PDF*:\n{invoice_url}\n\n"
            f"Thank you for doing business with *{incoming.biz_name}*! 🙏"
        )
    else:
        support = getattr(incoming, 'support_email', None) or incoming.biz_name
        return (
            f"I had trouble generating your invoice PDF right now, {incoming.sender_name}. 🔧\n\n"
            f"Please contact our team at *{support}* to get your invoice."
        )


async def _ensure_invoice_generated(incoming):
    """
    Checks if a confirmed order exists in the session without an invoice PDF,
    generates it in the background if needed.
    """
    try:
        from db.session_store import get_last_order_from_orders, update_order_invoice_url
        from utils.invoice import generate_and_upload_invoice

        order = await get_last_order_from_orders(incoming.tenant_id, incoming.session_id)
        if not order:
            return

        invoice_url = order.get("invoice_url")
        if not invoice_url:
            print(f"[INVOICE] Auto-generating invoice for order {order.get('order_id')}...")
            invoice_url = await generate_and_upload_invoice(
                order         = order,
                biz_name      = incoming.biz_name,
                tagline       = incoming.tagline,
                city          = incoming.city,
                support_email = incoming.support_email,
                website       = incoming.website,
                upi_id        = incoming.upi_id,
                account_name  = incoming.account_name,
            )
            if invoice_url:
                await update_order_invoice_url(order["order_id"], incoming.tenant_id, invoice_url)
                print(f"[INVOICE] Auto-generated invoice uploaded: {invoice_url}")

    except Exception as e:
        print(f"[INVOICE] Auto-generation failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# WHATSAPP SEND UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

async def send_whatsapp_reply(to: str, message: str) -> Optional[str]:
    """
    Sends a text reply to a WhatsApp user via Meta Graph API.
    Returns the message ID (wamid) if successful, None otherwise.
    """
    url     = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type":    "individual",
        "to":                to,
        "type":              "text",
        "text":              {"body": message},
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            print(f"[WHATSAPP] Reply sent to {to}")
            try:
                res_data = response.json()
                msg_id = res_data.get("messages", [{}])[0].get("id")
                return msg_id or "unknown_wamid"
            except Exception:
                return "unknown_wamid"
        else:
            print(f"[WHATSAPP] Error {response.status_code}: {response.text}")
            return None


async def send_whatsapp_image(to: str, image_url: str, caption: str = "") -> Optional[str]:
    """
    Sends a product image to a WhatsApp user via Meta Graph API.
    Returns the message ID (wamid) if successful, None otherwise.
    """
    try:
        url     = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"
        headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"}
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type":    "individual",
            "to":                to,
            "type":              "image",
            "image": {
                "link":    image_url,
                "caption": caption,
            },
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                print(f"[WHATSAPP] Image sent to {to} — {image_url[:60]}")
                try:
                    res_data = response.json()
                    msg_id = res_data.get("messages", [{}])[0].get("id")
                    return msg_id or "unknown_wamid"
                except Exception:
                    return "unknown_wamid"
            else:
                print(f"[WHATSAPP] Image failed {response.status_code}: {response.text[:100]}")
                return None
    except Exception as e:
        print(f"[WHATSAPP] Image send error: {e}")
        return None





        #s