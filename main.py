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
import json
import time
from typing import Optional
from datetime import datetime, timezone, timedelta
from openai import AzureOpenAI
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse

from config import (
    VERIFY_TOKEN, ACCESS_TOKEN, PHONE_NUMBER_ID,
    AZURE_AI_ENDPOINT, AZURE_AI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT, AZURE_AI_API_VERSION,
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
    incoming.support_phone = tenant_info.get("support_phone")
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
            # ── QTY+CONFIRM SPLIT ─────────────────────────────────────────────
            # "add 1 more unit and confirm" must update qty FIRST, then confirm.
            _split_qty_done = False
            try:
                from ai.negotiator import detect_quantity_change as _dqc
                _cur_qty = int(_pre_neg_state.get("quantity") or 0)
                if _cur_qty > 0:
                    _new_qty = await _dqc(incoming.text, _cur_qty)
                    if _new_qty and _new_qty != _cur_qty:
                        print(f"[QTY+CONFIRM] qty change {_cur_qty}→{_new_qty} with confirm — processing qty first")
                        _split_state = {**_pre_neg_state, "awaiting_invoice_confirmation": False, "quantity": _cur_qty}
                        _split_go = _pre_neg_state.get("global_offers")
                        if not _split_go:
                            try:
                                _sgo = await get_tenant_offers(incoming.tenant_id)
                                _split_go = _sgo.get("offers_text") if _sgo else None
                            except Exception: _split_go = None
                        _split_result = await handle_negotiation(
                            incoming              = incoming,
                            product_name          = _pre_neg_state.get("product_name", ""),
                            price_num             = float(_pre_neg_state.get("price_num", 0)),
                            regular_price         = float(_pre_neg_state.get("regular_price") or _pre_neg_state.get("price_num", 0)),
                            graphrag_discount_pct = int(_pre_neg_state.get("graphrag_discount_pct") or 0),
                            session_history       = session_history,
                            negotiation_state     = _split_state,
                            global_offers         = _split_go,
                        )
                        await save_negotiation_state(
                            incoming.tenant_id, incoming.session_id,
                            {**_split_result["state"], "awaiting_invoice_confirmation": True,
                             "quantity": _split_result.get("quantity", _new_qty)}
                        )
                        _split_reply = _split_result.get("reply", "")
                        if "Confirm" not in _split_reply and "confirm" not in _split_reply:
                            _split_reply += "\n\nReply *Confirm* to place your order and receive your invoice! 🎉"
                        sent = await send_whatsapp_reply(incoming.session_id, _split_reply)
                        if sent:
                            await save_outbound_message(tenant_id=incoming.tenant_id,
                                session_id=incoming.session_id, message_id=sent,
                                text=_split_reply, region=incoming.region)
                            await update_reply(incoming.message_id, _split_reply,
                                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), None)
                        return
            except Exception as _sqe:
                print(f"[QTY+CONFIRM] Split check failed: {_sqe}")

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
                        _ng_price_raw = float(_ng_result["state"].get("price_num") or _a)
                        _ng_auto_unit = float(_ng_result["state"].get("auto_offer_unit_price") or _a)
                        _ng_auto_pct  = int(_ng_result["state"].get("auto_offer_disc_pct") or 0)
                        _ng_s_save    = round((_ng_price_raw - _ng_auto_unit) * _q, 2)
                        _ng_n_save    = round((_ng_auto_unit - _a) * _q, 2)
                        _ng_tot_save  = round((_ng_price_raw - _a) * _q, 2)
                        _ng_lines = [
                            f"Here's your updated order summary, {incoming.sender_name}! 🎉",
                            "",
                            f"• *Product:* {_ng_product}",
                            f"• *Quantity:* {_q} units",
                        ]
                        if _ng_auto_pct and _ng_s_save > 0 and _ng_n_save > 0:
                            _ng_lines += [
                                f"• *Regular price:* Rs.{_ng_price_raw:,.0f}/unit",
                                f"• *Store offer {_ng_auto_pct}% OFF:* Rs.{_ng_auto_unit:,.0f}/unit",
                                f"• *Negotiated price:* Rs.{_a:,.0f}/unit",
                            ]
                        elif _ng_auto_pct and _ng_s_save > 0:
                            _ng_lines += [
                                f"• *Regular price:* Rs.{_ng_price_raw:,.0f}/unit",
                                f"• *Store offer {_ng_auto_pct}% OFF:* Rs.{_a:,.0f}/unit",
                            ]
                        else:
                            _ng_lines.append(f"• *Price per unit:* Rs.{_a:,.0f}")
                        _ng_lines += [
                            f"• *Subtotal:* Rs.{_sub:,.0f}",
                            f"• *GST ({int(incoming.gst_rate*100)}%):* Rs.{_gst:,.2f}",
                            f"• *Total Payable:* Rs.{_tot:,.2f}",
                        ]
                        if _ng_tot_save > 0:
                            if _ng_s_save > 0 and _ng_n_save > 0:
                                _ng_lines += [
                                    f"",
                                    f"🎁 *Total savings: Rs.{_ng_tot_save:,.0f}*",
                                    f"   • Store offer: Rs.{_ng_s_save:,.0f}",
                                    f"   • Negotiation: Rs.{_ng_n_save:,.0f}",
                                ]
                            else:
                                _ng_lines.append(f"\n🎁 *You save Rs.{_ng_tot_save:,.0f} on this order!*")
                        _ng_lines += ["", "Reply *Confirm* to place your order and receive your invoice! 🎉"]
                        reply = "\n".join(_ng_lines)
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
                                "YES: 'confirm', 'proceed', 'yes', 'ok', 'sure', 'do it', "
                                "'ok confirm', 'yes confirm', 'yes proceed', 'ok proceed', "
                                "'ok then proceed with the order', 'proceed with the order', "
                                "'go ahead', 'go ahead with the order', 'yes go ahead', 'sure proceed'\n"
                                "NO — contains quantity change, must NOT confirm:\n"
                                "'add 1 more unit and confirm', 'add 3 units and proceed', "
                                "'increase to 10 and confirm', 'make it 7 and proceed', "
                                "'can I get cheaper', 'any more discount', 'add more units'\n"
                                "RULE: if message has a qty change (add/increase/make it N), reply NO.\n"
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
                # Price resolution — must distinguish two cases:
                # Case A: Auto-tier only (rounds=0, no negotiation happened)
                #         → use auto_offer_unit_price (e.g. 8% = Rs.2294)
                # Case B: Negotiation rounds happened (rounds > 0)
                #         → use last_offer_price (the actual negotiated price, e.g. Rs.1229)
                #         NEVER use auto_offer_unit_price here — it is the PRE-negotiation
                #         price and using it overcharges the customer by the negotiated amount.
                _neg_rounds     = int(neg_state_check.get("rounds", 0))
                _auto_price     = neg_state_check.get("auto_offer_unit_price")
                _last_price     = neg_state_check.get("last_offer_price")
                if _neg_rounds > 0 and _last_price:
                    # Negotiation happened — last_offer_price IS the agreed negotiated price
                    agreed_price = float(_last_price)
                    print(f"[CONFIRM] Using negotiated price Rs.{agreed_price:,.0f} (rounds={_neg_rounds})")
                elif _auto_price:
                    # No negotiation — use auto-tier price
                    agreed_price = float(_auto_price)
                    print(f"[CONFIRM] Using auto-tier price Rs.{agreed_price:,.0f} (no negotiation)")
                else:
                    agreed_price = float(_last_price or 0)
                    print(f"[CONFIRM] Fallback to last_offer_price Rs.{agreed_price:,.0f}")
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
                            # Compute discount breakdown for invoice transparency
                            _price_num_raw   = float(neg_state_check.get("price_num") or agreed_price)
                            _auto_unit_price = float(neg_state_check.get("auto_offer_unit_price") or agreed_price)
                            _auto_disc_pct   = int(neg_state_check.get("auto_offer_disc_pct") or 0)
                            _store_disc_amt  = round((_price_num_raw - _auto_unit_price) * quantity, 2)
                            _neg_disc_amt    = round((_auto_unit_price - agreed_price) * quantity, 2)
                            _orig_amount     = round(_price_num_raw * quantity, 2)
                            new_order = await create_order(
                                tenant_id   = incoming.tenant_id,
                                session_id  = incoming.session_id,
                                sender_name = incoming.sender_name,
                                items       = items,
                            )
                            if new_order:
                                # Attach discount breakdown so invoice PDF shows full transparency
                                if _orig_amount > total_price:
                                    new_order["original_amount"]            = _orig_amount
                                if _store_disc_amt > 0:
                                    new_order["store_discount_pct"]         = _auto_disc_pct
                                    new_order["store_discount_amount"]      = _store_disc_amt
                                if _neg_disc_amt > 0:
                                    new_order["negotiation_discount_amount"] = _neg_disc_amt
                                await clear_negotiation_state(incoming.tenant_id, incoming.session_id)
                                print(f"[INVOICE] Order {new_order.get('order_id')} created with "
                                      f"store_disc=Rs.{_store_disc_amt:.0f} neg_disc=Rs.{_neg_disc_amt:.0f}")
                        except Exception as e:
                            print(f"[INVOICE] Negotiated order creation failed: {e}")
                            new_order = None
                        reply = await handle_invoice_request(incoming, negotiated_order=new_order)
                    else:
                        # First time — show order summary, set flag, wait for confirmation
                        updated_state = {**neg_state_check, "awaiting_invoice_confirmation": True}
                        await save_negotiation_state(incoming.tenant_id, incoming.session_id, updated_state)
                        print(f"[INVOICE] Showing order summary — awaiting confirmation")
                        # Transparent breakdown: regular → store offer → negotiated → GST
                        _s_price_raw  = float(neg_state_check.get("price_num") or agreed_price)
                        _s_auto_unit  = float(neg_state_check.get("auto_offer_unit_price") or agreed_price)
                        _s_auto_pct   = int(neg_state_check.get("auto_offer_disc_pct") or 0)
                        _s_store_save = round((_s_price_raw - _s_auto_unit) * quantity, 2)
                        _s_neg_save   = round((_s_auto_unit - agreed_price) * quantity, 2)
                        _s_total_save = round((_s_price_raw - agreed_price) * quantity, 2)
                        _s_lines = [
                            f"Here's your order summary, {incoming.sender_name}! Please review:",
                            "",
                            f"• *Product:* {product_name}",
                            f"• *Quantity:* {quantity} units",
                        ]
                        if _s_auto_pct and _s_store_save > 0 and _s_neg_save > 0:
                            # Store offer + negotiation both applied
                            _s_lines += [
                                f"• *Regular price:* Rs.{_s_price_raw:,.0f}/unit",
                                f"• *Store offer {_s_auto_pct}% OFF:* Rs.{_s_auto_unit:,.0f}/unit",
                                f"• *Negotiated price:* Rs.{agreed_price:,.0f}/unit",
                            ]
                        elif _s_auto_pct and _s_store_save > 0:
                            # Store offer only
                            _s_lines += [
                                f"• *Regular price:* Rs.{_s_price_raw:,.0f}/unit",
                                f"• *Store offer {_s_auto_pct}% OFF:* Rs.{agreed_price:,.0f}/unit",
                            ]
                        else:
                            _s_lines.append(f"• *Price per unit:* Rs.{agreed_price:,.0f}")
                        _s_lines += [
                            f"• *Subtotal:* Rs.{total_price:,.0f}",
                            f"• *GST ({int(incoming.gst_rate*100)}%):* Rs.{gst_amount:,.2f}",
                            f"• *Total Payable:* Rs.{total_with_gst:,.2f}",
                        ]
                        if _s_total_save > 0:
                            if _s_store_save > 0 and _s_neg_save > 0:
                                _s_lines += [
                                    f"",
                                    f"🎁 *Total savings: Rs.{_s_total_save:,.0f}*",
                                    f"   • Store offer: Rs.{_s_store_save:,.0f}",
                                    f"   • Negotiation: Rs.{_s_neg_save:,.0f}",
                                ]
                            else:
                                _s_lines.append(f"\n🎁 *You save Rs.{_s_total_save:,.0f} on this order!*")
                        _s_lines += ["", "Reply *Confirm* to place your order and receive your invoice! 🎉"]
                        reply = "\n".join(_s_lines)
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


# ═══════════════════════════════════════════════════════════════════════════
# Extracted modules — functions live here, main.py just calls them
# ═══════════════════════════════════════════════════════════════════════════
from ai.graphrag_handler import (
    call_graphrag_api,
    _send_structured_product_list,
    _coerce_pythonic_dict,
)
from ai.product_followup import (
    _try_resolve_product_followup,
    _parse_followup_message,
    _get_active_product_context,
    _handle_comparison,
)
from ai.response_handlers import (
    handle_greeting,
    handle_escalation,
    handle_unknown,
)
from ai.invoice_handler import (
    handle_invoice_request,
    _ensure_invoice_generated,
    _is_invoice_inquiry,
    _is_order_confirmation_reply,
    _generate_confirmation_prompt,
    _is_invoice_confirmation_request,
)
from adapter.whatsapp_adapter import send_whatsapp_reply, send_whatsapp_image