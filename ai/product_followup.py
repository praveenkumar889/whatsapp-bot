# ai/product_followup.py — Product follow-up resolution engine
#
# Extracted from main.py to keep the orchestrator lightweight.
# Contains: _try_resolve_product_followup, _parse_followup_message,
#           _get_active_product_context, _handle_comparison
# All imports must be explicit — no globals from main.py.

import re
import json
import time
from typing import Optional
from openai import AzureOpenAI

from config import (
    AZURE_AI_ENDPOINT, AZURE_AI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT, AZURE_AI_API_VERSION,
)
from db.session_store import (
    get_graphrag_product_selection,
    get_negotiation_state,
    save_negotiation_state,
    clear_negotiation_state,
    get_product_api_response,
    get_cached_product_by_name,
    save_last_discussed_product,
    get_last_discussed_product,
    save_graphrag_product_selection,
    get_tenant_offers,
    save_outbound_message,
)
from ai.negotiator import (
    is_negotiation_request,
    handle_negotiation,
)
from adapter.whatsapp_adapter import send_whatsapp_reply, send_whatsapp_image

_client = AzureOpenAI(
    azure_endpoint = AZURE_AI_ENDPOINT,
    api_key        = AZURE_AI_API_KEY,
    api_version    = AZURE_AI_API_VERSION,
    timeout        = 30.0,
    max_retries    = 0,
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
        response = _client.chat.completions.create(
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
        response = _client.chat.completions.create(
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
  • Price: Rs.[list_price] (if discount_pct > 0, add: (Y% off Rs.[regular_price]) using the real discount_pct value for Y)
  • [Key feature 1]
  • [Key feature 2]
  • [Key feature 3]
  • Best for: [use case]

PRICE RULE — CRITICAL:
- [list_price], [regular_price], and [discount_pct] are placeholders — you MUST replace them with
  the actual numeric values from that product's entry in PRODUCTS TO COMPARE below.
- NEVER output the literal characters "X,XXX", "Y", or "Z,ZZZ" — those do not exist in the data.
  If you see them in this prompt, they are only illustrating the STRUCTURE, not real values.
- If discount_pct is 0 or missing, just show "Price: Rs.[list_price]" with no discount clause.
- Do NOT invent any numbers — use only what is provided in PRODUCTS TO COMPARE.

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
        response = _client.chat.completions.create(
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

    # ── NUMBER SELECTION RESOLVER ────────────────────────────────────────────
    # Runs BEFORE any LLM call. When a numbered product list was shown,
    # resolve number references to product names while preserving intent context.
    #
    # Handles:
    #   "compare 11 and 12"       → "compare Figo Solar Wall Light and Nyla Solar Wall Light"
    #   "give me details about 11" → "give me details about Figo Solar Wall Light"
    #   "11" (bare)               → "Figo Solar Wall Light"
    #   "I want 2 units"          → skip (qty context)
    if selection and len(selection) > 1:
        import re as _re
        _txt = incoming.text.strip()
        _qty_ctx = bool(_re.search(
            r'\b(units?|pieces?|pcs?|qty|quantity|of them)\b', _txt, _re.IGNORECASE
        ))
        if not _qty_ctx:
            # Extract all numbers from the message
            _all_nums = _re.findall(r'(?<![\d])(?:#|no\.?\s*|sr\.?\s*|option\s+|product\s+|item\s+|number\s+)?(\d+)(?![\d])', _txt, _re.IGNORECASE)
            _in_range = [int(n) for n in _all_nums if 1 <= int(n) <= len(selection)]

            if len(_in_range) >= 2:
                # COMPARISON: "compare 11 and 12", "11 vs 12"
                _p1 = selection[_in_range[0]-1].get("product_name") or selection[_in_range[0]-1].get("name", "")
                _p2 = selection[_in_range[1]-1].get("product_name") or selection[_in_range[1]-1].get("name", "")
                if _p1 and _p2:
                    # Detect compare/vs/versus/difference keywords
                    _is_compare = bool(_re.search(
                        r'\b(compare|vs\.?|versus|difference|better|which)\b',
                        _txt, _re.IGNORECASE
                    ))
                    if _is_compare:
                        print(f"[NUMBER-SELECT] Compare #{_in_range[0]} vs #{_in_range[1]}: '{_p1}' vs '{_p2}'")
                        incoming.text = f"compare {_p1} and {_p2}"
                    else:
                        # "11 or 12", "11 and 12" without explicit compare word
                        print(f"[NUMBER-SELECT] Multi-select #{_in_range[0]} & #{_in_range[1]}: defaulting to compare")
                        incoming.text = f"compare {_p1} and {_p2}"

            elif len(_in_range) == 1:
                _n = _in_range[0]
                _chosen = selection[_n - 1]
                _chosen_name = _chosen.get("product_name") or _chosen.get("name", "")
                if _chosen_name:
                    # Replace just the number (and its optional prefix) in the message,
                    # preserving any intent context before/after it.
                    # "give me details about 11" → "give me details about Figo Solar..."
                    # "order 11" → "order Figo Solar..."
                    # "11" → "Figo Solar..."
                    _replaced = _re.sub(
                        r'(?:#|no\.?\s*|sr\.?\s*|option\s+|product\s+|item\s+|number\s+)?' + str(_n) + r'(?!\d)',
                        _chosen_name,
                        _txt,
                        count=1,
                        flags=_re.IGNORECASE
                    ).strip()
                    print(f"[NUMBER-SELECT] #{_n} → '{_chosen_name}' | '{_txt}' → '{_replaced}'")
                    incoming.text = _replaced

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
        _oiq = _client.chat.completions.create(
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
                    prod_check = _client.chat.completions.create(
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
                        # BUG-071: add next-tier upsell to confirmation summary
                        _conf_upsell = ""
                        try:
                            from ai.negotiator import parse_global_offer_tiers as _pt71, get_next_tier as _gnt71
                            _go71 = result["state"].get("global_offers", "")
                            if _go71:
                                _tiers71  = _pt71(_go71)
                                _ov71     = agreed * qty
                                _next71   = _gnt71(_ov71, _tiers71)
                                if _next71:
                                    _u71 = max(1, int((_next71[0] - _ov71) / agreed) + 1)
                                    _conf_upsell = (f"\nOrder {_u71} more unit(s) to reach "
                                                    f"Rs.{_next71[0]:,} and unlock {_next71[1]}% off!")
                        except Exception as _e71:
                            print(f"[CONFIRM] Upsell calc failed: {_e71}")
                        _auto_disc_pct  = result["state"].get("auto_offer_disc_pct", 0)
                        _auto_unit      = result["state"].get("auto_offer_unit_price")
                        _s_save         = round((price_num - (_auto_unit or agreed)) * qty, 2)
                        _n_save         = round(((_auto_unit or agreed) - agreed) * qty, 2) if _auto_unit else 0
                        _t_save         = round((price_num - agreed) * qty, 2)
                        lines = [
                            f"Here's your order summary, {incoming.sender_name}! Please review:",
                            "",
                            f"• *Product:* {product_name}",
                            f"• *Quantity:* {qty} units",
                        ]
                        if _auto_disc_pct and _auto_unit and _n_save > 0:
                            lines += [
                                f"• *Regular price:* Rs.{price_num:,.0f}/unit",
                                f"• *Store offer {_auto_disc_pct}% OFF:* Rs.{_auto_unit:,.0f}/unit",
                                f"• *Negotiated price:* Rs.{agreed:,.0f}/unit",
                            ]
                        elif _auto_disc_pct and _auto_unit:
                            lines += [
                                f"• *Regular price:* Rs.{price_num:,.0f}/unit",
                                f"• *Store offer {_auto_disc_pct}% OFF:* Rs.{agreed:,.0f}/unit",
                            ]
                        else:
                            lines.append(f"• *Price per unit:* Rs.{agreed:,.0f}")
                        lines += [
                            f"• *Subtotal:* Rs.{sub:,.0f}",
                            f"• *GST ({int(incoming.gst_rate*100)}%):* Rs.{gst:,.2f}",
                            f"• *Total Payable:* Rs.{total:,.2f}",
                        ]
                        if _t_save > 0:
                            if _s_save > 0 and _n_save > 0:
                                lines += [
                                    "",
                                    f"🎁 *Total savings: Rs.{_t_save:,.0f}*",
                                    f"   • Store offer: Rs.{_s_save:,.0f}",
                                    f"   • Negotiation: Rs.{_n_save:,.0f}",
                                ]
                            else:
                                lines.append(f"\n🎁 *You save Rs.{_t_save:,.0f} on this order!*")
                        if _conf_upsell:
                            lines.append(_conf_upsell)
                        lines += ["", "Reply *Confirm* to place your order and receive your invoice! 🎉"]
                        return "\n".join(lines)

                    if result["escalate"]:
                        await clear_negotiation_state(incoming.tenant_id, incoming.session_id)

                    incoming._graphrag_raw = json.dumps({
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
                check_resp = _client.chat.completions.create(
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
                    enrich_resp = _client.chat.completions.create(
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
            _oi = _client.chat.completions.create(
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
                _pm = _client.chat.completions.create(
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
                _fmt = _client.chat.completions.create(
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
                pronoun_resp = _client.chat.completions.create(
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

        incoming._graphrag_raw = json.dumps({
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
            bot_just_showed_list = "reply with the product" in last_bot_msg and ("name" in last_bot_msg or "number" in last_bot_msg)

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
            guard_response = _client.chat.completions.create(
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

    # ── Case 4: last_discussed_product DB fallback ───────────────────────────
    if not matched_product:
        try:
            _ld = await get_last_discussed_product(incoming.tenant_id, incoming.session_id)
            if _ld:
                for p in selection:
                    pname = (p.get("product_name") or p.get("name") or "").lower()
                    if _ld.lower()[:12] in pname or pname[:12] in _ld.lower():
                        matched_product = p
                        break
                if not matched_product:
                    _ldc = await get_cached_product_by_name(incoming.tenant_id, _ld)
                    if _ldc:
                        matched_product = {"product_name": _ld, "name": _ld}
                if matched_product:
                    print(f"[FOLLOW-UP] Case 4 DB fallback: last_discussed='{_ld}'")
        except Exception as _lde:
            print(f"[FOLLOW-UP] Case 4 fallback failed: {_lde}")

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
            img_intent_resp = _client.chat.completions.create(
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
                f"Need help with anything else?\n"
                f"• 💰 Pricing & offers\n"
                f"• 📦 Place an order\n"
                f"• 🔒 Warranty\n\n"
                f"Or just tell me how many units you'd like and I'll set it up for you!"
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
        response = _client.chat.completions.create(
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

        incoming._graphrag_raw = json.dumps({
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