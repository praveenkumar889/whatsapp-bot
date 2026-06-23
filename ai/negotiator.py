# ai/negotiator.py — Price Negotiator Module
#
# NEGOTIATION DESIGN:
#   Floor  = second tier discount from global_offers (typically 5%)
#   Spread over 3-4 TURNS — never give the full 5% upfront:
#     Turn 1: ~1/3 of floor discount
#     Turn 2: ~2/3 of floor discount
#     Turn 3-4: floor = 5% (final, never lower)
#
# Handles TOTAL and PER-UNIT price counter-offers:
#   "can I get for Rs.1000 each" → per-unit counter
#   "can I get the total for Rs.4000" → total → /quantity → per-unit
#
# ZERO HARDCODING:
#   - Tier thresholds/discounts → from global_offers via LLM parse
#   - Floor % → second tier from real offers data
#   - All replies → LLM generated

import json
from typing import Optional
from openai import AzureOpenAI
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
    timeout        = 30.0,
    max_retries    = 0,
)

MAX_NEGOTIATION_ROUNDS  = 4    # 3-4 turns before hitting floor
FALLBACK_FLOOR_DISC_PCT = 5    # fallback negotiation floor % when no global_offers available


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL OFFER TIER LOGIC  (replaces hardcoded get_tier_discount)
# ══════════════════════════════════════════════════════════════════════════════

def parse_global_offer_tiers(global_offers: str) -> list:
    """
    Parses global_offers text → sorted [(min_order_value, discount_pct), ...]
    LLM-driven — zero regex or format hardcoding.
    """
    if not global_offers or not global_offers.strip():
        return []
    try:
        response = _client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT, max_tokens=150, temperature=0,
            messages=[
                {"role": "system", "content": (
                    "Extract value-based discount tiers from this store offers text.\n"
                    "Return ONLY a JSON array of [min_order_value, discount_pct] pairs.\n"
                    "Example: [[2500, 2], [7500, 5], [14500, 8]]\n"
                    "Sort ascending by min_order_value. Return [] if none found.\n"
                    "Reply with ONLY the JSON array."
                )},
                {"role": "user", "content": global_offers},
            ],
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(len(t) == 2 for t in parsed):
            return sorted(parsed, key=lambda x: x[0])
        return []
    except Exception as e:
        print(f"[NEGOTIATOR] parse_global_offer_tiers failed: {e}")
        return []


def get_negotiation_floor_disc(tiers: list) -> int:
    """
    Returns the negotiation floor discount %.
    Uses second tier (index 1) — typically 5% for Inventaa.
    Falls back to FALLBACK_FLOOR_DISC_PCT if no tiers available.
    """
    if len(tiers) >= 2:
        return tiers[1][1]
    elif len(tiers) == 1:
        return tiers[0][1]
    return FALLBACK_FLOOR_DISC_PCT


def get_applicable_tier(order_value: float, tiers: list) -> tuple:
    applicable = (0, 0)
    for min_val, disc_pct in tiers:
        if order_value >= min_val:
            applicable = (min_val, disc_pct)
        else:
            break
    return applicable


def get_next_tier(order_value: float, tiers: list) -> Optional[tuple]:
    for min_val, disc_pct in tiers:
        if order_value < min_val:
            return (min_val, disc_pct)
    return None


def calculate_offer(price_num: float, quantity: int, tiers: list = None) -> dict:
    """
    Calculates offer using global_offers tiers.
    Floor = second tier (5%). First offer = 1/3 of the way to floor.
    Gives natural 3-turn negotiation before hitting floor.
    """
    tiers      = tiers or []
    floor_disc = get_negotiation_floor_disc(tiers)
    floor_price = round(price_num * (1 - floor_disc / 100), 2)
    order_value = price_num * quantity
    _, current_disc = get_applicable_tier(order_value, tiers) if tiers else (0, 0)
    max_disc = max((d for _, d in tiers), default=0) if tiers else 0

    gap         = price_num - floor_price
    offer_price = round(price_num - gap / 3, 2)

    return {
        "offer_price":       offer_price,
        "total_price":       round(offer_price * quantity, 2),
        "floor_price":       floor_price,
        "floor_disc":        floor_disc,
        "tier_discount_pct": current_disc,
        "has_discount":      floor_disc > 0,
        "price_num":         price_num,
        "quantity":          quantity,
        "order_value":       order_value,
        "tiers":             tiers,
        "current_tier_disc": current_disc,
        "max_tier_disc":     max_disc,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LLM DETECTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

async def is_negotiation_request(
    message: str,
    session_history: list = None,
) -> bool:
    """
    LLM detects if customer is asking for a discount or negotiating price.
    Zero hardcoded keywords — purely LLM-driven.

    Returns True for messages like:
        "Can you give me a discount?"
        "That's too expensive"
        "Any better price?"
        "Can you reduce the price?"
        "Is there any offer?"
        "Can you do better?"
    """
    try:
        messages = [
            {"role": "system", "content": (
                "Determine if the customer is asking for a price discount, "
                "negotiating the price, saying the price is too high, or asking "
                "for any deal, offer, or price reduction.\n"
                "Reply ONLY with 'YES' or 'NO'."
            )},
        ]
        if session_history:
            messages.extend(session_history[-4:])
        messages.append({"role": "user", "content": message})

        response = _client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 5,
            temperature = 0,
            messages    = messages,
        )
        return "YES" in response.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"[NEGOTIATOR] is_negotiation_request failed: {e}")
        return False


async def extract_quantity(
    message: str,
    product_name: str,
    session_history: list = None,
) -> Optional[int]:
    """
    LLM extracts quantity from customer message.
    Returns None if no quantity mentioned — caller asks for it.
    """
    try:
        messages = [
            {"role": "system", "content": (
                f"The customer is discussing buying '{product_name}'.\n"
                "Extract ONLY the number of units they want to buy.\n"
                "Reply with ONLY the integer, or 'NONE' if not mentioned."
            )},
        ]
        if session_history:
            messages.extend(session_history[-4:])
        messages.append({"role": "user", "content": message})

        response = _client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 10,
            temperature = 0,
            messages    = messages,
        )
        raw = response.choices[0].message.content.strip().upper()
        if raw == "NONE" or not raw.isdigit():
            return None
        qty = int(raw)
        return qty if qty > 0 else None
    except Exception as e:
        print(f"[NEGOTIATOR] extract_quantity failed: {e}")
        return None


async def detect_quantity_change(
    message: str,
    current_quantity: int,
    product_name: str,
    session_history: list = None,   # kept for signature compat but NOT passed to LLM
) -> Optional[int]:
    """
    Detects if the customer is asking to CHANGE the quantity already locked
    into this negotiation.

    Returns the NEW TOTAL quantity, or None if no change requested.

    CRITICAL — NO session_history passed to LLM:
        Session history contains upsell messages like "Order 2 more units
        to reach Rs.7,500!". When passed as context, the LLM reads those
        messages and misinterprets "add 1 more unit" as adding the upsell
        amount (2) instead of 1, giving wrong totals (1+2=3 instead of 1+1=2).
        The current_quantity parameter is the ONLY source of truth for state.
    """
    try:
        response = _client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 10,
            temperature = 0,
            messages    = [
                {"role": "system", "content": (
                    f"The customer's order currently has exactly {current_quantity} unit(s) "
                    f"of '{product_name}'.\n"
                    f"THIS IS THE ONLY SOURCE OF TRUTH — ignore any other quantities "
                    f"you may have seen.\n\n"
                    "If the customer is changing the quantity, reply with the NEW TOTAL.\n"
                    "Relative changes:\n"
                    f"  'add 1 more' → {current_quantity + 1}\n"
                    f"  'add 2 more' → {current_quantity + 2}\n"
                    f"  'remove 1'   → {current_quantity - 1}\n"
                    "Absolute changes:\n"
                    f"  'make it 5'  → 5\n"
                    f"  'I want 8'   → 8\n"
                    "If NOT a quantity change (price negotiation, acceptance, question): reply NONE.\n"
                    "Reply with ONLY the integer or NONE."
                )},
                {"role": "user", "content": message},
            ],
        )
        raw = response.choices[0].message.content.strip().upper()
        if raw == "NONE" or not raw.isdigit():
            return None
        new_qty = int(raw)
        if new_qty > 0 and new_qty != current_quantity:
            return new_qty
        return None
    except Exception as e:
        print(f"[NEGOTIATOR] detect_quantity_change failed: {e}")
        return None


async def detect_counter_offer(
    message: str,
    session_history: list = None,
    quantity: int = None,
    current_price_num: float = None,
) -> Optional[float]:
    """
    Detects if customer is proposing a specific price and returns it AS PER UNIT.

    Handles both total-price and per-unit-price counter-offers:
        "Can you do Rs.2,500 per unit?"  → 2500.0   (per unit, direct)
        "Can we go for Rs.5,000 total?"  → 1250.0   (total / quantity = per unit)
        "can I get it for 4000" (qty=4, price=1306) → 1000.0 (4000 is total, /4)
        "Still too high"                 → None

    Logic: if the stated price is LESS than current_price_num, it's per-unit.
           if the stated price is MORE than current_price_num but LESS than
           current_price_num * quantity, it's a total price → divide by quantity.
           LLM also explicitly asked to classify total vs per-unit.
    """
    try:
        ctx = ""
        if current_price_num and quantity:
            ctx = (
                f"\nContext: current price is Rs.{current_price_num:,.0f}/unit, "
                f"quantity is {quantity} units, "
                f"current total is Rs.{current_price_num * quantity:,.0f}."
            )

        messages = [
            {"role": "system", "content": (
                "The customer may be proposing a specific price.\n"
                "Determine:\n"
                "1. The price amount they stated (number only, strip Rs. and commas)\n"
                "2. Whether it is a PER-UNIT price or a TOTAL price for all units\n\n"
                f"{ctx}\n"
                "Rules for total vs per-unit:\n"
                "- If price is clearly less than current per-unit price → it's PER UNIT\n"
                "- If price is between current per-unit and current total → it's TOTAL price\n"
                "- Words like 'each', 'per unit', 'per piece' → PER UNIT\n"
                "- Words like 'total', 'overall', 'for all', 'for the order' → TOTAL\n"
                "- When ambiguous and price is close to total → assume TOTAL\n\n"
                "Reply with ONLY one of:\n"
                "  UNIT:<number>   (e.g. UNIT:2500)\n"
                "  TOTAL:<number>  (e.g. TOTAL:4000)\n"
                "  NONE"
            )},
        ]
        if session_history:
            messages.extend(session_history[-4:])
        messages.append({"role": "user", "content": message})

        response = _client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 20,
            temperature = 0,
            messages    = messages,
        )
        raw = response.choices[0].message.content.strip().upper()

        if raw == "NONE" or not raw:
            return None

        if raw.startswith("UNIT:"):
            val = raw[5:].replace("RS.", "").replace("₹", "").replace(",", "").strip()
            result = float(val) if val.replace(".", "").isdigit() else None
            if result:
                print(f"[NEGOTIATOR] Counter-offer: Rs.{result:,.0f}/unit (per-unit price)")
            return result

        if raw.startswith("TOTAL:"):
            val = raw[6:].replace("RS.", "").replace("₹", "").replace(",", "").strip()
            total = float(val) if val.replace(".", "").isdigit() else None
            if total and quantity and quantity > 0:
                per_unit = round(total / quantity, 2)
                print(f"[NEGOTIATOR] Counter-offer: Rs.{total:,.0f} total → Rs.{per_unit:,.0f}/unit")
                return per_unit
            return total  # fallback if no quantity context

        return None

    except Exception as e:
        print(f"[NEGOTIATOR] detect_counter_offer failed: {e}")
        return None


async def detect_more_discount_request(
    message: str,
    session_history: list = None,
) -> bool:
    """
    Detects if customer is asking for more/further discount without
    specifying a particular price (e.g. "any more discount?",
    "can you do better?", "more than 10%?", "give me extra discount").
    Different from detect_counter_offer which needs a specific price number.
    """
    try:
        response = _client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 5,
            temperature = 0,
            messages    = [
                {"role": "system", "content": (
                    "Is the customer asking for more/further/additional discount "
                    "or a better price, WITHOUT mentioning a specific price number?\n"
                    "Examples of YES: 'any more discount?', 'can you do better?', "
                    "'more than 10%', 'give me extra off', 'any further reduction?'\n"
                    "Examples of NO: 'can you do Rs.1,200?', 'how about 1500?', "
                    "'I accept', 'ok proceed'\n"
                    "Reply ONLY 'YES' or 'NO'."
                )},
                {"role": "user", "content": message},
            ],
        )
        return "YES" in response.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"[NEGOTIATOR] detect_more_discount_request failed: {e}")
        return False


async def detect_acceptance(
    message: str,
    session_history: list = None,
) -> bool:
    """
    Detects if customer is PURELY accepting the current offer with no new price.
    Examples: "OK", "Deal", "Sounds good", "Proceed", "Yes", "I'll take it", "We go for 1840"

    CRITICAL: "Can we go for 1800?" is a COUNTER-OFFER — must return False.
    Only return True when customer is agreeing/confirming, NOT when proposing a new price.
    """
    try:
        response = _client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 5,
            temperature = 0,
            messages    = [
                {"role": "system", "content": (
                    "Is the customer ACCEPTING or AGREEING to the current price offer?\n"
                    "\n"
                    "Answer YES only for clear acceptance/agreement:\n"
                    "  YES: 'OK', 'Deal', 'Proceed', 'Yes', 'I accept', 'Let\'s go', 'We go for [price]', 'That works'\n"
                    "\n"
                    "Answer NO for counter-offers or questions:\n"
                    "  NO: 'Can we go for 1800?', 'How about 1700?', 'Can you do 600?', 'What about 1500?'\n"
                    "  NO: Any message asking IF a price is possible (contains 'can', 'could', 'would', 'any chance')\n"
                    "  NO: Any message with a question mark proposing a new price\n"
                    "\n"
                    "Reply ONLY with 'YES' or 'NO'."
                )},
                {"role": "user", "content": message},
            ],
        )
        return "YES" in response.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"[NEGOTIATOR] detect_acceptance failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# LLM REPLY GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

async def _reply_ask_quantity(
    sender: str,
    product_name: str,
    price_num: float,
    regular_price: float,
    discount_pct: int,
    biz_name: str,
) -> str:
    """Asks how many units customer wants before giving discount offer."""
    try:
        response = _client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 120,
            temperature = 0.4,
            messages    = [
                {"role": "system", "content": (
                    f"You are a friendly sales assistant for {biz_name}.\n"
                    f"Customer is asking for a discount on *{product_name}*.\n"
                    f"Current price: Rs.{price_num:,.0f} (already {discount_pct}% off Rs.{regular_price:,.0f})\n"
                    "Ask how many units they want — the quantity determines the extra discount they qualify for.\n"
                    f"Address customer as {sender}. Be warm, concise (max 3 lines)."
                )},
                {"role": "user", "content": "Ask for quantity."},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[NEGOTIATOR] _reply_ask_quantity failed: {e}")
        return (
            f"I'd love to help you get a better price, {sender}! 😊\n\n"
            f"How many units of *{product_name}* are you looking to buy?"
        )


async def _reply_no_discount(
    sender: str,
    product_name: str,
    price_num: float,
    regular_price: float,
    discount_pct: int,
    quantity: int,
    biz_name: str,
) -> str:
    """Tells customer no extra discount for < 5 units but mentions how to qualify."""
    try:
        response = _client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 150,
            temperature = 0.4,
            messages    = [
                {"role": "system", "content": (
                    f"You are a friendly sales assistant for {biz_name}.\n"
                    f"Customer wants {quantity} unit(s) of *{product_name}*.\n"
                    f"Current price: Rs.{price_num:,.0f} (already {discount_pct}% off Rs.{regular_price:,.0f})\n"
                    "For orders below 5 units, no additional discount is available.\n"
                    "However, mention that buying 5+ units qualifies for extra discounts.\n"
                    "Be warm, honest, and helpful. Max 4 lines.\n"
                    f"Address customer as {sender}. Use *bold* for prices."
                )},
                {"role": "user", "content": "Give the no-discount response."},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[NEGOTIATOR] _reply_no_discount failed: {e}")
        total = round(price_num * quantity, 2)
        return (
            f"{sender}, for {quantity} unit(s) the price is *Rs.{price_num:,.0f}* per unit "
            f"(Total: *Rs.{total:,.0f}*).\n\n"
            f"💡 Buy 5+ units to unlock extra discounts!\n\n"
            f"Would you like to proceed at this price?"
        )


async def _reply_first_offer(
    sender: str,
    product_name: str,
    price_num: float,
    regular_price: float,
    graphrag_discount_pct: int,
    offer: dict,
    biz_name: str,
) -> str:
    """Presents the tier-based first offer to the customer."""
    try:
        # Actual % off price_num for this starting offer (not the tier %)
        actual_offer_pct = round((1 - offer["offer_price"] / price_num) * 100, 1) if price_num > 0 else 0

        context = (
            f"Product: {product_name}\n"
            f"Regular price: Rs.{regular_price:,.0f}\n"
            f"Already discounted price: Rs.{price_num:,.0f} ({graphrag_discount_pct}% off)\n"
            f"Customer quantity: {offer['quantity']} units\n"
            f"Offer price: Rs.{offer['offer_price']:,.0f} per unit ({actual_offer_pct}% extra off Rs.{price_num:,.0f})\n"
            f"Total for {offer['quantity']} units: Rs.{offer['total_price']:,.0f}\n"
            f"IMPORTANT: Do NOT say '{offer['tier_discount_pct']}% off' — that is internal, not the offer %."
        )
        response = _client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 200,
            temperature = 0.4,
            messages    = [
                {"role": "system", "content": (
                    f"You are a friendly sales assistant for {biz_name}.\n"
                    "Present a quantity-based price offer to the customer.\n"
                    "MUST show all 3 prices clearly:\n"
                    "  1. Original price (regular_price)\n"
                    "  2. Already discounted price (price_num) with its % off\n"
                    f"  3. Final offer price with the EXACT extra discount % ({actual_offer_pct}% extra off)\n"
                    "ALWAYS include the discount percentage next to the final price.\n"
                    "Example format: *Rs.X* per unit (*Y% extra off*)\n"
                    "Be warm and concise (max 6 lines). Use *bold* for prices and percentages.\n"
                    "End with: 'Would you like to proceed at this price?'\n"
                    f"Address customer as {sender}. Do NOT reveal the floor price.\n\n"
                    f"OFFER DETAILS:\n{context}"
                )},
                {"role": "user", "content": "Present the offer."},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[NEGOTIATOR] _reply_first_offer failed: {e}")
        return (
            f"Great news, {sender}! 🎉\n\n"
            f"For *{offer['quantity']} units* of *{product_name}*:\n"
            f"• Regular price: Rs.{regular_price:,.0f}\n"
            f"• Our price: Rs.{price_num:,.0f} ({graphrag_discount_pct}% off)\n"
            f"• Your price ({actual_offer_pct}% extra off): *Rs.{offer['offer_price']:,.0f}* per unit\n"
            f"• Total: *Rs.{offer['total_price']:,.0f}*\n\n"
            f"Would you like to proceed at this price?"
        )


async def _reply_counter_offer(
    sender: str,
    product_name: str,
    customer_price: float,
    new_offer: float,
    quantity: int,
    total: float,
    rounds: int,
    is_final: bool,
    biz_name: str,
) -> str:
    """Responds to customer's counter-offer with a midway price."""
    try:
        response = _client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 150,
            temperature = 0.4,
            messages    = [
                {"role": "system", "content": (
                    f"You are a sales negotiator for {biz_name}.\n"
                    f"Customer proposed Rs.{customer_price:,.0f} per unit.\n"
                    f"You can offer Rs.{new_offer:,.0f} per unit "
                    f"(Total Rs.{total:,.0f} for {quantity} units of {product_name}).\n"
                    + ("This is your FINAL offer — be firm but polite.\n" if is_final else "")
                    + "Be warm, concise (max 4 lines). Use *bold* for prices.\n"
                    f"Address customer as {sender}. End with 'Shall we proceed?'"
                )},
                {"role": "user", "content": "Give counter-offer response."},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[NEGOTIATOR] _reply_counter_offer failed: {e}")
        prefix = "This is my *final offer*" if is_final else "Here's what I can do"
        return (
            f"{prefix}, {sender}:\n\n"
            f"*{quantity} units* of *{product_name}* at *Rs.{new_offer:,.0f}* per unit\n"
            f"Total: *Rs.{total:,.0f}*\n\n"
            f"Shall we proceed?"
        )


async def _reply_below_floor(
    sender: str,
    product_name: str,
    customer_price: float,
    floor_price: float,
    quantity: int,
    biz_name: str,
) -> str:
    """Politely rejects counter-offer below floor and states the minimum."""
    try:
        floor_total = round(floor_price * quantity, 2)
        response = _client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 150,
            temperature = 0.4,
            messages    = [
                {"role": "system", "content": (
                    f"You are a sales negotiator for {biz_name}.\n"
                    f"Customer proposed Rs.{customer_price:,.0f} — below our minimum.\n"
                    f"Our absolute minimum: Rs.{floor_price:,.0f} per unit "
                    f"(Total Rs.{floor_total:,.0f} for {quantity} units of {product_name}).\n"
                    "Politely decline and offer the floor price.\n"
                    "Be warm, understanding. Max 4 lines. Use *bold* for prices.\n"
                    f"Address customer as {sender}. End with 'Shall we proceed at this price?'"
                )},
                {"role": "user", "content": "Give the rejection + floor price response."},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[NEGOTIATOR] _reply_below_floor failed: {e}")
        floor_total = round(floor_price * quantity, 2)
        return (
            f"I understand, {sender}, but Rs.{customer_price:,.0f} is below what we can offer. 🙏\n\n"
            f"Our absolute best price for *{quantity} units* of *{product_name}* is "
            f"*Rs.{floor_price:,.0f}* per unit (Total: *Rs.{floor_total:,.0f}*).\n\n"
            f"Shall we proceed at this price?"
        )


async def _reply_acceptance(
    sender: str,
    product_name: str,
    agreed_price: float,
    quantity: int,
    total: float,
    biz_name: str,
) -> str:
    """Confirms the deal and asks customer to reply 'Proceed'."""
    try:
        response = _client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 150,
            temperature = 0.4,
            messages    = [
                {"role": "system", "content": (
                    f"You are a sales assistant for {biz_name}.\n"
                    f"Customer agreed to the price for {product_name}.\n"
                    f"Agreed price: Rs.{agreed_price:,.0f} per unit\n"
                    f"Quantity: {quantity} units\n"
                    f"Total: Rs.{total:,.0f}\n"
                    "Confirm the deal warmly with a clear order summary.\n"
                    "End with: 'Please reply *Proceed* to confirm your order and receive your invoice!'\n"
                    f"Address customer as {sender}. Max 6 lines. Use *bold* for key info."
                )},
                {"role": "user", "content": "Generate deal confirmation."},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[NEGOTIATOR] _reply_acceptance failed: {e}")
        return (
            f"🎉 Great, {sender}! Deal confirmed!\n\n"
            f"*Order Summary:*\n"
            f"• Product: {product_name}\n"
            f"• Quantity: {quantity} units\n"
            f"• Price per unit: *Rs.{agreed_price:,.0f}*\n"
            f"• Total: *Rs.{total:,.0f}*\n\n"
            f"Please reply *Proceed* to confirm your order and receive your invoice!"
        )


async def _reply_escalate(
    sender: str,
    product_name: str,
    last_offer: float,
    quantity: int,
    biz_name: str,
) -> str:
    """Firm final response — we cannot go lower, no escalation to human."""
    try:
        total = round(last_offer * quantity, 2)
        response = _client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 120,
            temperature = 0.3,
            messages    = [
                {"role": "system", "content": (
                    f"You are a sales negotiator for {biz_name}.\n"
                    f"After negotiation for {product_name}, "
                    f"our absolute best price is Rs.{last_offer:,.0f}/unit "
                    f"(Total Rs.{total:,.0f} for {quantity} units).\n"
                    "Firmly tell the customer this is our lowest price.\n"
                    "Do NOT mention any sales team or escalation.\n"
                    "Give them the option to accept or decline this price.\n"
                    f"Address customer as {sender}. Max 3 lines. Use *bold* for prices."
                )},
                {"role": "user", "content": "Generate firm final price response."},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[NEGOTIATOR] _reply_escalate failed: {e}")
        total = round(last_offer * quantity, 2)
        return (
            f"{sender}, *Rs.{last_offer:,.0f}/unit* is our absolute best price for "
            f"*{product_name}* (Total: *Rs.{total:,.0f}* for {quantity} units). 🙏\n\n"
            f"We are unable to go lower than this. Would you like to proceed?"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN NEGOTIATION HANDLER
# ══════════════════════════════════════════════════════════════════════════════

async def handle_negotiation(
    incoming,
    product_name:          str,
    price_num:             float,
    regular_price:         float,
    graphrag_discount_pct: int,
    session_history:       list,
    negotiation_state:     dict,
    global_offers:         str = None,
) -> dict:
    """
    Core negotiation handler.
    Floor = second tier from global_offers (5%). Spread over 3-4 turns.
    Handles both total-price and per-unit counter-offers.
    """
    msg      = incoming.text
    sender   = incoming.sender_name
    biz_name = incoming.biz_name
    rounds   = negotiation_state.get("rounds", 0)
    quantity = negotiation_state.get("quantity")
    had_existing_quantity = bool(quantity)

    _cached_tiers = negotiation_state.get("_tiers")
    if _cached_tiers is not None:
        tiers = _cached_tiers
    elif global_offers:
        tiers = parse_global_offer_tiers(global_offers)
        print(f"[NEGOTIATOR] Parsed tiers: {tiers}")
    else:
        tiers = []

    floor_disc  = get_negotiation_floor_disc(tiers)
    floor_price = round(price_num * (1 - floor_disc / 100), 2)
    awaiting_qty = negotiation_state.get("awaiting_quantity", False)

    def _updated_state(**kwargs) -> dict:
        return {
            **negotiation_state,
            "product_name":  product_name,
            "price_num":     price_num,
            "floor_price":   floor_price,
            "global_offers": global_offers,
            "_tiers":        tiers,
            **kwargs,
        }

    # ── Step 1: We asked for quantity last turn — parse it now ────────────────
    if awaiting_qty:
        quantity = await extract_quantity(msg, product_name, session_history)

        if not quantity:
            # Customer didn't give a number — ask again
            reply = (
                f"I didn't catch that, {sender}. "
                f"How many units of *{product_name}* would you like to buy?"
            )
            return {
                "reply":        reply,
                "state":        _updated_state(awaiting_quantity=True, rounds=rounds),
                "order_ready":  False,
                "escalate":     False,
                "agreed_price": None,
                "quantity":     None,
            }

        # Got quantity — check if it qualifies for any discount
        offer = calculate_offer(price_num, quantity, tiers)

        if not offer["has_discount"]:
            # Less than 5 units — no extra discount
            reply = await _reply_no_discount(
                sender, product_name, price_num, regular_price,
                graphrag_discount_pct, quantity, biz_name
            )
            return {
                "reply":        reply,
                "state":        _updated_state(
                    quantity         = quantity,
                    awaiting_quantity = False,
                    rounds           = rounds,
                    last_offer_price = price_num,
                ),
                "order_ready":  False,
                "escalate":     False,
                "agreed_price": None,
                "quantity":     quantity,
            }

        # Qualifies for tier discount — present first offer
        rounds += 1
        reply = await _reply_first_offer(
            sender, product_name, price_num, regular_price,
            graphrag_discount_pct, offer, biz_name
        )
        return {
            "reply":        reply,
            "state":        _updated_state(
                quantity          = quantity,
                awaiting_quantity = False,
                rounds            = rounds,
                last_offer_price  = offer["offer_price"],
                floor_price       = offer["floor_price"],
            ),
            "order_ready":  False,
            "escalate":     False,
            "agreed_price": None,
            "quantity":     quantity,
        }

    # ── Step 2: No quantity yet — try current message first, then session history ─
    # Priority order:
    # 1. Quantity in current message: "I want 10 units at a discount" → qty=10
    # 2. Quantity from recent session history: customer said "50" two messages ago
    # 3. Only ask if no quantity found anywhere
    if not quantity:
        inline_qty = await extract_quantity(msg, product_name, session_history)
        if inline_qty:
            print(f"[NEGOTIATOR] Quantity found inline in message: {inline_qty} units")
            quantity = inline_qty
            # Fall through to Step 5 with this quantity
        else:
            # No quantity in current message — check session history
            # LLM scans last few messages to find the most recently mentioned quantity
            history_qty = None
            if session_history:
                try:
                    hist_resp = _client.chat.completions.create(
                        model       = AZURE_OPENAI_DEPLOYMENT,
                        max_tokens  = 10,
                        temperature = 0,
                        messages    = [
                            {"role": "system", "content": (
                                f"The customer is discussing buying '{product_name}'.\n"
                                "Look through the conversation history and find the most "
                                "recently mentioned quantity/number of units the customer wants.\n"
                                "Reply with ONLY the integer number, or 'NONE' if no quantity was mentioned."
                            )},
                        ] + session_history[-6:] + [
                            {"role": "user", "content": "What quantity did the customer most recently mention?"}
                        ],
                    )
                    hist_raw = hist_resp.choices[0].message.content.strip().upper()
                    if hist_raw != "NONE" and hist_raw.isdigit():
                        history_qty = int(hist_raw)
                        if history_qty > 0:
                            print(f"[NEGOTIATOR] Quantity found in session history: {history_qty} units")
                except Exception as e:
                    print(f"[NEGOTIATOR] History quantity lookup failed: {e}")

            if history_qty:
                quantity = history_qty
                # Fall through to Step 5 with history quantity
            else:
                reply = await _reply_ask_quantity(
                    sender, product_name, price_num, regular_price,
                    graphrag_discount_pct, biz_name
                )
                return {
                    "reply":        reply,
                    "state":        _updated_state(awaiting_quantity=True, rounds=rounds),
                    "order_ready":  False,
                    "escalate":     False,
                    "agreed_price": None,
                    "quantity":     None,
                }

    # ── Step 2.5: Quantity was already locked in — check if THIS message ──────
    # is asking to change it ("add 1 more unit", "make it 6"). Must run before
    # Step 3/4 so a quantity-change request is never misread as acceptance or
    # a price counter-offer using the stale quantity.
    if had_existing_quantity:
        new_quantity = await detect_quantity_change(msg, quantity, product_name, session_history)
        if new_quantity:
            print(f"[NEGOTIATOR] Quantity change detected: {quantity} -> {new_quantity}")
            quantity    = new_quantity
            floor_price = round(price_num * (1 - floor_disc / 100), 2)
            offer       = calculate_offer(price_num, quantity, tiers)
            rounds     += 1

            # ── Check if this is ONLY a quantity change (no discount request) ──
            # Focused LLM call with NO session_history — history has upsell
            # messages like "Order 1 more to unlock 8%!" which confuse the
            # LLM into treating pure quantity changes as discount requests.
            _is_discount_req = True  # safe default
            try:
                _dc = _client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT, max_tokens=5, temperature=0,
                    messages=[
                        {"role": "system", "content": (
                            "Does this message ask for a LOWER PRICE, discount, or negotiation?\n"
                            "YES: 'give me a discount', 'can I get extra off', "
                            "'reduce the price', 'any additional discount'\n"
                            "NO: 'add 2 more units', 'ok then add 4 units', "
                            "'make it 9 units', 'then add 1 more unit', 'add one more'\n"
                            "Reply ONLY 'YES' or 'NO'."
                        )},
                        {"role": "user", "content": msg},
                    ],
                )
                _is_discount_req = "YES" in _dc.choices[0].message.content.strip().upper()
            except Exception as _dce:
                print(f"[NEGOTIATOR] discount-check failed: {_dce}")
                _is_discount_req = False

            if not _is_discount_req:
                # Pure quantity change — auto-apply the applicable global offer tier
                order_value   = price_num * quantity
                _active_tiers = tiers
                if not _active_tiers:
                    try:
                        from db.session_store import get_tenant_offers as _gto
                        _to = await _gto(getattr(incoming, "tenant_id", ""))
                        if _to and _to.get("offers_text"):
                            _active_tiers = parse_global_offer_tiers(_to["offers_text"])
                    except Exception as _te:
                        print(f"[NEGOTIATOR] tenant_offers fetch failed: {_te}")
                _, auto_disc = get_applicable_tier(order_value, _active_tiers) if _active_tiers else (0, 0)

                if auto_disc > 0:
                    # Tier applies — compute auto-offer price
                    auto_price = round(price_num * (1 - auto_disc / 100), 2)
                    auto_total = round(auto_price * quantity, 2)
                    next_t     = get_next_tier(order_value, _active_tiers) if _active_tiers else None
                    upsell     = ""
                    if next_t:
                        units_to_next = max(1, int((next_t[0] - order_value) / price_num) + 1)
                        upsell = (f"\nOrder {units_to_next} more unit(s) to reach "
                                  f"Rs.{next_t[0]:,} and unlock {next_t[1]}% off!")

                    try:
                        resp = _client.chat.completions.create(
                            model=AZURE_OPENAI_DEPLOYMENT, max_tokens=250, temperature=0.3,
                            messages=[
                                {"role": "system", "content": (
                                    f"You are a sales assistant for {biz_name}.\n"
                                    f"Customer updated to {quantity} units.\n"
                                    f"The store's {auto_disc}% offer tier is now automatically applied.\n"
                                    f"Show a clean order summary:\n"
                                    f"- Product: {product_name}\n"
                                    f"- Quantity: {quantity} units\n"
                                    f"- Regular price: Rs.{price_num:,.2f}/unit\n"
                                    f"- {auto_disc}% store offer applied: Rs.{auto_price:,.2f}/unit\n"
                                    f"- Total: Rs.{auto_total:,.2f}\n"
                                    f"{'Upsell: ' + upsell if upsell else ''}\n"
                                    f"End with 'Please confirm and we'll process your order!'\n"
                                    f"Address as {sender}. Use *bold* for prices."
                                )},
                                {"role": "user", "content": msg},
                            ],
                        )
                        reply = resp.choices[0].message.content.strip()
                    except Exception:
                        reply = (
                            f"Updated order for {sender}! 🎉\n\n"
                            f"• *Product:* {product_name}\n"
                            f"• *Quantity:* {quantity} units\n"
                            f"• *{auto_disc}% store offer applied:* *Rs.{auto_price:,.2f}/unit*\n"
                            f"• *Total: Rs.{auto_total:,.2f}*\n"
                            + (upsell if upsell else "")
                            + "\n\nPlease confirm and we'll process your order! 🎉"
                        )

                    return {
                        "reply":        reply,
                        "state":        _updated_state(
                            quantity              = quantity,
                            rounds                = rounds,
                            last_offer_price      = auto_price,
                            floor_price           = floor_price,
                            awaiting_quantity     = False,
                            auto_offer_unit_price = auto_price,
                            auto_offer_disc_pct   = auto_disc,
                        ),
                        "order_ready":  False,
                        "escalate":     False,
                        "agreed_price": None,
                        "quantity":     quantity,
                    }
                else:
                    # No tier applies for this order value — show plain order summary
                    try:
                        resp = _client.chat.completions.create(
                            model=AZURE_OPENAI_DEPLOYMENT, max_tokens=200, temperature=0.3,
                            messages=[
                                {"role": "system", "content": (
                                    f"Customer updated order to {quantity} units of {product_name}.\n"
                                    f"Show a clean order summary: Rs.{price_num:,.2f}/unit × {quantity} "
                                    f"= Rs.{price_num * quantity:,.2f} total.\n"
                                    f"No extra discount applies yet.\n"
                                    f"End with 'Please confirm and we'll process your order!'\n"
                                    f"Address as {sender}. Use *bold* for prices."
                                )},
                                {"role": "user", "content": msg},
                            ],
                        )
                        reply = resp.choices[0].message.content.strip()
                    except Exception:
                        reply = (
                            f"Updated order for {sender}:\n"
                            f"• *{product_name}* × {quantity} units\n"
                            f"• Rs.{price_num:,.2f}/unit × {quantity} = *Rs.{price_num * quantity:,.2f}*\n\n"
                            f"Please confirm and we'll process your order! 🎉"
                        )
                    return {
                        "reply":        reply,
                        "state":        _updated_state(
                            quantity          = quantity,
                            rounds            = rounds,
                            last_offer_price  = price_num,
                            floor_price       = floor_price,
                            awaiting_quantity = False,
                        ),
                        "order_ready":  False,
                        "escalate":     False,
                        "agreed_price": None,
                        "quantity":     quantity,
                    }

            # Discount was also requested with the quantity change → fall through to negotiation
            if not offer["has_discount"]:
                reply = await _reply_no_discount(
                    sender, product_name, price_num, regular_price,
                    graphrag_discount_pct, quantity, biz_name
                )
                return {
                    "reply":        reply,
                    "state":        _updated_state(
                        quantity          = quantity,
                        rounds            = rounds,
                        last_offer_price  = price_num,
                        floor_price       = floor_price,
                        awaiting_quantity = False,
                    ),
                    "order_ready":  False,
                    "escalate":     False,
                    "agreed_price": None,
                    "quantity":     quantity,
                }

            reply = await _reply_first_offer(
                sender, product_name, price_num, regular_price,
                graphrag_discount_pct, offer, biz_name
            )
            return {
                "reply":        reply,
                "state":        _updated_state(
                    quantity          = quantity,
                    rounds            = rounds,
                    last_offer_price  = offer["offer_price"],
                    floor_price       = offer["floor_price"],
                    awaiting_quantity = False,
                ),
                "order_ready":  False,
                "escalate":     False,
                "agreed_price": None,
                "quantity":     quantity,
            }

    # ── Step 3: We have quantity — check acceptance first ─────────────────
    if rounds > 0 and await detect_acceptance(msg, session_history):
        last_offer = negotiation_state.get("last_offer_price", price_num)

        # KEY FIX: If customer stated a SPECIFIC price in their acceptance message
        # (e.g. "We go for 1870 that's the final price"), use THEIR price --
        # not our last_offer_price. Only honour it if their price >= floor.
        customer_stated_price = await detect_counter_offer(msg, session_history, quantity=quantity, current_price_num=price_num)
        if customer_stated_price is not None and customer_stated_price >= floor_price:
            agreed_price = customer_stated_price
            print(f"[NEGOTIATOR] Customer stated price Rs.{customer_stated_price} >= floor Rs.{floor_price} -- using customer price")
        else:
            agreed_price = last_offer
            print(f"[NEGOTIATOR] No valid customer price stated -- using last_offer Rs.{last_offer}")

        total = round(agreed_price * quantity, 2)
        reply = await _reply_acceptance(
            sender, product_name, agreed_price, quantity, total, biz_name
        )
        return {
            "reply":        reply,
            "state":        _updated_state(quantity=quantity, rounds=rounds, last_offer_price=agreed_price),
            "order_ready":  True,
            "escalate":     False,
            "agreed_price": agreed_price,
            "quantity":     quantity,
        }

    # ── Step 4: Check for counter-offer or "more discount" request ───────────
    if rounds > 0:
        # First check: customer asking for more discount without a specific price
        # e.g. "any more discount more than 10%?" "can you do better?"
        if await detect_more_discount_request(msg, session_history):
            last_offer    = negotiation_state.get("last_offer_price", price_num)
            order_value  = price_num * quantity
            _, cur_disc  = get_applicable_tier(order_value, tiers) if tiers else (0, 0)
            already_at_floor = round(last_offer, 2) <= round(floor_price, 2)

            already_at_floor = round(last_offer, 2) <= round(floor_price, 2)

            # Build upsell hint from real global offer tiers
            order_value  = price_num * quantity
            next_t       = get_next_tier(order_value, tiers) if tiers else None
            if next_t:
                units_needed  = max(1, int((next_t[0] - order_value) / price_num) + 1)
                next_tier_msg = (
                    f"order {units_needed} more unit(s) to reach Rs.{next_t[0]:,} "
                    f"and unlock {next_t[1]}% extra off automatically at checkout"
                )
            else:
                next_tier_msg = f"you already have the maximum {floor_disc}% extra discount available"

            if already_at_floor or last_offer <= floor_price:
                try:
                    resp = _client.chat.completions.create(
                        model       = AZURE_OPENAI_DEPLOYMENT,
                        max_tokens  = 120,
                        temperature = 0.4,
                        messages    = [
                            {"role": "system", "content": (
                                f"You are a friendly sales assistant for {biz_name}.\n"
                                f"Customer has {quantity} units and already has the best "
                                f"price of Rs.{last_offer:,.0f}/unit ({floor_disc}% extra off).\n"
                                f"This is the maximum negotiated discount.\n"
                                f"Tip for more savings: {next_tier_msg}.\n"
                                "Politely explain this and mention how they can get more.\n"
                                f"Address as {sender}. Max 3 lines. Use *bold* for prices."
                            )},
                            {"role": "user", "content": "Explain the discount limit."},
                        ],
                    )
                    reply = resp.choices[0].message.content.strip()
                except Exception:
                    reply = (
                        f"{sender}, *Rs.{last_offer:,.0f}/unit* is the best price "
                        f"for your order ({floor_disc}% extra off). "
                        f"Tip: {next_tier_msg.capitalize()}! "
                        f"Would you like to proceed at this price?"
                    )

                return {
                    "reply":        reply,
                    "state":        _updated_state(quantity=quantity, rounds=rounds),
                    "order_ready":  False,
                    "escalate":     False,
                    "agreed_price": None,
                    "quantity":     quantity,
                }

            # Customer not yet at max tier — move midway toward floor as goodwill
            rounds   += 1
            is_final  = rounds >= MAX_NEGOTIATION_ROUNDS
            new_offer = max(round((last_offer + floor_price) / 2, 2), floor_price)
            total     = round(new_offer * quantity, 2)
            print(f"[NEGOTIATOR] More discount — Rs.{last_offer} → Rs.{new_offer} (floor=Rs.{floor_price})")
            reply = await _reply_counter_offer(
                sender, product_name, last_offer, new_offer,
                quantity, total, rounds, is_final, biz_name
            )
            return {
                "reply":        reply,
                "state":        _updated_state(
                    quantity          = quantity,
                    rounds            = rounds,
                    last_offer_price  = new_offer,
                    awaiting_quantity = False,
                ),
                "order_ready":  False,
                "escalate":     False,
                "agreed_price": None,
                "quantity":     quantity,
            }

        counter_price = await detect_counter_offer(msg, session_history, quantity=quantity, current_price_num=price_num)

        if counter_price is not None:
            rounds += 1
            is_final  = rounds >= MAX_NEGOTIATION_ROUNDS
            last_offer = negotiation_state.get("last_offer_price", price_num)

            if counter_price < floor_price:
                if is_final:
                    # Rounds exhausted AND below floor — now reveal minimum firmly.
                    # Floor is only shown after full negotiation, never on first ask.
                    floor_total = round(floor_price * quantity, 2)
                    try:
                        firm_resp = _client.chat.completions.create(
                            model       = AZURE_OPENAI_DEPLOYMENT,
                            max_tokens  = 150,
                            temperature = 0.3,
                            messages    = [
                                {"role": "system", "content": (
                                    f"You are a sales negotiator for {biz_name}.\n"
                                    f"After {rounds} rounds of negotiation, customer proposed Rs.{counter_price:,.0f} which is below our minimum.\n"
                                    f"Our absolute minimum is Rs.{floor_price:,.0f}/unit "
                                    f"(Total Rs.{floor_total:,.0f} for {quantity} units of {product_name}).\n"
                                    "Firmly but politely tell the customer this is the lowest we can go.\n"
                                    "We cannot provide a lower price under any circumstances.\n"
                                    "Do NOT mention escalation or sales team.\n"
                                    "Give them two clear options: accept the floor price or decline.\n"
                                    f"Address as {sender}. Max 3 lines. Use *bold* for prices."
                                )},
                                {"role": "user", "content": "Give the firm final response."},
                            ],
                        )
                        reply = firm_resp.choices[0].message.content.strip()
                    except Exception:
                        reply = (
                            f"{sender}, we truly cannot go below *Rs.{floor_price:,.0f}/unit*. 🙏\n\n"
                            f"That's our absolute best price for {quantity} units of *{product_name}* "
                            f"(Total: *Rs.{floor_total:,.0f}*).\n\n"
                            f"Would you like to proceed at *Rs.{floor_price:,.0f}/unit*?"
                        )
                    return {
                        "reply":        reply,
                        "state":        _updated_state(
                            quantity          = quantity,
                            rounds            = 1,
                            last_offer_price  = floor_price,
                            awaiting_quantity = False,
                        ),
                        "order_ready":  False,
                        "escalate":     False,
                        "agreed_price": None,
                        "quantity":     quantity,
                    }
                else:
                    # Below floor but rounds NOT exhausted — keep negotiating.
                    # Move midway between last_offer and floor WITHOUT revealing floor.
                    new_offer = max(round((last_offer + floor_price) / 2, 2), floor_price)
                    total     = round(new_offer * quantity, 2)
                    print(f"[NEGOTIATOR] Below floor, rounds={rounds}/{MAX_NEGOTIATION_ROUNDS} — countering Rs.{new_offer} (floor Rs.{floor_price} not revealed yet)")
                    reply = await _reply_counter_offer(
                        sender, product_name, counter_price, new_offer,
                        quantity, total, rounds, False, biz_name
                    )
            else:
                # Above floor — meet midway between last_offer and customer price
                midway    = round((last_offer + counter_price) / 2, 2)
                new_offer = max(midway, floor_price)
                total     = round(new_offer * quantity, 2)
                reply     = await _reply_counter_offer(
                    sender, product_name, counter_price, new_offer,
                    quantity, total, rounds, is_final, biz_name
                )

            return {
                "reply":        reply,
                "state":        _updated_state(
                    quantity          = quantity,
                    rounds            = rounds,
                    last_offer_price  = new_offer,
                    awaiting_quantity = False,
                ),
                "order_ready":  False,
                "escalate":     False,
                "agreed_price": None,
                "quantity":     quantity,
            }

    # ── Step 5: First time — present tier offer ───────────────────────────────
    offer  = calculate_offer(price_num, quantity, tiers)
    rounds += 1

    if not offer["has_discount"]:
        reply = await _reply_no_discount(
            sender, product_name, price_num, regular_price,
            graphrag_discount_pct, quantity, biz_name
        )
        return {
            "reply":        reply,
            "state":        _updated_state(
                quantity          = quantity,
                rounds            = rounds,
                last_offer_price  = price_num,
                awaiting_quantity = False,
            ),
            "order_ready":  False,
            "escalate":     False,
            "agreed_price": None,
            "quantity":     quantity,
        }

    reply = await _reply_first_offer(
        sender, product_name, price_num, regular_price,
        graphrag_discount_pct, offer, biz_name
    )
    return {
        "reply":        reply,
        "state":        _updated_state(
            quantity          = quantity,
            rounds            = rounds,
            last_offer_price  = offer["offer_price"],
            floor_price       = offer["floor_price"],
            awaiting_quantity = False,
        ),
        "order_ready":  False,
        "escalate":     False,
        "agreed_price": None,
        "quantity":     quantity,
    }