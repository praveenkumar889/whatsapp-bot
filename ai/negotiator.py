# ai/negotiator.py — Price Negotiator Module
#
# FLOW:
#   1. Customer sees product list with price_num + regular_price + discount%
#      e.g. "Rs.2,653 (Save 25% off Rs.3,538)"
#   2. Customer asks for a discount
#   3. Bot asks: "How many units are you looking to buy?"
#   4. Customer replies with quantity
#   5. Bot applies tier discount on price_num (NOT regular_price):
#        1–4  units → 0%  off price_num (no extra discount)
#        5–9  units → 5%  off price_num
#        10–14 units → 10% off price_num
#        ≥15  units → 15% off price_num  ← also the floor_price
#   6. Customer can counter-offer (max 3 rounds)
#   7. Customer accepts → order summary → "Reply Proceed to confirm"
#   8. Customer says Proceed → create_order() + invoice
#
# HARDCODED (intentionally — these are fixed business rules):
#   - Tier thresholds: 5, 10, 15
#   - Tier discounts:  5%, 10%, 15%
#   - Floor multiplier: 0.85 (15% off price_num)
#   - Max rounds: 3
#
# ZERO HARDCODING of:
#   - Business names, product names, prices (all from DB/GraphRAG)
#   - Customer names (from incoming object)
#   - Reply messages (all LLM-generated)

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

# ── Business rules — correctly hardcoded ──────────────────────────────────────
MAX_NEGOTIATION_ROUNDS = 3
FLOOR_MULTIPLIER       = 0.85   # floor_price = price_num × 0.85


# ══════════════════════════════════════════════════════════════════════════════
# TIER LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def get_tier_discount(quantity: int) -> float:
    """
    Returns extra discount % based on quantity.
    Applied on price_num (GraphRAG discounted price) — NOT on regular_price.

    Tiers (business rules — hardcoded intentionally):
        1–4   → 0.00  (no extra discount)
        5–9   → 0.05  (5% off)
        10–14 → 0.10  (10% off)
        ≥15   → 0.15  (15% off) ← same as floor
    """
    if quantity >= 15:
        return 0.15
    elif quantity >= 10:
        return 0.10
    elif quantity >= 5:
        return 0.05
    else:
        return 0.00


def calculate_offer(price_num: float, quantity: int) -> dict:
    """
    Calculates the first offer price and floor price for a given quantity.

    KEY DESIGN — negotiation room:
        Floor  = tier price (best we can do for this quantity, never revealed upfront)
        Offer  = midway between price_num and floor (starting point for negotiation)

        Example (price_num=Rs.789, quantity=10, tier=10%):
            floor  = Rs.789 × 0.90 = Rs.710  ← minimum, not shown first
            offer  = (Rs.789 + Rs.710) / 2   = Rs.750  ← shown to customer first

        Customer negotiates Rs.750 → Rs.730 → Rs.710 (floor, final offer)
        This feels like a real negotiation — not a price list.
    """
    tier_discount = get_tier_discount(quantity)

    if tier_discount == 0:
        floor_price = price_num
        offer_price = price_num
    else:
        # Floor = best possible price for this tier
        floor_price = round(price_num * (1 - tier_discount), 2)
        # Start offer midway — gives 2-3 rounds of negotiation room
        offer_price = round((price_num + floor_price) / 2, 2)

    total_price       = round(offer_price * quantity, 2)
    tier_discount_pct = int(tier_discount * 100)

    return {
        "offer_price":       offer_price,
        "total_price":       total_price,
        "floor_price":       floor_price,
        "tier_discount_pct": tier_discount_pct,
        "has_discount":      tier_discount > 0,
        "price_num":         price_num,
        "quantity":          quantity,
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


async def detect_counter_offer(
    message: str,
    session_history: list = None,
) -> Optional[float]:
    """
    Detects if customer is making a counter-offer with a specific price.
    Returns the price per unit they proposed, or None if no specific price.

    Examples:
        "Can you do 2500?"     → 2500.0
        "How about Rs.2,200?"  → 2200.0
        "Still too high"       → None
    """
    try:
        messages = [
            {"role": "system", "content": (
                "The customer may be proposing a specific price per unit.\n"
                "Extract the price per unit they are suggesting.\n"
                "Strip currency symbols and commas.\n"
                "Reply with ONLY the number (e.g. 2500), or 'NONE' if no specific price."
            )},
        ]
        if session_history:
            messages.extend(session_history[-4:])
        messages.append({"role": "user", "content": message})

        response = _client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 15,
            temperature = 0,
            messages    = messages,
        )
        raw = response.choices[0].message.content.strip().upper()
        if raw == "NONE":
            return None
        cleaned = raw.replace("RS.", "").replace("₹", "").replace(",", "").strip()
        return float(cleaned) if cleaned.replace(".", "").isdigit() else None
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
    product_name:      str,
    price_num:         float,
    regular_price:     float,
    graphrag_discount_pct: int,
    session_history:   list,
    negotiation_state: dict,
) -> dict:
    """
    Core negotiation handler called from main.py.

    Args:
        incoming:              IncomingMessage object
        product_name:          Product being negotiated
        price_num:             GraphRAG discounted price (e.g. Rs.2,653)
        regular_price:         Original price before GraphRAG discount (e.g. Rs.3,538)
        graphrag_discount_pct: GraphRAG's already-applied discount % (e.g. 25)
        session_history:       Last N messages for LLM context
        negotiation_state:     Current state from DB:
            {
                "rounds":           int,    # negotiation rounds so far
                "quantity":         int,    # units customer wants (None until provided)
                "last_offer_price": float,  # last price we offered
                "floor_price":      float,  # absolute minimum
                "product_name":     str,
                "price_num":        float,
                "awaiting_quantity": bool,  # True if we asked for units and waiting
            }

    Returns dict:
        "reply":        str            — message to send customer
        "state":        dict           — updated state to save to DB
        "order_ready":  bool           — True if customer accepted → trigger order
        "escalate":     bool           — True → escalate to human
        "agreed_price": float | None   — final agreed price per unit
        "quantity":     int | None     — confirmed quantity
    """
    msg          = incoming.text
    sender       = incoming.sender_name
    biz_name     = incoming.biz_name
    rounds       = negotiation_state.get("rounds", 0)
    quantity     = negotiation_state.get("quantity")
    # Floor MUST be calculated from actual quantity tier — never use fixed multiplier.
    # 10 units → tier=10% → floor=Rs.710 (not Rs.671 which is 15% tier)
    # Recalculate from quantity every time to prevent stale state bugs.
    _saved_floor = negotiation_state.get("floor_price")
    if quantity and _saved_floor:
        # Verify saved floor matches current quantity tier
        _expected_floor = round(price_num * (1 - get_tier_discount(quantity)), 2)
        floor_price = _expected_floor  # always use tier-based floor
    elif quantity:
        floor_price = round(price_num * (1 - get_tier_discount(quantity)), 2)
    else:
        floor_price = _saved_floor or round(price_num * FLOOR_MULTIPLIER, 2)
    awaiting_qty = negotiation_state.get("awaiting_quantity", False)

    def _updated_state(**kwargs) -> dict:
        return {
            **negotiation_state,
            "product_name":     product_name,
            "price_num":        price_num,
            "floor_price":      floor_price,
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
        offer = calculate_offer(price_num, quantity)

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

    # ── Step 3: We have quantity — check acceptance first ─────────────────
    if rounds > 0 and await detect_acceptance(msg, session_history):
        last_offer = negotiation_state.get("last_offer_price", price_num)

        # KEY FIX: If customer stated a SPECIFIC price in their acceptance message
        # (e.g. "We go for 1870 that's the final price"), use THEIR price --
        # not our last_offer_price. Only honour it if their price >= floor.
        customer_stated_price = await detect_counter_offer(msg, session_history)
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
            tier_discount = get_tier_discount(quantity)
            tier_pct      = int(tier_discount * 100)

            # Floor is always recalculated from quantity tier — never trust saved value
            floor_price   = round(price_num * (1 - tier_discount), 2)

            # already_at_max_tier = we already reached the floor for this quantity
            # Use floor_price (tier-based) not any saved floor from old state
            already_at_max_tier = round(last_offer, 2) <= round(floor_price, 2)

            if already_at_max_tier or last_offer <= floor_price:
                # Explain tier limits and what quantity unlocks next tier
                if quantity < 5:
                    next_tier_msg = "buying 5+ units unlocks a 5% extra discount"
                elif quantity < 10:
                    next_tier_msg = "buying 10+ units unlocks a 10% extra discount"
                elif quantity < 15:
                    next_tier_msg = f"buying 15+ units unlocks a 15% extra discount"
                else:
                    next_tier_msg = "you already have our maximum quantity discount"

                try:
                    resp = _client.chat.completions.create(
                        model       = AZURE_OPENAI_DEPLOYMENT,
                        max_tokens  = 120,
                        temperature = 0.4,
                        messages    = [
                            {"role": "system", "content": (
                                f"You are a friendly sales assistant for {biz_name}.\n"
                                f"Customer has {quantity} units and already has the best "
                                f"price of Rs.{last_offer:,.0f}/unit ({tier_pct}% extra off).\n"
                                f"This is the maximum discount for {quantity} units.\n"
                                f"Tip for more discount: {next_tier_msg}.\n"
                                "Politely explain this and mention how they can get more.\n"
                                f"Address as {sender}. Max 3 lines. Use *bold* for prices."
                            )},
                            {"role": "user", "content": "Explain the tier limit."},
                        ],
                    )
                    reply = resp.choices[0].message.content.strip()
                except Exception:
                    tip = next_tier_msg.capitalize()
                    reply = (
                        f"{sender}, *Rs.{last_offer:,.0f}/unit* is the best price "
                        f"for {quantity} units ({tier_pct}% extra off). "
                        f"Tip: {tip}! "
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

        counter_price = await detect_counter_offer(msg, session_history)

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
    offer  = calculate_offer(price_num, quantity)
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