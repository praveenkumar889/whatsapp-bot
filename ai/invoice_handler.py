# ai/invoice_handler.py — Invoice and order confirmation handling
#
# Extracted from main.py to keep the orchestrator lightweight.

from typing import Optional
from openai import AzureOpenAI

from config import (
    AZURE_AI_ENDPOINT, AZURE_AI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT, AZURE_AI_API_VERSION,
)
from db.session_store import (
    get_negotiation_state,
    save_negotiation_state,
    save_outbound_message,
)
from adapter.whatsapp_adapter import send_whatsapp_reply

_client = AzureOpenAI(
    azure_endpoint = AZURE_AI_ENDPOINT,
    api_key        = AZURE_AI_API_KEY,
    api_version    = AZURE_AI_API_VERSION,
    timeout        = 30.0,
    max_retries    = 0,
)

async def _is_invoice_inquiry(message: str) -> bool:
    """
    Uses LLM to determine if the customer's message is asking for their invoice,
    bill, receipt, or payment document. Zero hardcoding.
    """
    try:
        response = _client.chat.completions.create(
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
        response = _client.chat.completions.create(
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
        response = _client.chat.completions.create(
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
        response = _client.chat.completions.create(
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
            support_phone = getattr(incoming, "support_phone", None),
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