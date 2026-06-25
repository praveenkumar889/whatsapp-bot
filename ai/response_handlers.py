# ai/response_handlers.py — Simple intent handlers
#
# Extracted from main.py to keep the orchestrator lightweight.
# Contains: handle_greeting, handle_escalation, handle_unknown

import json
from datetime import datetime, timezone

from openai import AzureOpenAI
from config import (
    AZURE_AI_ENDPOINT, AZURE_AI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT, AZURE_AI_API_VERSION,
)
from db.session_store import get_tenant_offers

_client = AzureOpenAI(
    azure_endpoint = AZURE_AI_ENDPOINT,
    api_key        = AZURE_AI_API_KEY,
    api_version    = AZURE_AI_API_VERSION,
    timeout        = 30.0,
    max_retries    = 0,
)


async def handle_escalation(incoming) -> str:
    """
    Handles HUMAN_ESCALATION intent — customer is upset or needs a human.
    GPT reads the specific complaint and acknowledges it empathetically.
    """
    try:
        response = _client.chat.completions.create(
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
        response = _client.chat.completions.create(
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
        response = _client.chat.completions.create(
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