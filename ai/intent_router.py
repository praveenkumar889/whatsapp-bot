# ai/intent_router.py — LLM Intent Classification Engine with Session History

import json
from typing import List
from openai import AzureOpenAI

from models.schemas import IntentResult
from config import (
    AZURE_AI_ENDPOINT,
    AZURE_AI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_AI_API_VERSION,
    BUSINESS_NAME,
)

_client = AzureOpenAI(
    azure_endpoint = AZURE_AI_ENDPOINT,
    api_key        = AZURE_AI_API_KEY,
    api_version    = AZURE_AI_API_VERSION,
)

VALID_INTENTS = {"WORKFLOW_ACTION", "FAQ_KNOWLEDGE", "HUMAN_ESCALATION", "GREETING", "UNKNOWN"}

SYSTEM_PROMPT = f"""
You are an intent classification AI for {BUSINESS_NAME}, an order tracking and management platform.

Your job is to read a customer's WhatsApp message and classify it into exactly ONE of these intents:

WORKFLOW_ACTION   — Customer wants to DO something transactional:
                    place order using SKU code, track order, check order status,
                    request invoice, confirm payment, ask about a specific SKU product.
                    Examples:
                    - "I want to order 5 units of 10C-2012"
                    - "Where is my order #1023?"
                    - "What did I order?"
                    - "Features of 10c-2012"
                    - "I want to know about ALT20C"

FAQ_KNOWLEDGE     — Customer is asking about products by NAME or CATEGORY
                    (not by SKU code), or asking general product questions:
                    "gate lights", "solar lights", "outdoor lights", "what lights do you have",
                    product comparisons, general questions about types of lights.
                    Examples:
                    - "I want gate lights"
                    - "Tell me about solar wall lights"
                    - "What outdoor lights do you have?"
                    - "Show me divine lights"
                    - "I want to order solar lights"
                    - "What is the best light for my gate?"
                    - "Do you have bollard lights?"

HUMAN_ESCALATION  — Customer is upset or needs a human agent:
                    complaints, refund disputes, request to speak to manager.
                    Examples:
                    - "I want to talk to a manager"
                    - "My order was wrong and I am very angry"

GREETING          — Customer is greeting OR expressing thanks/farewell:
                    good morning, good afternoon, hi, hello, namaste,
                    thank you, thanks, bye, goodbye, see you, ok thanks.
                    Examples:
                    - "Hi"
                    - "Thank you"
                    - "Thanks a lot"
                    - "Bye"

UNKNOWN           — Message is gibberish, completely unclear, or out of scope.
                    Examples: "asdfgh", random text

KEY DISTINCTION — WORKFLOW_ACTION vs FAQ_KNOWLEDGE:
- Message has a SKU code (like 10C-2012, ALT20C, 24C-2055) → WORKFLOW_ACTION
- Message has a product category/name (like "gate lights", "solar lights") → FAQ_KNOWLEDGE
- Message is placing an order with SKU → WORKFLOW_ACTION
- Message is asking about/browsing products by name → FAQ_KNOWLEDGE

RULES:
1. Reply ONLY with valid JSON. No explanation, no markdown, no code fences.
2. Always include exactly two keys: "intent" and "confidence_score".
3. confidence_score must be a float between 0.0 and 1.0.
4. If confidence is below 0.50, set intent to "UNKNOWN".
5. Never invent a new intent name.

Output format:
{{"intent": "WORKFLOW_ACTION", "confidence_score": 0.97}}
"""


async def classify_intent(
    customer_message: str,
    session_history:  List[dict] = None,
) -> IntentResult:
    """
    Classifies a customer message into one of five intents.

    NOW ACCEPTS SESSION HISTORY:
        The last 10 messages of the conversation are passed as context.
        This allows the AI to understand:
        - "3kg" → after "how many units?" → WORKFLOW_ACTION (not UNKNOWN)
        - "chicken pickle" → after "which product?" → WORKFLOW_ACTION
        - "what did I order?" → WORKFLOW_ACTION with context

    Args:
        customer_message: The current message the customer sent.
        session_history:  List of previous messages in format:
                         [{"role": "user", "content": "..."}, ...]

    Returns:
        IntentResult — always, never raises.
    """
    raw = ""
    try:
        # Build messages array with history + current message
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add session history for context
        if session_history:
            messages.extend(session_history)
            # History gives the AI full conversation context so it can
            # understand follow-up messages like "3kg" or "chicken pickle"

        # Add current message
        messages.append({"role": "user", "content": customer_message})

        response = _client.chat.completions.create(
            model       = AZURE_OPENAI_DEPLOYMENT,
            max_tokens  = 60,
            temperature = 0,
            messages    = messages,
        )

        raw    = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        intent           = str(parsed.get("intent", "UNKNOWN")).upper()
        confidence_score = float(parsed.get("confidence_score", 0.0))
        confidence_score = max(0.0, min(1.0, confidence_score))

        if intent not in VALID_INTENTS:
            intent           = "UNKNOWN"
            confidence_score = 0.0

        if confidence_score < 0.50:
            intent = "UNKNOWN"

        print(f"[INTENT ROUTER] '{customer_message[:60]}' => {intent} ({confidence_score})")
        return IntentResult(
            intent           = intent,
            confidence_score = confidence_score,
            raw_text         = customer_message,
        )

    except json.JSONDecodeError as e:
        print(f"[INTENT ROUTER] JSON parse error: {e} | raw='{raw}'")
        return IntentResult(intent="UNKNOWN", confidence_score=0.0, raw_text=customer_message)

    except Exception as e:
        print(f"[INTENT ROUTER ERROR] {e}")
        return IntentResult(intent="UNKNOWN", confidence_score=0.0, raw_text=customer_message)