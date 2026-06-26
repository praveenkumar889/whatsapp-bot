from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class IncomingMessage:
    """
    Fully normalised platform-neutral message.
    Zero WhatsApp knowledge downstream — all internal systems only see this object.
    Produced by the Communication Adapter after parsing the raw Meta webhook payload.
    """

    # ── Tracing ───────────────────────────────────────────────────────────
    trace_id:        str
    message_id:      str
    session_id:      str
    channel:         str        # "whatsapp" — always for now
    timestamp:       int        # Unix timestamp from Meta — when customer pressed send

    # ── Tenant (required) ────────────────────────────────────────────────
    tenant_id:       str        # DB-resolved business isolation key
    waba_id:         str        # Meta WhatsApp Business Account ID
    phone_number_id: str        # Meta phone number ID — used in outbound reply URL
    biz_name:        str        # Business name — shown on invoice header
    region:          str        # "india" — data residency
    timezone:        str        # "Asia/Kolkata" — time-aware greetings
    language:        str        # "en" — future AI prompt language

    # ── Sender ────────────────────────────────────────────────────────────
    sender_name:     str        # "Praveen" — personalise replies
    sender_phone:    str        # "918897729577" — "to" field in outbound API

    # ── Message ───────────────────────────────────────────────────────────
    text:            str
    original_type:   str        # "text" | "image" | "audio"
    received_at:     str        # ISO UTC timestamp — server receipt time

    # ── Optional: Tenant invoice fields (populated from tenants table) ────
    # All None if not set in DB. Invoice.py skips missing fields gracefully.
    # MUST come after all required (non-default) fields — Python dataclass rule.
    tagline:         Optional[str] = None   # "LED Lighting Solutions | Made in India"
    city:            Optional[str] = None   # "Chennai, Tamil Nadu"
    support_email:   Optional[str] = None   # "support@inventaa.in"
    support_phone:   Optional[str] = None   # "+91 72990 39181"
    website:         Optional[str] = None   # "inventaa.in"
    upi_id:          Optional[str] = None   # "inventaa@upi"
    account_name:    Optional[str] = None   # "Inventaa LED Innovation Pvt Ltd"

    # ── Optional: Media (None for text messages) ──────────────────────────
    media_id:        Optional[str]   = None
    media_mime_type: Optional[str]   = None
    media_binary:    Optional[bytes] = None
    media_url:       Optional[str]   = None

    # ── Optional: Quoted Message Context ──────────────────────────────────
    quoted_message_id: Optional[str] = None  # wamid Meta sent in context.id
    quoted_caption:    Optional[str] = None  # resolved text of that message

    raw:             dict = field(default_factory=dict)


@dataclass
class IntentResult:
    """Output of the LLM intent router (ai/intent_router.py)."""
    intent:           str    # WORKFLOW_ACTION | FAQ_KNOWLEDGE | HUMAN_ESCALATION | GREETING | UNKNOWN
    confidence_score: float  # 0.0–1.0
    raw_text:         str    # original customer message


@dataclass
class OrderItem:
    """
    A single product line in a multi-product order.
    Produced by entity_extractor when customer orders multiple products.
    """
    product_name:   Optional[str]   # "LED flood light"
    quantity_value: Optional[int]   # 10
    quantity_unit:  Optional[str]   # "units"

    @property
    def is_complete(self) -> bool:
        """True if both product and quantity are known."""
        return self.product_name is not None and self.quantity_value is not None

    @property
    def missing(self) -> List[str]:
        """Returns list of missing fields for this item."""
        m = []
        if not self.product_name:
            m.append("product_name")
        if self.quantity_value is None:
            m.append("quantity")
        return m

    @property
    def quantity_str(self) -> Optional[str]:
        if self.quantity_value and self.quantity_unit:
            return f"{self.quantity_value} {self.quantity_unit}"
        elif self.quantity_value:
            return str(self.quantity_value)
        return None


@dataclass
class EntityResult:
    """
    Output of the entity extraction engine (ai/entity_extractor.py).

    MULTI-PRODUCT SUPPORT:
        items          → list of OrderItem (one per product in the message)
        missing_entities → flattened list of what's missing across ALL items

    SINGLE-PRODUCT BACKWARD COMPAT:
        product_name, quantity_value, quantity_unit → from items[0] if exists
    """
    items:             List[OrderItem]  # All products extracted from message
    delivery_date:     str             # YYYY-MM-DD — always present, defaults to today+5
    invoice_number:    Optional[str]   # null for now, future use
    payment_reference: Optional[str]   # null for now, future use
    missing_entities:  List[str]       # flattened missing fields across all items
    raw_text:          str             # original customer message
    tenant_id:         str             # business isolation key

    # ── Single-product backward compat properties ─────────────────────────
    @property
    def product_name(self) -> Optional[str]:
        return self.items[0].product_name if self.items else None

    @property
    def quantity_value(self) -> Optional[int]:
        return self.items[0].quantity_value if self.items else None

    @property
    def quantity_unit(self) -> Optional[str]:
        return self.items[0].quantity_unit if self.items else None

    @property
    def quantity(self) -> Optional[str]:
        return self.items[0].quantity_str if self.items else None

    @property
    def all_complete(self) -> bool:
        """True if every item has both product and quantity."""
        return bool(self.items) and all(item.is_complete for item in self.items)

    @property
    def is_multi_product(self) -> bool:
        return len(self.items) > 1