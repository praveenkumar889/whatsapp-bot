# utils/invoice.py — Invoice PDF Generator
#
# PURPOSE:
#   Generates a professional invoice PDF for confirmed orders.
#   Uploads to Supabase Storage → returns public URL.
#
# ZERO HARDCODING:
#   Every business detail (name, tagline, city, email, website, UPI)
#   comes from the tenants table via function parameters.
#   Works for ANY client — Inventaa, or any future tenant.
#
# FONTS: Built-in ReportLab Helvetica — no external font files needed.
#        Works identically on Windows, Linux, macOS, any server.
#        Uses "Rs." instead of ₹ symbol — no Unicode font dependency.

import io
from datetime import datetime, timezone, timedelta
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

from supabase import create_client, Client  # type: ignore[import]
from config import SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_STORAGE_BUCKET

_supabase: Optional[Client] = None

def _get_client() -> Client:
    global _supabase
    if _supabase is None:
        _supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase


# ── Color palette ─────────────────────────────────────────────────────────────
DARK_BLUE  = colors.HexColor("#1a1a2e")
MID_GRAY   = colors.HexColor("#555555")
LIGHT_GRAY = colors.HexColor("#f4f4f4")
BORDER     = colors.HexColor("#dddddd")
WHITE      = colors.white

# ── Currency prefix ───────────────────────────────────────────────────────────
# Using "Rs." instead of ₹ symbol — works on all fonts, all OS, all PDF viewers.
R = "Rs."


def generate_invoice_pdf(
    order:         dict,
    biz_name:      str,
    tagline:       Optional[str] = None,
    city:          Optional[str] = None,
    support_email: Optional[str] = None,
    website:       Optional[str] = None,
    upi_id:        Optional[str] = None,
    account_name:  Optional[str] = None,
    sender_name:   Optional[str] = None,
    sender_phone:  Optional[str] = None,
) -> bytes:
    """
    Generates invoice PDF in memory and returns raw bytes.

    Uses built-in Helvetica font — zero external font dependencies.
    All business details come from parameters (tenants table).
    """
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(
        buffer,
        pagesize     = A4,
        rightMargin  = 1.8 * cm,
        leftMargin   = 1.8 * cm,
        topMargin    = 1.5 * cm,
        bottomMargin = 1.5 * cm,
    )
    PAGE_W   = A4[0] - 3.6 * cm
    elements = []

    # ── Styles — all using built-in Helvetica ─────────────────────────────────
    def style(name, font="Helvetica", size=9, color=DARK_BLUE, align=TA_LEFT,
              bold=False, leading=None, before=0, after=0):
        fn = "Helvetica-Bold" if bold else font
        ld = leading or (size * 1.35)
        return ParagraphStyle(name, fontName=fn, fontSize=size, leading=ld,
                              textColor=color, alignment=align,
                              spaceBefore=before, spaceAfter=after)

    s_company = style("co",  bold=True, size=20, align=TA_CENTER, leading=26)
    s_sub     = style("sb",  size=8,    color=MID_GRAY, align=TA_CENTER, leading=13)
    s_section = style("sec", bold=True, size=13, before=4, after=6)
    s_normal  = style("nm",  size=9,    after=3)
    s_footer  = style("ft",  size=8,    color=MID_GRAY, align=TA_CENTER, leading=13)

    # ── Header — all from DB, nothing hardcoded ───────────────────────────────
    elements.append(Spacer(1, 0.2 * cm))
    elements.append(Paragraph(biz_name.upper(), s_company))
    elements.append(Spacer(1, 0.15 * cm))

    if tagline:
        elements.append(Paragraph(tagline, s_sub))
        elements.append(Spacer(1, 0.08 * cm))

    if city:
        elements.append(Paragraph(city, s_sub))

    contact_parts = []
    if support_email:
        contact_parts.append(support_email)
    if website:
        contact_parts.append(website)
    if contact_parts:
        elements.append(Paragraph("  |  ".join(contact_parts), s_sub))

    elements.append(Spacer(1, 0.35 * cm))
    elements.append(HRFlowable(width="100%", thickness=2, color=DARK_BLUE))
    elements.append(Spacer(1, 0.3 * cm))

    # ── Invoice title ─────────────────────────────────────────────────────────
    elements.append(Paragraph("TAX INVOICE", s_section))

    # ── Meta info table ───────────────────────────────────────────────────────
    ist          = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    invoice_date = ist.strftime("%d %B %Y")

    c1 = PAGE_W * 0.20
    c2 = PAGE_W * 0.35
    c3 = PAGE_W * 0.20
    c4 = PAGE_W * 0.25

    def lbl(t):
        return Paragraph(
            f"<font name='Helvetica-Bold' color='#555555'>{t}</font>", s_normal)
    def val(t):
        return Paragraph(
            f"<font name='Helvetica' color='#1a1a2e'>{t}</font>", s_normal)

    # Use sender_name/sender_phone overrides if provided (from incoming object).
    # This fixes the N/A bug when order was created by GraphRAG API with missing fields.
    customer_name  = sender_name  or order.get("sender_name")  or "N/A"
    customer_phone = sender_phone or order.get("session_id")   or "N/A"

    meta = Table([
        [lbl("Order ID:"),  val(order.get("order_id","N/A")),
         lbl("Invoice Date:"), val(invoice_date)],
        [lbl("Customer:"),  val(customer_name),
         lbl("Status:"),    val(order.get("status","CONFIRMED"))],
        [lbl("Phone:"),     val(customer_phone),
         lbl(""),           val("")],
    ], colWidths=[c1, c2, c3, c4])
    meta.setStyle(TableStyle([
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("VALIGN",        (0,0),(-1,-1), "TOP"),
    ]))
    elements.append(meta)
    elements.append(Spacer(1, 0.4 * cm))

    # ── Items table — supports 1 or many line items ───────────────────────────
    total_price    = float(order.get("total_price", 0))
    total_with_gst = float(order.get("total_with_gst", 0))
    gst_amount     = round(total_with_gst - total_price, 2)

    # Use order_items if available (multi-product), fall back to single item from header
    order_items = order.get("items") or []
    if not order_items and order.get("product_name"):
        qty_val = order.get("quantity_value", 0)
        qty_unit = order.get("quantity_unit") or "units"
        order_items = [{
            "product_name":   order.get("product_name"),
            "quantity_value": qty_val,
            "quantity_unit":  qty_unit,
            "unit_price":     float(order.get("unit_price") or 0),
            "total_price":    total_price,
        }]

    def hdr(t):
        return Paragraph(
            f"<font name='Helvetica-Bold'>{t}</font>",
            ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9,
                           textColor=WHITE, alignment=TA_CENTER, leading=13))

    def cel(t, align=TA_LEFT):
        return Paragraph(
            f"<font name='Helvetica'>{t}</font>",
            ParagraphStyle("c", fontName="Helvetica", fontSize=9,
                           textColor=DARK_BLUE, alignment=align, leading=13))

    cno  = PAGE_W * 0.07
    cprd = PAGE_W * 0.43
    cqty = PAGE_W * 0.13
    cunt = PAGE_W * 0.18
    ctot = PAGE_W * 0.19

    rows = [[hdr("#"), hdr("Product"), hdr("Qty"), hdr("Unit Price"), hdr("Total")]]
    alt  = [LIGHT_GRAY, WHITE]
    for idx, item in enumerate(order_items, 1):
        iv  = item.get("quantity_value", 0)
        iu  = item.get("quantity_unit") or "units"
        ip  = float(item.get("unit_price", 0))
        it  = float(item.get("total_price", 0))
        iqs = f"{iv} {iu}" if iu else str(iv)
        rows.append([
            cel(str(idx), TA_CENTER),
            cel(item.get("product_name", "N/A")),
            cel(iqs, TA_CENTER),
            cel(f"{R}{ip:,.2f}", TA_RIGHT),
            cel(f"{R}{it:,.2f}", TA_RIGHT),
        ])

    items_tbl = Table(rows, colWidths=[cno, cprd, cqty, cunt, ctot])
    items_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  DARK_BLUE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), alt),
        ("GRID",          (0,0),(-1,-1), 0.4, BORDER),
        ("TOPPADDING",    (0,0),(-1,-1), 7),
        ("BOTTOMPADDING", (0,0),(-1,-1), 7),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    elements.append(items_tbl)
    elements.append(Spacer(1, 0.2 * cm))

    # ── Totals ────────────────────────────────────────────────────────────────
    GREEN = colors.HexColor("#1a7a40")

    def trow(label, amount, bold=False, color=None, prefix=""):
        fn = "Helvetica-Bold" if bold else "Helvetica"
        fs = 10 if bold else 9
        tc = color or DARK_BLUE
        sl = ParagraphStyle("tl", fontName=fn, fontSize=fs,
                            textColor=tc, alignment=TA_RIGHT, leading=fs*1.4)
        sv = ParagraphStyle("tv", fontName=fn, fontSize=fs,
                            textColor=tc, alignment=TA_RIGHT, leading=fs*1.4)
        amt_str = f"{prefix}{R}{abs(amount):,.2f}" if prefix else f"{R}{amount:,.2f}"
        return ["", "", "",
                Paragraph(label, sl),
                Paragraph(amt_str, sv)]

    # Pull optional discount fields (populated for negotiated / discounted orders)
    original_amt    = float(order.get("original_amount") or 0)
    store_disc_pct  = order.get("store_discount_pct", 0)
    store_disc_amt  = float(order.get("store_discount_amount") or 0)
    neg_disc_amt    = float(order.get("negotiation_discount_amount") or 0)

    total_rows = []
    if original_amt and original_amt > total_price + 0.01:
        total_rows.append(trow("Original Amount:", original_amt))
    if store_disc_amt > 0:
        label = f"Store Offer ({store_disc_pct}% OFF):" if store_disc_pct else "Store Offer:"
        total_rows.append(trow(label, store_disc_amt, color=GREEN, prefix="-"))
    if neg_disc_amt > 0:
        total_rows.append(trow("Negotiation Discount:", neg_disc_amt, color=GREEN, prefix="-"))
    total_rows += [
        trow("Subtotal:",      total_price),
        trow("GST (18%):",     gst_amount),
        trow("TOTAL PAYABLE:", total_with_gst, bold=True),
    ]
    _total_saved = round(store_disc_amt + neg_disc_amt, 2)
    if _total_saved > 0:
        total_rows.append(trow("You Saved:", _total_saved, color=GREEN))

    # Line positions depend on how many rows were added above
    _subtotal_idx = len(total_rows) - (4 if _total_saved > 0 else 3)
    _total_idx    = len(total_rows) - (2 if _total_saved > 0 else 1)
    totals = Table(total_rows, colWidths=[cno, cprd, cqty, cunt, ctot])
    totals.setStyle(TableStyle([
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("RIGHTPADDING",  (3,0),(-1,-1), 8),
        ("LINEABOVE",     (3,_subtotal_idx),(-1,_subtotal_idx), 0.5, BORDER),
        ("LINEABOVE",     (3,_total_idx),(-1,_total_idx),   1.2, DARK_BLUE),
        ("LINEBELOW",     (3,_total_idx),(-1,_total_idx),   1.2, DARK_BLUE),
    ]))
    elements.append(totals)
    elements.append(Spacer(1, 0.6 * cm))

    # ── Payment section — from DB, nothing hardcoded ──────────────────────────
    elements.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    elements.append(Spacer(1, 0.2 * cm))

    payment_parts = []
    if upi_id:
        payment_parts.append(f"UPI ID: {upi_id}")
    if account_name:
        payment_parts.append(f"Account: {account_name}")

    if payment_parts:
        elements.append(Paragraph(
            "<font name='Helvetica-Bold'>Payment Details:</font>",
            ParagraphStyle("pb", fontName="Helvetica-Bold", fontSize=9,
                           textColor=DARK_BLUE)
        ))
        elements.append(Paragraph(
            "  |  ".join(payment_parts),
            ParagraphStyle("pd", fontName="Helvetica", fontSize=9,
                           textColor=MID_GRAY, leading=14)
        ))

    elements.append(Spacer(1, 0.5 * cm))

    # ── Footer ────────────────────────────────────────────────────────────────
    elements.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    elements.append(Spacer(1, 0.2 * cm))
    elements.append(Paragraph(
        "Thank you for your order!  |  This is a computer-generated invoice.",
        s_footer,
    ))
    if website:
        elements.append(Paragraph(website, s_footer))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


async def upload_invoice_to_storage(
    pdf_bytes: bytes,
    order_id:  str,
    tenant_id: str,
) -> Optional[str]:
    """Uploads invoice PDF to Supabase Storage, returns public URL."""
    try:
        safe_id = order_id.replace("#", "_")
        path    = f"invoices/{tenant_id}/{safe_id}.pdf"

        public_url = (
            f"{SUPABASE_URL}/storage/v1/object/public"
            f"/{SUPABASE_STORAGE_BUCKET}/{path}"
        )

        try:
            _get_client().storage.from_(SUPABASE_STORAGE_BUCKET).upload(
                path         = path,
                file         = pdf_bytes,
                file_options = {"content-type": "application/pdf", "upsert": "true"},
            )
            print(f"[INVOICE] Uploaded -> {public_url}")
        except Exception as upload_err:
            err_msg = str(upload_err)
            if "Duplicate" in err_msg or "already exists" in err_msg or "409" in err_msg:
                print(f"[INVOICE] Already exists in storage -> {public_url}")
            else:
                raise upload_err

        return public_url

    except Exception as e:
        print(f"[INVOICE] Upload failed: {e}")
        return None


async def generate_and_upload_invoice(
    order:         dict,
    biz_name:      str           = "",
    tagline:       Optional[str] = None,
    city:          Optional[str] = None,
    support_email: Optional[str] = None,
    website:       Optional[str] = None,
    upi_id:        Optional[str] = None,
    account_name:  Optional[str] = None,
    sender_name:   Optional[str] = None,
    sender_phone:  Optional[str] = None,
) -> Optional[str]:
    """
    Full flow: generate PDF -> upload to Supabase -> return URL.
    All business details come from parameters (populated from tenants table).
    sender_name and sender_phone override any N/A stored in the orders table.
    """
    try:
        pdf_bytes = generate_invoice_pdf(
            order         = order,
            biz_name      = biz_name or "Order Tracking AI",
            tagline       = tagline,
            city          = city,
            support_email = support_email,
            website       = website,
            upi_id        = upi_id,
            account_name  = account_name,
            sender_name   = sender_name,
            sender_phone  = sender_phone,
        )
        return await upload_invoice_to_storage(
            pdf_bytes = pdf_bytes,
            order_id  = order["order_id"],
            tenant_id = order["tenant_id"],
        )
    except Exception as e:
        print(f"[INVOICE] generate_and_upload failed: {e}")
        return None