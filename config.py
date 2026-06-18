# ─────────────────────────────────────────────────────────────────────────────
# config.py — Central Configuration Loader
#
# PURPOSE:
#   Single file that reads ALL environment variables from the .env file and
#   exposes them as typed Python constants.
#
# WHY THIS EXISTS:
#   Instead of calling os.getenv() scattered across every file, every module
#   imports from config.py. This means:
#   - One place to change a variable name
#   - One place to set default values
#   - Easy to mock in tests without touching .env
#
# USAGE:
#   from config import ACCESS_TOKEN, SUPABASE_URL
# ─────────────────────────────────────────────────────────────────────────────

from dotenv import load_dotenv
import os

# load_dotenv() reads the .env file and injects all variables into os.environ.
# Must be called BEFORE any os.getenv() calls.
load_dotenv()


# ── WhatsApp / Meta Cloud API ─────────────────────────────────────────────────

PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
# Meta's internal ID for your specific WhatsApp sender number.
# Used in every outbound message URL: graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages

WABA_ID = os.getenv("WABA_ID")
# WhatsApp Business Account ID — the parent account that owns the phone number.
# Required for WABA-level Meta API operations.

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
# Permanent Bearer token for Meta Graph API authentication.
# Used for: sending replies AND downloading media files (both require this).
# CRITICAL: Keep this secret — anyone with this token can send messages as your business.

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
# Secret string set in both .env and Meta Developer Portal.
# Meta sends it back during webhook setup — your server checks it matches.
# Used only once during initial webhook verification.


# ── Azure OpenAI ──────────────────────────────────────────────────────────────

AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT")
# Base URL of your Azure OpenAI resource.
# Format: https://{resource-name}.cognitiveservices.azure.com

AZURE_AI_API_KEY = os.getenv("AZURE_AI_API_KEY")
# Secret API key for authenticating with your Azure OpenAI resource.
# CRITICAL: Never log or expose this key.

AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
# The deployment name of your GPT-4.1 model inside Azure Portal.
# Default: "gpt-4.1" so the app works if this env var is not set.

AZURE_AI_API_VERSION = os.getenv("AZURE_AI_API_VERSION", "2024-12-01-preview")
# The Azure REST API version required in every API call URL.
# Default provided so the app doesn't break if the env var is missing.


# ── Supabase ──────────────────────────────────────────────────────────────────

SUPABASE_URL = os.getenv("SUPABASE_URL")
# Your Supabase project URL.
# Found at: Supabase Dashboard → Settings → API → Project URL
# Format: https://your-project-id.supabase.co
# Used to initialise the Supabase client in db/session_store.py and adapter.

SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
# The service_role secret key — has full DB and Storage access.
# Found at: Supabase Dashboard → Settings → API → service_role (secret)
# CRITICAL: Never expose this in frontend code or public repos.
# Use this on server-side only — it bypasses Row Level Security.

SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "whatsapp-media")
# The name of the Supabase Storage bucket where images and audio are stored.
# Created manually in: Supabase Dashboard → Storage → New bucket
# Default: "whatsapp-media"
# Files are stored at: {SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}


# ── Colleague's Products API ──────────────────────────────────────────────────

PRODUCTS_API_URL = os.getenv(
    "PRODUCTS_API_URL",
    "https://inventaa-products-api.vercel.app/products/batch"
)
# Colleague's product catalog API.
# Endpoint: POST /products/batch
# Body:     {"skus": ["24C-2055", "12M-2014B"]}
# Returns:  product name, price, image_url, specs etc.
#
# HOW TO CHANGE THE URL:
#   Option 1 — set in .env file:  PRODUCTS_API_URL=https://new-url.com/products/batch
#   Option 2 — change the default value above
#
# STATUS: API works but our server IP needs to be whitelisted by colleague.
#         Once whitelisted, SKU-based lookups work automatically.


# ── Colleague's GraphRAG Agent API ───────────────────────────────────────────

GRAPHRAG_API_URL = os.getenv(
    "GRAPHRAG_API_URL",
    "https://inventaa-graphrag.vercel.app/route"
)
# Colleague's Hybrid RAG Agent API — FastAPI + LangChain + Neo4j Graph.
# Endpoint: POST /route
# Body:     Full message row (same schema as messages table)
# Behaviour:
#   intent = "FAQ_KNOWLEDGE" → routes to Neo4j semantic search
#                              searches entire product catalog by meaning
#                              "gate lights" finds all gate light variants
#   intent = anything else   → bypasses search, returns empty / forwards
#
# HOW TO CHANGE THE URL:
#   Set GRAPHRAG_API_URL in .env file


# ── Application ───────────────────────────────────────────────────────────────

APP_NAME = os.getenv("APP_NAME", "Order Tracking AI")
# Display name of the application — used in server titles and logs.

BUSINESS_NAME = os.getenv("BUSINESS_NAME", "Order Tracking AI")
# Injected into the AI system prompt so the LLM knows which business it serves.