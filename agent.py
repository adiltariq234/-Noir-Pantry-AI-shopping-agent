"""
agent.py — Shopping assistant agent (hardened).

Fixes over the original:
  * user_id is injected via a closure (make_checkout_tool) — the LLM can NEVER
    supply/spoof it.
  * Rating / price / organic filtering is enforced in Python, not left to the LLM.
  * Off-topic images are rejected again (is_store_product guard restored).
  * orders table is auto-migrated to include user_id.
  * Upload allow-list + size cap + path-traversal guard.
  * Ratings with no reviews return None (not a misleading 0.0).
"""

import base64
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_mistralai import ChatMistralAI


load_dotenv()

# ---------------------------------------------------------------------------
# Config / logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).resolve().parent
DB_PATH    = BASE_DIR / "store.db"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_BYTES   = 5 * 1024 * 1024          # 5 MB

_api_key = os.environ.get("MISTRAL_API_KEY")
if not _api_key:
    raise EnvironmentError("MISTRAL_API_KEY is not set. Use a secrets manager in production.")

llm = ChatMistralAI(model="mistral-small-latest", temperature=0)   # one model handles text + vision


# ---------------------------------------------------------------------------
# One-time DB migration (adds orders.user_id if missing)
# ---------------------------------------------------------------------------
def ensure_schema() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                product_name TEXT,
                price REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cols = {r[1] for r in cur.execute("PRAGMA table_info(orders)")}
        if "user_id" not in cols:
            cur.execute("ALTER TABLE orders ADD COLUMN user_id TEXT")
            logger.info("Migrated orders table: added user_id column.")
        conn.commit()
    finally:
        conn.close()

ensure_schema()


# ---------------------------------------------------------------------------
# Reviews data access (kept here so only agent.py + app.py are required)
# ---------------------------------------------------------------------------
def get_product_rating(product_id: int) -> dict:
    """Return average rating and review count for a single product."""
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT AVG(rating), COUNT(*) FROM reviews WHERE product_id = ?",
            (product_id,),
        ).fetchone()
    finally:
        conn.close()

    average = round(row[0], 2) if row and row[0] is not None else None
    count = int(row[1]) if row else 0
    return {
        "product_id": product_id,
        "average_rating": average,
        "review_count": count,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_upload_path(image_path: str) -> Path:
    resolved = (UPLOAD_DIR / image_path).resolve()
    if not resolved.is_relative_to(UPLOAD_DIR):
        raise ValueError(f"Access denied: '{image_path}' is outside the upload directory.")
    if resolved.suffix.lower() not in ALLOWED_IMAGE_EXT:
        raise ValueError(f"Unsupported file type: '{resolved.suffix}'.")
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: '{image_path}'")
    if resolved.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("Image is too large (max 5 MB).")
    return resolved


def _parse_vision_response(raw: str) -> dict:
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Vision LLM returned invalid JSON: {exc}") from exc
    required = {"is_store_product", "product_type", "search_query", "is_organic", "description"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Vision response missing fields: {missing}")
    return data


# ---------------------------------------------------------------------------
# Tools (stateless ones are module-level; checkout is built per-session)
# ---------------------------------------------------------------------------
@tool
def search_products(
    query: str,
    max_price: Optional[float] = None,
    is_organic: Optional[bool] = None,
    min_rating: Optional[float] = None,
) -> str:
    """
    Search products by keyword (name/description/category). Optionally filter by
    max_price, organic status, and minimum average rating. Rating/price/organic
    filtering is enforced here in code — results already satisfy every filter.
    Returns a JSON array of {id, name, category, price, description, is_organic,
    average_rating, review_count}.
    """
    if len(query) > 200:
        return json.dumps({"error": "Search query too long."})
    if max_price is not None and (max_price < 0 or max_price > 1_000_000):
        return json.dumps({"error": "max_price out of valid range."})

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        sql = "SELECT id, name, category, price, description, is_organic FROM products WHERE 1=1"
        params: list = []
        if query:
            sql += " AND (name LIKE ? OR description LIKE ? OR category LIKE ?)"
            like = f"%{query}%"
            params += [like, like, like]
        if max_price is not None:
            sql += " AND price <= ?"
            params.append(max_price)
        if is_organic is not None:
            sql += " AND is_organic = ?"
            params.append(1 if is_organic else 0)
        rows = cur.execute(sql, params).fetchall()
    except sqlite3.Error as exc:
        logger.error("search_products DB error: %s", exc)
        return json.dumps({"error": "Database error. Please try again."})
    finally:
        conn.close()

    products = []
    for r in rows:
        rating = get_product_rating(r[0])
        avg = rating.get("average_rating")
        # Enforce min_rating in code (products with no reviews are excluded when a
        # minimum rating is requested).
        if min_rating is not None:
            if avg is None or avg < min_rating:
                continue
        products.append({
            "id": r[0], "name": r[1], "category": r[2], "price": r[3],
            "description": r[4], "is_organic": bool(r[5]),
            "average_rating": avg, "review_count": rating.get("review_count", 0),
        })
    return json.dumps(products)


@tool
def get_rating(product_id: int) -> str:
    """Get average rating and review count for a product id."""
    if not isinstance(product_id, int) or product_id <= 0:
        return json.dumps({"error": "Invalid product_id."})
    try:
        return json.dumps(get_product_rating(product_id))
    except Exception as exc:
        logger.error("get_rating error for %s: %s", product_id, exc)
        return json.dumps({"error": "Could not fetch rating. Please try again."})


@tool
def describe_product_image(image_path: str) -> str:
    """
    Analyse a product image (filename inside uploads/) and return JSON attributes:
    is_store_product, product_type, search_query, is_organic, description.
    """
    try:
        safe_path = _safe_upload_path(image_path)
    except (ValueError, FileNotFoundError) as exc:
        logger.warning("describe_product_image rejected: %s", exc)
        return json.dumps({"error": str(exc)})

    data = base64.b64encode(safe_path.read_bytes()).decode()
    ext  = safe_path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    message = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
        {"type": "text", "text": (
            "Our store ONLY sells grocery/pantry products in these categories: honey, oil, "
            "nuts, seeds, grains, tea, coffee, snacks, dairy-alternatives (almond/oat/coconut/"
            "soy milk).\n\nReturn ONLY a JSON object with:\n"
            "- is_store_product: true only if the image clearly shows a product in a category "
            "above, else false (animals, people, electronics, clothing, etc.)\n"
            "- product_type: e.g. honey, olive oil, almonds — or null if not a store product\n"
            "- search_query: short keyword e.g. 'honey' — or null if not a store product\n"
            "- is_organic: true/false/null\n"
            "- description: one sentence about what you actually see\n"
            "JSON only, no markdown."
        )},
    ])
    try:
        resp = llm.invoke([message])
        return json.dumps(_parse_vision_response(resp.content))
    except Exception as exc:
        logger.error("describe_product_image vision error: %s", exc)
        return json.dumps({"error": "Could not analyse the image. Please try again."})


def make_checkout_tool(user_id: str):
    """Build a checkout tool bound to an authenticated user_id (closure — the LLM
    never sees or supplies user_id, so it cannot be spoofed)."""
    @tool
    def checkout(product_id: int) -> str:
        """Place an order for the given product ID for the current session user."""
        if not isinstance(product_id, int) or product_id <= 0:
            return "Error: invalid product ID."
        conn = sqlite3.connect(DB_PATH)
        try:
            cur = conn.cursor()
            row = cur.execute("SELECT name, price FROM products WHERE id = ?", (product_id,)).fetchone()
            if not row:
                return "Sorry, that product is no longer available."
            name, price = row
            cur.execute(
                "INSERT INTO orders (product_id, product_name, price, user_id) VALUES (?,?,?,?)",
                (product_id, name, price, user_id),
            )
            order_id = cur.lastrowid
            conn.commit()
        except sqlite3.Error as exc:
            logger.error("checkout DB error (product=%s user=%s): %s", product_id, user_id, exc)
            return "Sorry, your order could not be placed. Please try again."
        finally:
            conn.close()
        return (f"Order #{order_id} confirmed! '{name}' ordered for ${price:.2f}. "
                f"Arriving in 3-5 business days. Thank you!")
    return checkout


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful shopping assistant. Follow these rules strictly.\n\n"
    "IMAGE SEARCH — when the user provides an image filename:\n"
    "1. Call describe_product_image with the filename.\n"
    "2. If is_store_product is false, tell the user plainly the image doesn't match anything we "
    "sell (we carry honey, oil, nuts, seeds, grains, tea, coffee, snacks, dairy-alternatives) and "
    "mention what the image shows using the description. Do NOT call search_products.\n"
    "3. If true, call search_products with the returned search_query and is_organic, then follow "
    "the BROWSING flow from step 2.\n\n"
    "BROWSING — when the user describes what they want:\n"
    "1. Call search_products with any price/organic/min_rating filters the user gave. Rating and "
    "price filters are enforced by the tool, so trust the returned list.\n"
    "2. If the list is empty, say so plainly and suggest different keywords. Never invent products.\n"
    "3. Present products as a numbered list, plain text (no backticks/bold):\n\n"
    "   #<n>. <name> (ID:<product_id>) - $<price> *<rating> - <organic or non-organic>\n\n"
    "   Add a blank line between entries. Always include (ID:X).\n"
    "4. If one product qualifies, show it and ask 'Would you like to order it? Say yes or the number.'\n"
    "5. Do NOT call checkout at this stage.\n\n"
    "ORDERING — only when the user explicitly confirms (yes / order #2 / the first one):\n"
    "1. Find the (ID:X) from your previous message.\n"
    "2. Call checkout with that product_id.\n"
    "3. Confirm in plain text.\n\n"
    "Never order without explicit confirmation. Never guess a product_id."
)


def build_agent(user_id: str):
    """Create a per-session agent whose checkout is bound to user_id."""
    if not user_id or not isinstance(user_id, str) or len(user_id) > 128:
        raise ValueError("A valid authenticated user_id is required.")
    return create_agent(
        tools=[search_products, get_rating, describe_product_image, make_checkout_tool(user_id)],
        model=llm,
        system_prompt=SYSTEM_PROMPT,
    )


if __name__ == "__main__":
    agent = build_agent("demo-user-001")
    result = agent.invoke({"messages": [
        {"role": "user", "content": "I want organic honey with 4.5+ rating and less than $20."}
    ]})
    print(result["messages"][-1].content)
