# Noir Pantry — Multimodal AI Shopping Agent

A conversational AI shopping assistant for a grocery/pantry storefront. Customers can search
for products in plain language, upload a photo to find matching items, and check out —
all through a single chat interface backed by a tool-using LLM agent.

**Live demo:** https://noir-pantry.streamlit.app/

---

## Features

- **Conversational product search** — ask for a product by name, set a max price, a minimum
  rating, or an organic-only filter, in plain English.
- **Photo-based search** — upload a product image; a vision-capable LLM identifies the product
  category and description, then searches the catalogue for matching items.
- **Rating-aware results** — every product is returned with its live average rating and review
  count, pulled from the reviews table.
- **Confirmation-gated checkout** — the agent never places an order on its own; it always asks
  for explicit confirmation first.
- **Admin order view** — a password-protected sidebar panel for browsing recent orders.

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangChain (`create_agent`) |
| LLM | Mistral (`mistral-small-latest`) — one model handles both conversational reasoning and image understanding |
| Frontend | Streamlit |
| Database | SQLite (products, reviews, orders) |

## How It Works

The agent has four tools:

1. **`search_products`** — queries SQLite by keyword, with optional `max_price`, `is_organic`,
   and `min_rating` filters. Filtering is enforced in application code, not left to the model.
2. **`get_rating`** — returns the average rating and review count for a product.
3. **`describe_product_image`** — sends an uploaded image to the LLM's vision endpoint, checks
   it against the store's actual categories (honey, oil, nuts, seeds, grains, tea, coffee,
   snacks, dairy-alternatives), and returns a structured description used to drive a follow-up
   search.
4. **`checkout`** — places an order for a given product ID, only ever called after the user
   explicitly confirms.

A system prompt governs the conversational flow: image uploads go through the vision tool
first, text queries go straight to search, and no order is placed without an explicit "yes."

## Security Notes

This project was deliberately hardened past a naive first pass:

- **No user-identity spoofing** — each session's `user_id` is bound into the `checkout` tool via
  a Python closure when the agent is built, so the LLM never sees or supplies it and cannot be
  prompted into placing an order as another user.
- **Business rules enforced in code, not by the model** — price, rating, and organic filters are
  applied after the SQL query runs, so a persuasive prompt can't talk the agent into ignoring
  them.
- **Upload safety** — images are restricted to `.jpg` / `.jpeg` / `.png` / `.webp`, capped at
  5 MB, and resolved paths are checked against the upload directory to block path traversal.
- **Safe schema migration** — the `orders` table is auto-migrated to add a `user_id` column if
  it's missing, without touching existing data.
- **Constant-time admin auth** — the admin panel compares the password with `hmac.compare_digest`
  to avoid timing attacks.

## Running Locally

```bash
pip install -r requirements.txt
```

Create a `.env` file with:

```
MISTRAL_API_KEY=your_key_here
ADMIN_PASSWORD=optional_admin_password
```

Then run:

```bash
streamlit run app.py
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
