"""
Daily Zendesk conversation pipeline.

Run once per day (e.g. via Windows Task Scheduler at 8:00 AM).
Each run:
  1. Pulls tickets CREATED THE PREVIOUS CALENDAR DAY from Zendesk
  2. Extracts every message (customer + bot/agent) per ticket
  3. Pairs each customer question with its bot/agent answer
  4. Categorizes + grades each pair with OpenAI
  5. Appends the scored rows to one growing master CSV
     (does not overwrite previous days' results)
"""

import csv
import json
import os
import re
import sys
import time
import unicodedata
import requests
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()  # reads variables from a local .env file (not committed to git)

LOCAL_TZ = ZoneInfo("America/Chicago")  # Wisconsin — Central Time (handles CDT/CST automatically)

# ── CONFIG ────────────────────────────────────────────────────────────────────
ZENDESK_SUBDOMAIN = os.environ["ZENDESK_SUBDOMAIN"]
ZENDESK_EMAIL     = os.environ["ZENDESK_EMAIL"]
ZENDESK_API_TOKEN = os.environ["ZENDESK_API_TOKEN"]
AUTH              = (ZENDESK_EMAIL + "/token", ZENDESK_API_TOKEN)

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

MASTER_CSV = sys.argv[1] if len(sys.argv) > 1 else "zendesk_conversations_scored_master.csv"

# BigQuery (optional — if GCP_SERVICE_ACCOUNT_JSON isn't set, BigQuery upload is skipped entirely)
GCP_SERVICE_ACCOUNT_JSON = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
BQ_TABLE_REF             = os.environ.get("BQ_TABLE_REF")  # e.g. "focus-on-energy.zendesk_scoring.daily_scored"

SITEMAP_URL        = "https://focusonenergy.com/sitemaps-2-sitemap.xml"
CRAWL_CACHE_FILE   = "website_content_cache.json"  # reused across runs — commit this to the repo
MAX_PAGES_TO_CRAWL = 5000
MAX_CHARS_PER_PAGE = 4000
TOP_N_PAGES        = 4

INSTRUCTIONS_CSV = "instructions_export.csv"  # Title, Instruction, Status columns

SUGGESTED_CATEGORIES = [
    "Rebates",
    "Trade Ally",
    "IRA / HEAR Program",
    "Online Home Assessment",
    "Business Incentives",
    "Appliance Recycling",
    "Energy Audit",
    "Solar",
    "HVAC",
    "Windows & Insulation",
    "Billing / Payment Assistance",
    "General Program Info",
]

SKIP_PHRASES = [
    "how would you rate your experience",
    "how was your experience",
    "talk to human",
]


# ── WEBSITE CRAWLING (for grounding the grading in real site content) ─────────

def load_crawl_cache():
    if os.path.exists(CRAWL_CACHE_FILE):
        print(f"💾 Found saved crawl: {CRAWL_CACHE_FILE}")
        try:
            with open(CRAWL_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"   Loaded {len(data)} cached pages — skipping crawl.\n")
            return data
        except Exception as e:
            print(f"   ⚠️ Could not load cache: {e} — will re-crawl.")
    return None


def save_crawl_cache(website_content):
    try:
        with open(CRAWL_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(website_content, f, ensure_ascii=False)
        print(f"💾 Crawl saved to '{CRAWL_CACHE_FILE}' — will be reused next run.\n")
    except Exception as e:
        print(f"   ⚠️ Could not save cache: {e}")


def fetch_sitemap_urls(sitemap_url):
    print(f"📡 Fetching sitemap: {sitemap_url}")
    headers = {"User-Agent": "Mozilla/5.0"}
    all_page_urls = []

    def fetch_and_parse(url, depth=0):
        try:
            resp = requests.get(url, timeout=15, headers=headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, features="xml")
            locs = [loc.text.strip() for loc in soup.find_all("loc")]
            for loc in locs:
                if loc.endswith(".xml"):
                    if depth < 3:
                        fetch_and_parse(loc, depth + 1)
                elif "focusonenergy.com" in loc and not loc.endswith(
                        (".pdf", ".jpg", ".png", ".gif", ".svg")):
                    all_page_urls.append(loc)
        except Exception as e:
            print(f"   ⚠️ Failed to fetch {url}: {e}")

    fetch_and_parse(sitemap_url)
    unique_urls = list(dict.fromkeys(all_page_urls))
    print(f"   Found {len(unique_urls)} real page URLs")
    return unique_urls


def scrape_page_text(url):
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "svg", "form"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:MAX_CHARS_PER_PAGE]
    except Exception:
        return ""


def crawl_website(sitemap_url):
    """Load from cache if available, otherwise crawl and save."""
    cached = load_crawl_cache()
    if cached is not None:
        return cached

    urls = fetch_sitemap_urls(sitemap_url)
    if not urls:
        print("⚠️  No URLs found — grading will proceed without website context.")
        return {}

    urls = urls[:MAX_PAGES_TO_CRAWL]
    print(f"🕷️  Crawling {len(urls)} pages (this only happens once)...")
    website_content = {}
    for i, url in enumerate(urls, 1):
        text = scrape_page_text(url)
        if text:
            website_content[url] = text
        if i % 50 == 0:
            print(f"   → {i}/{len(urls)} pages crawled...")
        time.sleep(0.3)

    print(f"✅ Crawled {len(website_content)} pages successfully")
    save_crawl_cache(website_content)
    return website_content


def find_relevant_pages(question, bot_answer, website_content, top_n=TOP_N_PAGES):
    if not website_content:
        return ""
    combined = (question + " " + bot_answer).lower()
    words     = set(re.findall(r"\b[a-z]{4,}\b", combined))
    stopwords = {"that", "this", "with", "have", "from", "they", "will",
                 "your", "what", "when", "where", "which", "there", "their",
                 "more", "also", "some", "been", "would", "could", "should"}
    keywords = words - stopwords
    scored = []
    for url, text in website_content.items():
        text_lower = text.lower()
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scored.append((score, url, text))
    scored.sort(reverse=True)
    top_pages = scored[:top_n]
    if not top_pages:
        return ""
    parts = []
    for score, url, text in top_pages:
        parts.append(f"--- PAGE: {url} ---\n{text[:MAX_CHARS_PER_PAGE]}")
    return "\n\n".join(parts)


# ── LOAD BOT INSTRUCTIONS (for identifying which ones applied to each answer) ─

def load_instructions(path=INSTRUCTIONS_CSV):
    """
    Loads the exported instruction set (Title, Instruction, Status columns).
    Only 'Active' instructions are used — inactive ones are excluded so the
    model isn't asked to match against rules that aren't actually live.
    """
    if not os.path.isfile(path):
        print(f"⚠️  Instructions file '{path}' not found — grading will proceed without it.")
        return []

    instructions = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if (row.get("Status", "").strip().lower() == "active"):
                title = row.get("Title", "").strip()
                text  = row.get("Instruction", "").strip()
                if title and text:
                    instructions.append({"title": title, "instruction": text})

    print(f"📜 Loaded {len(instructions)} active instructions from {path}\n")
    return instructions


def format_instructions_for_prompt(instructions):
    if not instructions:
        return ""
    lines = [f'- "{i["title"]}": {i["instruction"]}' for i in instructions]
    return "\n".join(lines)


# ── DATE RANGE (previous calendar day, or manual override via CLI args) ───────

def get_previous_day_range():
    """
    Normal behavior: returns (yesterday, today) so the pipeline pulls just
    the previous calendar day — this is what runs every scheduled day.

    Override for a one-time backfill: pass start and end dates as the 2nd
    and 3rd command-line arguments (YYYY-MM-DD), e.g.:
        python daily_zendesk_pipeline.py test_output.csv 2026-08-24 2026-08-27
    This pulls everything created in [start_date, end_date) — end_date is
    exclusive, so use the day AFTER the last day you want included.
    """
    if len(sys.argv) > 3:
        return sys.argv[2], sys.argv[3]
    today = datetime.now(LOCAL_TZ).date()
    yesterday = today - timedelta(days=1)
    return yesterday.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")


# ── ZENDESK: FETCH TICKETS CREATED IN DATE RANGE ──────────────────────────────

def fetch_tickets_for_date_range(start_date, end_date):
    tickets_all = []
    url = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2/search.json"
    query = f"type:ticket created>={start_date} created<{end_date}"
    params = {"query": query, "per_page": 100, "sort_by": "created_at", "sort_order": "asc"}

    while url:
        resp = requests.get(url, auth=AUTH, params=params if not tickets_all else None)
        resp.raise_for_status()
        data = resp.json()
        tickets_all.extend(data.get("results", []))
        url = data.get("next_page")

    return tickets_all


# ── ZENDESK: EXTRACT MESSAGES FROM A TICKET'S AUDITS ──────────────────────────

def parse_timestamp(ts_raw, source):
    """Normalize both chat (epoch ms) and comment (ISO8601) timestamps to UTC datetime."""
    try:
        if source == "chat":
            return datetime.fromtimestamp(float(ts_raw) / 1000, tz=timezone.utc)
        else:
            return datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def fetch_messages_for_ticket(ticket):
    ticket_id    = ticket["id"]
    requester_id = ticket.get("requester_id")
    status       = ticket.get("status")

    messages = []
    audits_resp = requests.get(
        f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2/tickets/{ticket_id}/audits.json",
        auth=AUTH
    )
    if audits_resp.status_code != 200:
        return messages

    audits = audits_resp.json().get("audits", [])

    for audit in audits:
        for event in audit.get("events", []):
            etype = event.get("type", "")

            if etype == "ChatStartedEvent":
                history = event.get("value", {}).get("history", [])
                for item in history:
                    if item.get("type") == "ChatMessage":
                        actor_type = item.get("actor_type", "unknown")
                        ts = parse_timestamp(item.get("timestamp"), "chat")
                        messages.append({
                            "ticket_id": ticket_id,
                            "status":    status,
                            "role":      "customer" if actor_type == "end-user" else actor_type,
                            "message":   item.get("message"),
                            "ts":        ts,
                        })

            elif etype == "Comment" and event.get("public"):
                body = event.get("plain_body") or event.get("body", "")
                if body:
                    author_id = audit.get("author_id")
                    ts = parse_timestamp(audit.get("created_at"), "comment")
                    messages.append({
                        "ticket_id": ticket_id,
                        "status":    status,
                        "role":      "customer" if author_id == requester_id else "agent_or_bot",
                        "message":   body,
                        "ts":        ts,
                    })

    return messages


# ── FILTERS ────────────────────────────────────────────────────────────────────

def is_skip_message(text):
    lower = text.lower()
    return any(phrase in lower for phrase in SKIP_PHRASES)


def is_emoji_only(text):
    clean = text.strip()
    if not clean:
        return True
    return not any(unicodedata.category(c).startswith("L") for c in clean)


# ── PAIR Q&A PER TICKET ────────────────────────────────────────────────────────

def pair_messages(all_messages):
    """Groups messages by ticket_id, sorts chronologically, pairs customer Q -> bot/agent A."""
    tickets = {}
    for m in all_messages:
        tickets.setdefault(m["ticket_id"], []).append(m)

    pairs = []
    for ticket_id, msgs in tickets.items():
        msgs = sorted(msgs, key=lambda m: m["ts"])

        current_question = None
        current_q_msg     = None
        current_answer_parts = []
        current_a_msgs    = []

        def flush():
            if current_question is not None:
                answer_text = " ".join(current_answer_parts).strip()
                q_local = current_q_msg["ts"].astimezone(LOCAL_TZ)
                a_local = current_a_msgs[-1]["ts"].astimezone(LOCAL_TZ) if current_a_msgs else None
                pairs.append({
                    "ticket_id":     ticket_id,
                    "status":        current_q_msg["status"],
                    "question_date": q_local.strftime("%Y-%m-%d"),
                    "question_time": q_local.strftime("%H:%M:%S"),
                    "answer_time":   a_local.strftime("%H:%M:%S") if a_local else "",
                    "question":      current_question,
                    "bot_answer_raw": answer_text,
                })

        for m in msgs:
            message = (m.get("message") or "").strip()
            role    = (m.get("role") or "").strip().lower()
            if not message or is_skip_message(message) or is_emoji_only(message):
                continue

            if role == "customer":
                flush()
                current_question = message
                current_q_msg = m
                current_answer_parts = []
                current_a_msgs = []
            else:
                if current_question is not None:
                    current_answer_parts.append(message)
                    current_a_msgs.append(m)

        flush()

    return pairs


# ── SOURCES EXTRACTION ─────────────────────────────────────────────────────────

def extract_sources(bot_answer):
    match = re.search(r'\bsources\b\s*:?', bot_answer, flags=re.IGNORECASE)
    if match:
        answer_part  = bot_answer[:match.start()].strip()
        sources_part = bot_answer[match.end():].strip()
        return answer_part, sources_part
    return bot_answer.strip(), ""


# ── OPENAI: CATEGORIZE ─────────────────────────────────────────────────────────

def categorize_question_with_openai(question):
    prompt = f"""QUESTION:
{question}

Here is a list of existing categories used for this energy efficiency program chatbot:
{", ".join(SUGGESTED_CATEGORIES)}

Classify the question above into the SINGLE most fitting category from that list.
If none of the categories fit well, invent a new short category name (2-4 words) that best describes the topic.
If the question is too vague, unrelated, or unclear to categorize, respond with exactly: Unknown

Respond with ONLY the category name — no explanation, no punctuation, no markdown."""

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a classifier that assigns a short topic category to customer questions for the Focus on Energy efficiency program chatbot."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 20,
                "temperature": 0
            },
            timeout=20
        )
        resp_json = response.json()
        if "choices" not in resp_json:
            print(f"    ⚠️ Categorization API error: {resp_json}")
            return "Unknown"
        category = resp_json["choices"][0]["message"]["content"].strip().strip('"\'`. ')
        return category if category else "Unknown"
    except Exception as e:
        print(f"    ⚠️ Categorization failed: {e}")
        return "Unknown"


# ── OPENAI: GRADE ───────────────────────────────────────────────────────────────

def grade_answer_with_openai(question, bot_answer, article_titles, website_context="", instructions_text="", max_retries=5):
    website_section = ""
    if website_context:
        website_section = f"""
RELEVANT CONTENT FROM THE FOCUS ON ENERGY WEBSITE:
(Use this as the source of truth when evaluating accuracy)
{website_context}
"""

    instructions_section = ""
    if instructions_text:
        instructions_section = f"""
ACTIVE BOT INSTRUCTIONS (rules the bot is supposed to follow):
Each instruction has a short title and its full text.
{instructions_text}
"""

    prompt = f"""QUESTION ASKED BY USER:
{question}

BOT ANSWER:
{bot_answer}

ARTICLES / LINKS SUGGESTED BY BOT:
{article_titles if article_titles else "None"}
{website_section}
{instructions_section}
Please grade the bot's answer on each of the following 5 criteria using a scale of 1-5:
1 = Very Poor, 2 = Poor, 3 = Average, 4 = Good, 5 = Excellent

CRITERIA:
1. ACCURACY & HELPFULNESS
   - Does the bot answer match the actual website content?
   - Does it give correct, factual answers based on the website?
   - Does it actually solve the user's problem?

2. RETRIEVAL QUALITY
   - Did it reference the right information from the program?
   - Did it hallucinate information not on the website?

3. CONVERSATIONAL UX
   - Is the response clear, natural and easy to understand?
   - Does it handle the question appropriately?

4. SAFETY & COMPLIANCE
   - Does it avoid harmful or incorrect outputs?
   - Does it stay on topic for an energy efficiency program?

5. BUSINESS OUTCOMES
   - Is the answer likely to reduce support tickets?
   - Does it guide the user toward a clear next step or solution?

Also identify which of the ACTIVE BOT INSTRUCTIONS (if any) were relevant to this
question and answer — meaning the instruction's subject matter applied here,
regardless of whether the bot actually followed it correctly. List their exact
titles only. If none apply, use an empty list.

In your notes, specifically call out:
- Whether the bot answer matches the website content
- Any factual errors or missing info compared to the website
- Whether the bot actually followed the relevant instruction(s), if any applied
- What was done well

Respond ONLY with a valid JSON object:
{{
  "accuracy_helpfulness": <1-5>,
  "retrieval_quality": <1-5>,
  "conversational_ux": <1-5>,
  "safety_compliance": <1-5>,
  "business_outcomes": <1-5>,
  "instructions_involved": ["<exact title>", "<exact title>"],
  "notes": "<3-4 sentence detailed explanation comparing bot answer to website content and relevant instructions>"
}}"""

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a Focus on Energy subject matter expert and chatbot QA evaluator. You have access to actual website content and must use it as ground truth when evaluating accuracy."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 600,
                "temperature": 0
            },
            timeout=30
        )
        resp_json = response.json()

        # Handle OpenAI rate limiting with retry + backoff instead of failing immediately
        if "error" in resp_json and resp_json["error"].get("code") == "rate_limit_exceeded" and max_retries > 0:
            msg = resp_json["error"].get("message", "")
            wait_match = re.search(r'try again in ([\d.]+)(ms|s)', msg)
            if wait_match:
                amount, unit = wait_match.groups()
                wait_sec = float(amount) / 1000 if unit == "ms" else float(amount)
            else:
                wait_sec = 5
            wait_sec = min(wait_sec + 1, 30)  # small buffer, capped so we never wait absurdly long
            print(f"    ⏳ Rate limited — waiting {wait_sec:.1f}s before retry ({max_retries} left)...")
            time.sleep(wait_sec)
            return grade_answer_with_openai(question, bot_answer, article_titles, website_context,
                                             instructions_text, max_retries=max_retries - 1)

        if "choices" not in resp_json:
            print(f"    ⚠️ Grading API error: {resp_json}")
            return {"accuracy_helpfulness": 0, "retrieval_quality": 0, "conversational_ux": 0,
                    "safety_compliance": 0, "business_outcomes": 0, "instructions_involved": [],
                    "notes": f"API error: {resp_json}"}
        content = resp_json["choices"][0]["message"]["content"].strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content.strip())
        if "instructions_involved" not in parsed or not isinstance(parsed["instructions_involved"], list):
            parsed["instructions_involved"] = []
        return parsed
    except Exception as e:
        print(f"    ⚠️ Grading failed: {e}")
        return {"accuracy_helpfulness": 0, "retrieval_quality": 0, "conversational_ux": 0,
                "safety_compliance": 0, "business_outcomes": 0, "instructions_involved": [],
                "notes": f"Grading error: {e}"}


def calculate_success_rate(grades):
    total = sum(grades.get(k, 0) for k in
                ["accuracy_helpfulness", "retrieval_quality", "conversational_ux",
                 "safety_compliance", "business_outcomes"])
    return round((total / 25) * 100, 1)


def detect_outlier(success_rate):
    return "YES" if success_rate < 50 else "NO"


# ── APPEND TO MASTER CSV ────────────────────────────────────────────────────────

FIELDNAMES = [
    "ticket_id", "status", "question_date", "question_time", "answer_time",
    "question", "category", "bot_answer", "article_titles",
    "accuracy_helpfulness", "retrieval_quality", "conversational_ux",
    "safety_compliance", "business_outcomes", "instructions_involved", "notes",
    "success_rate", "outlier", "responded", "graded_at"
]


def load_existing_keys():
    """
    Returns a set of (ticket_id, question) pairs already present in the master CSV,
    so re-running the pipeline never appends duplicate rows.
    """
    keys = set()
    if not os.path.isfile(MASTER_CSV):
        return keys
    with open(MASTER_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            keys.add((str(row.get("ticket_id", "")), row.get("question", "")))
    return keys


def append_results(results):
    file_exists = os.path.isfile(MASTER_CSV)
    with open(MASTER_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)


# ── UPLOAD NEW ROWS TO BIGQUERY ────────────────────────────────────────────────

BQ_SCHEMA = [
    bigquery.SchemaField("ticket_id", "INTEGER"),
    bigquery.SchemaField("status", "STRING"),
    bigquery.SchemaField("question_date", "STRING"),
    bigquery.SchemaField("question_time", "STRING"),
    bigquery.SchemaField("answer_time", "STRING"),
    bigquery.SchemaField("question", "STRING"),
    bigquery.SchemaField("category", "STRING"),
    bigquery.SchemaField("bot_answer", "STRING"),
    bigquery.SchemaField("article_titles", "STRING"),
    bigquery.SchemaField("accuracy_helpfulness", "INTEGER"),
    bigquery.SchemaField("retrieval_quality", "INTEGER"),
    bigquery.SchemaField("conversational_ux", "INTEGER"),
    bigquery.SchemaField("safety_compliance", "INTEGER"),
    bigquery.SchemaField("business_outcomes", "INTEGER"),
    bigquery.SchemaField("instructions_involved", "STRING"),
    bigquery.SchemaField("notes", "STRING"),
    bigquery.SchemaField("success_rate", "FLOAT"),
    bigquery.SchemaField("outlier", "STRING"),
    bigquery.SchemaField("responded", "STRING"),
    bigquery.SchemaField("graded_at", "STRING"),
]

BQ_INT_COLS   = ["ticket_id", "accuracy_helpfulness", "retrieval_quality",
                  "conversational_ux", "safety_compliance", "business_outcomes"]
BQ_FLOAT_COLS = ["success_rate"]


def upload_to_bigquery(results):
    """
    Appends this run's new rows to a BigQuery table (WRITE_APPEND — never
    truncates existing data). Skipped entirely if GCP credentials or the
    table reference aren't configured, so this stays optional.

    Uploads as JSON rows (load_table_from_json) rather than a pandas
    dataframe — this avoids a pyarrow/pandas version incompatibility that
    can cause "TypeError: expected bytes, NoneType found" with
    load_table_from_dataframe on some environments.
    """
    if not results:
        return
    if not GCP_SERVICE_ACCOUNT_JSON or not BQ_TABLE_REF:
        print("ℹ️  BigQuery not configured (GCP_SERVICE_ACCOUNT_JSON / BQ_TABLE_REF missing) — skipping upload.")
        return

    try:
        rows = []
        for r in results:
            row = {}
            for key, value in r.items():
                if key in BQ_INT_COLS:
                    row[key] = int(value) if str(value).lstrip("-").isdigit() else None
                elif key in BQ_FLOAT_COLS:
                    try:
                        row[key] = float(value) if value not in (None, "") else None
                    except (ValueError, TypeError):
                        row[key] = None
                else:
                    row[key] = value if value not in (None, "") else None
            rows.append(row)

        schema = [f for f in BQ_SCHEMA if f.name in rows[0].keys()]

        gcp_info = json.loads(GCP_SERVICE_ACCOUNT_JSON)
        credentials = service_account.Credentials.from_service_account_info(gcp_info)
        client = bigquery.Client(credentials=credentials, project=gcp_info["project_id"])

        job = client.load_table_from_json(
            rows,
            BQ_TABLE_REF,
            job_config=bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND",  # adds rows, never deletes/overwrites existing data
                schema=schema,
            )
        )
        job.result()
        print(f"☁️  Uploaded {len(rows)} rows to BigQuery: {BQ_TABLE_REF}")
    except Exception as e:
        print(f"⚠️  BigQuery upload failed: {e}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    start_date, end_date = get_previous_day_range()
    print(f"📅 Pulling conversations created {start_date} (previous day)\n")

    # Crawl website once — loads from cache on subsequent runs
    website_content = crawl_website(SITEMAP_URL)

    # Load active bot instructions once
    instructions = load_instructions()
    instructions_text = format_instructions_for_prompt(instructions)

    print("Fetching tickets from Zendesk...")
    tickets = fetch_tickets_for_date_range(start_date, end_date)
    print(f"  → {len(tickets)} tickets found\n")

    if not tickets:
        print("No tickets found for this date. Nothing to score.")
        return

    print("Extracting messages...")
    all_messages = []
    for i, ticket in enumerate(tickets, 1):
        msgs = fetch_messages_for_ticket(ticket)
        all_messages.extend(msgs)
        print(f"  → Ticket {i}/{len(tickets)}: #{ticket['id']} — {len(msgs)} messages")

    print(f"\n📋 Pairing questions and answers...")
    pairs = pair_messages(all_messages)
    print(f"  → {len(pairs)} Q&A pairs\n")

    existing_keys = load_existing_keys()
    original_count = len(pairs)
    pairs = [p for p in pairs if (str(p["ticket_id"]), p["question"]) not in existing_keys]
    skipped = original_count - len(pairs)
    if skipped:
        print(f"⏭️  Skipping {skipped} pair(s) already present in {MASTER_CSV}\n")

    if not pairs:
        print("Nothing new to score — all pairs already exist in the master CSV.")
        return

    results = []
    for i, pair in enumerate(pairs, 1):
        question = pair["question"]
        print(f"[{i}/{len(pairs)}] {question[:70]}...")

        bot_answer, article_titles = extract_sources(pair["bot_answer_raw"])
        if not article_titles:
            urls_in_text = re.findall(r'https?://[^ ]+', bot_answer)
            if urls_in_text:
                article_titles = "; ".join(urls_in_text)

        responded = "NO" if not bot_answer else "YES"

        website_context = find_relevant_pages(question, bot_answer, website_content)

        category = categorize_question_with_openai(question)
        grades       = grade_answer_with_openai(question, bot_answer, article_titles, website_context, instructions_text)
        success_rate = calculate_success_rate(grades)
        outlier      = detect_outlier(success_rate)

        instructions_involved = grades.get("instructions_involved", [])
        instructions_str = "; ".join(instructions_involved) if instructions_involved else "None"

        print(f"    🏷️ {category}  ⭐{success_rate}%  Outlier:{outlier}  Responded:{responded}")

        results.append({
            "ticket_id":            pair["ticket_id"],
            "status":               pair["status"],
            "question_date":        pair["question_date"],
            "question_time":        pair["question_time"],
            "answer_time":          pair["answer_time"],
            "question":             question,
            "category":             category,
            "bot_answer":           bot_answer,
            "article_titles":       article_titles if article_titles else "None",
            "accuracy_helpfulness": grades.get("accuracy_helpfulness", 0),
            "retrieval_quality":    grades.get("retrieval_quality", 0),
            "conversational_ux":    grades.get("conversational_ux", 0),
            "safety_compliance":    grades.get("safety_compliance", 0),
            "business_outcomes":    grades.get("business_outcomes", 0),
            "instructions_involved": instructions_str,
            "notes":                grades.get("notes", ""),
            "success_rate":         success_rate,
            "outlier":              outlier,
            "responded":            responded,
            "graded_at":            datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        })

    append_results(results)
    upload_to_bigquery(results)

    total       = len(results)
    responded   = sum(1 for r in results if r["responded"] == "YES")
    outliers    = sum(1 for r in results if r["outlier"] == "YES")
    avg_success = sum(r["success_rate"] for r in results) / total if total else 0
    print(f"\n{'='*60}")
    print(f"✅ Responded: {responded}/{total}   ❌ No response: {total-responded}/{total}")
    print(f"⚠️  Outliers: {outliers}/{total}")
    print(f"⭐ Average success rate: {avg_success:.1f}%")
    print(f"📄 Appended to: {MASTER_CSV}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()