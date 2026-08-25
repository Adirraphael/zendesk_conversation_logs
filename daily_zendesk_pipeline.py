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
import unicodedata
import requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()  # reads variables from a local .env file (not committed to git)

LOCAL_TZ = ZoneInfo("America/Chicago")  # Wisconsin — Central Time (handles CDT/CST automatically)

# ── CONFIG ────────────────────────────────────────────────────────────────────
ZENDESK_SUBDOMAIN = os.environ["ZENDESK_SUBDOMAIN"]
ZENDESK_EMAIL     = os.environ["ZENDESK_EMAIL"]
ZENDESK_API_TOKEN = os.environ["ZENDESK_API_TOKEN"]
AUTH              = (ZENDESK_EMAIL + "/token", ZENDESK_API_TOKEN)

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

MASTER_CSV = "zendesk_conversations_scored_master.csv"

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


# ── DATE RANGE (previous calendar day) ────────────────────────────────────────

def get_previous_day_range():
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

def grade_answer_with_openai(question, bot_answer, article_titles):
    prompt = f"""QUESTION ASKED BY USER:
{question}

BOT ANSWER:
{bot_answer}

ARTICLES / LINKS SUGGESTED BY BOT:
{article_titles if article_titles else "None"}

Please grade the bot's answer on each of the following 5 criteria using a scale of 1-5:
1 = Very Poor, 2 = Poor, 3 = Average, 4 = Good, 5 = Excellent

CRITERIA:
1. ACCURACY & HELPFULNESS
2. RETRIEVAL QUALITY
3. CONVERSATIONAL UX
4. SAFETY & COMPLIANCE
5. BUSINESS OUTCOMES

Respond ONLY with a valid JSON object:
{{
  "accuracy_helpfulness": <1-5>,
  "retrieval_quality": <1-5>,
  "conversational_ux": <1-5>,
  "safety_compliance": <1-5>,
  "business_outcomes": <1-5>,
  "notes": "<2-3 sentence explanation of the scores>"
}}"""

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a Focus on Energy subject matter expert and chatbot QA evaluator. Evaluate responses from the Focus on Energy AI assistant based on program accuracy, completeness, relevance, clarity, and actionability."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0
            },
            timeout=30
        )
        resp_json = response.json()
        if "choices" not in resp_json:
            print(f"    ⚠️ Grading API error: {resp_json}")
            return {"accuracy_helpfulness": 0, "retrieval_quality": 0, "conversational_ux": 0,
                    "safety_compliance": 0, "business_outcomes": 0, "notes": f"API error: {resp_json}"}
        content = resp_json["choices"][0]["message"]["content"].strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except Exception as e:
        print(f"    ⚠️ Grading failed: {e}")
        return {"accuracy_helpfulness": 0, "retrieval_quality": 0, "conversational_ux": 0,
                "safety_compliance": 0, "business_outcomes": 0, "notes": f"Grading error: {e}"}


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
    "safety_compliance", "business_outcomes", "notes",
    "success_rate", "outlier", "responded", "graded_at"
]


def append_results(results):
    file_exists = os.path.isfile(MASTER_CSV)
    with open(MASTER_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    start_date, end_date = get_previous_day_range()
    print(f"📅 Pulling conversations created {start_date} (previous day)\n")

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

        category = categorize_question_with_openai(question)
        grades       = grade_answer_with_openai(question, bot_answer, article_titles)
        success_rate = calculate_success_rate(grades)
        outlier      = detect_outlier(success_rate)

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
            "notes":                grades.get("notes", ""),
            "success_rate":         success_rate,
            "outlier":              outlier,
            "responded":            responded,
            "graded_at":            datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        })

    append_results(results)

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