import csv
import json
import os
import re
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # reads variables from a local .env file (not committed to git)

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_INPUT_FILE  = "zendesk_conversations_logs.csv"     # <-- your existing per-message export
OUTPUT_CSV      = "zendesk_conversations_scored.csv"
OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]

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

# Messages to drop entirely before pairing Q&A (satisfaction prompts, not real answers)
SKIP_PHRASES = [
    "how would you rate your experience",
    "how was your experience",
    "talk to human",
]


# ── CATEGORIZE WITH OPENAI ────────────────────────────────────────────────────

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


# ── GRADE WITH OPENAI ─────────────────────────────────────────────────────────

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


# ── HELPERS ───────────────────────────────────────────────────────────────────

def calculate_success_rate(grades):
    total = sum(grades.get(k, 0) for k in
                ["accuracy_helpfulness", "retrieval_quality", "conversational_ux",
                 "safety_compliance", "business_outcomes"])
    return round((total / 25) * 100, 1)


def detect_outlier(success_rate):
    return "YES" if success_rate < 50 else "NO"


def extract_sources(bot_answer):
    """Split on 'Sources:' — everything before is the answer, everything after is articles."""
    if "Sources:" in bot_answer:
        parts = bot_answer.split("Sources:", 1)
        return parts[0].strip(), parts[1].strip()
    elif "sources:" in bot_answer.lower():
        idx = bot_answer.lower().index("sources:")
        return bot_answer[:idx].strip(), bot_answer[idx+8:].strip()
    return bot_answer.strip(), ""


def is_skip_message(text):
    lower = text.lower()
    return any(phrase in lower for phrase in SKIP_PHRASES)


def is_emoji_only(text):
    """Return True if the text has no real alphabetic content — just emojis/symbols."""
    import unicodedata
    clean = text.strip()
    if not clean:
        return True
    return not any(
        unicodedata.category(c).startswith('L')
        for c in clean
    )


# ── LOAD & PAIR Q&A FROM EXISTING PER-MESSAGE CSV ─────────────────────────────

def load_and_pair(path):
    """
    Reads the per-message export (one row per chat bubble) and groups
    consecutive customer -> system messages into Q&A pairs per ticket,
    preserving chronological order and original columns for traceability.
    """
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))

    # Group by ticket_id, keep chronological order (assumes rows already sorted;
    # re-sort defensively by timestamp if present)
    def sort_key(r):
        return r.get("timestamp", "") or r.get("date", "")

    tickets = {}
    for r in rows:
        tickets.setdefault(r.get("ticket_id", ""), []).append(r)

    pairs = []
    for ticket_id, msgs in tickets.items():
        msgs = sorted(msgs, key=sort_key)

        current_question = None
        current_q_row     = None
        current_answer_parts = []
        current_a_rows    = []

        def flush():
            if current_question is not None:
                answer_text = " ".join(current_answer_parts).strip()
                pairs.append({
                    "ticket_id":        ticket_id,
                    "status":           current_q_row.get("status", ""),
                    "question_date":    current_q_row.get("date", ""),
                    "question_time":    current_q_row.get("timestamp", current_q_row.get("time", "")),
                    "answer_time":      current_a_rows[-1].get("timestamp", current_a_rows[-1].get("time", "")) if current_a_rows else "",
                    "question":         current_question,
                    "bot_answer_raw":   answer_text,
                })

        for r in msgs:
            role = (r.get("role") or "").strip().lower()
            message = (r.get("message") or "").strip()
            if not message:
                continue
            if is_skip_message(message):
                continue
            if is_emoji_only(message):
                continue

            if role == "customer":
                # New question starts — flush the previous pair first
                flush()
                current_question = message
                current_q_row = r
                current_answer_parts = []
                current_a_rows = []
            else:
                # system / bot / agent message — attach to current question if one is open
                if current_question is not None:
                    current_answer_parts.append(message)
                    current_a_rows.append(r)
                # if there's no open question yet (e.g. bot greeting before any customer msg), skip it

        flush()  # flush the last pair in this ticket

    return pairs


# ── SAVE ──────────────────────────────────────────────────────────────────────

def save_results(results):
    if not results:
        print("⚠️ No results to save.")
        return
    fieldnames = [
        "ticket_id", "status", "question_date", "question_time", "answer_time",
        "question", "category", "bot_answer", "article_titles",
        "accuracy_helpfulness", "retrieval_quality", "conversational_ux",
        "safety_compliance", "business_outcomes", "notes",
        "success_rate", "outlier", "responded", "graded_at"
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total       = len(results)
    responded   = sum(1 for r in results if r["responded"] == "YES")
    outliers    = sum(1 for r in results if r["outlier"] == "YES")
    avg_success = sum(r["success_rate"] for r in results) / total if total else 0
    print(f"\n{'='*60}")
    print(f"✅ Responded: {responded}/{total}   ❌ No response: {total-responded}/{total}")
    print(f"⚠️  Outliers: {outliers}/{total}")
    print(f"⭐ Average success rate: {avg_success:.1f}%")
    print(f"📄 Results saved to: {OUTPUT_CSV}")
    print(f"{'='*60}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    pairs = load_and_pair(CSV_INPUT_FILE)
    print(f"📋 Paired {len(pairs)} question/answer turns from CSV\n")

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

        print(f"    → Categorizing...")
        category = categorize_question_with_openai(question)
        print(f"    🏷️  Category: {category}")

        print(f"    → Grading with GPT-4o...")
        grades       = grade_answer_with_openai(question, bot_answer, article_titles)
        success_rate = calculate_success_rate(grades)
        outlier      = detect_outlier(success_rate)

        print(f"    📊 A:{grades.get('accuracy_helpfulness',0)} "
              f"R:{grades.get('retrieval_quality',0)} "
              f"UX:{grades.get('conversational_ux',0)} "
              f"S:{grades.get('safety_compliance',0)} "
              f"B:{grades.get('business_outcomes',0)}  "
              f"⭐{success_rate}%  Outlier:{outlier}  Responded:{responded}\n")

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
            "graded_at":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    save_results(results)


if __name__ == "__main__":
    main()

