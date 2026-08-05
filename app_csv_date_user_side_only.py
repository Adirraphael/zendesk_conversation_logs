import requests
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # reads variables from a local .env file (not committed to git)

# ── CONFIG ────────────────────────────────────────────────────────────────────
subdomain  = os.environ["ZENDESK_SUBDOMAIN"]
email      = os.environ["ZENDESK_EMAIL"]
api_token  = os.environ["ZENDESK_API_TOKEN"]
auth       = (email + "/token", api_token)
output_csv = f"zendesk_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# ── FETCH ALL TICKETS ─────────────────────────────────────────────────────────
print("Fetching all tickets...")
tickets_all = []
url    = f"https://{subdomain}.zendesk.com/api/v2/search.json"
params = {"query": "type:ticket", "per_page": 100, "sort_by": "created_at", "sort_order": "asc"}

while url:
    resp = requests.get(url, auth=auth, params=params if not tickets_all else None)
    data = resp.json()
    tickets_all.extend(data.get("results", []))
    url = data.get("next_page")

print(f"  → {len(tickets_all)} total tickets found")

# ── FETCH MESSAGES FROM TICKET AUDITS ────────────────────────────────────────
messages = []
event_types_seen = set()

for i, ticket in enumerate(tickets_all):
    ticket_id    = ticket["id"]
    requester_id = ticket.get("requester_id")
    status       = ticket.get("status")
    print(f"  → Ticket {i+1}/{len(tickets_all)}: #{ticket_id} [{status}]", end="  ")

    audits_resp = requests.get(
        f"https://{subdomain}.zendesk.com/api/v2/tickets/{ticket_id}/audits.json",
        auth=auth
    )
    if audits_resp.status_code != 200:
        print("skip")
        continue

    audits = audits_resp.json().get("audits", [])
    found  = 0

    for audit in audits:
        for event in audit.get("events", []):
            etype = event.get("type", "")
            event_types_seen.add(etype)

            if etype == "ChatStartedEvent":
                history = event.get("value", {}).get("history", [])
                for item in history:
                    if item.get("type") == "ChatMessage" and item.get("actor_type") == "end-user":
                        messages.append({
                            "ticket_id":  ticket_id,
                            "status":     status,
                            "timestamp":  item.get("timestamp"),
                            "ts_source":  "chat",
                            "author_id":  item.get("actor_id"),
                            "message":    item.get("message")
                        })
                        found += 1

            elif etype == "Comment" and event.get("public"):
                if audit.get("author_id") == requester_id:
                    body = event.get("plain_body") or event.get("body", "")
                    if body:
                        messages.append({
                            "ticket_id":  ticket_id,
                            "status":     status,
                            "timestamp":  audit.get("created_at"),
                            "ts_source":  "comment",
                            "author_id":  audit.get("author_id"),
                            "message":    body
                        })
                        found += 1

    print(f"{found} messages")

print(f"\n✅ Total messages: {len(messages)}")

# ── BUILD DATAFRAME ───────────────────────────────────────────────────────────
df = pd.DataFrame(messages)

if not df.empty:
    chat_mask    = df["ts_source"] == "chat"
    comment_mask = df["ts_source"] == "comment"

    chat_ts    = pd.to_datetime(df.loc[chat_mask,    "timestamp"].astype(float), unit="ms", errors="coerce").dt.tz_localize("UTC")
    comment_ts = pd.to_datetime(df.loc[comment_mask, "timestamp"], errors="coerce", utc=True)

    df["timestamp"] = None
    df.loc[chat_mask,    "timestamp"] = chat_ts.astype(object)
    df.loc[comment_mask, "timestamp"] = comment_ts.astype(object)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["timestamp"] = df["timestamp"].astype("datetime64[us, UTC]")

    df = df.drop_duplicates().sort_values("timestamp", ascending=False).reset_index(drop=True)
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["time"] = df["timestamp"].dt.strftime("%H:%M:%S")
    df = df[["ticket_id", "status", "author_id", "message", "date", "time", "timestamp"]]
    df["ticket_id"] = df["ticket_id"].astype(int)
    df["author_id"] = df["author_id"].astype(str)

    print(f"\n📊 {len(df)} messages across {df['ticket_id'].nunique()} tickets\n")

    # ── EXPORT TO CSV ─────────────────────────────────────────────────────────
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Saved to {output_csv}")

else:
    print("⚠️ No messages found.")