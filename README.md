Scripts
app_csv_date_user_side_only.py

Pulls all Zendesk tickets and exports only the customer's messages (questions) to a timestamped CSV — no bot/agent replies included. Useful when you only need what customers asked, not how the bot responded.

app_csv_full_conversation.py

Same as above, but captures both sides of the conversation — customer messages and bot/agent replies — with a role column marking who sent each message (customer vs bot/agent). This is the input other scripts (like the scoring script) expect.


Typical workflow
Run app_csv_full_conversation.py → produces zendesk_conversations_*.csv
Run conversation_logs_scoring.py (pointed at that CSV) → produces zendesk_conversations_scored.csv
