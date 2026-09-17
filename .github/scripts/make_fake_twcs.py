"""
make_fake_twcs.py

CI-only helper. Builds a small synthetic file with the exact schema of the
real Kaggle `twcs.csv` (tweet_id, author_id, inbound, created_at, text,
response_tweet_id, in_response_to_tweet_id) and enough SpotifyCares
(customer -> agent) pairs, spread across all six intents, that
src/classifier.py's stratified train/val split has enough rows per class to
not error out.

This is NOT the real dataset and produces meaningless model numbers -- it
exists purely so CI can exercise src/prepare_data.py -> ... -> src/agent.py
end-to-end on every push without needing Kaggle credentials. See
README.md "Step 1" for how a human gets the real data.
"""
import pandas as pd

EXAMPLES = [
    ("I was charged twice for my premium subscription this month",
     "Hey there! Sorry about that -- can you DM us your account email so we can take a look? /JI"),
    ("the app keeps crashing on my iphone every time I open it",
     "Hey! Sorry to hear that. Can you tell us your device, OS and app version? /AB"),
    ("why is this album not available in my country",
     "Hi there! Unfortunately licensing can vary by region -- more info here. /CD"),
    ("please add a sleep timer feature to the app",
     "Hi! Thanks for the suggestion, we'll pass it on to the team. /EF"),
    ("this is so annoying, fix it please",
     "Hi there, sorry to hear that -- can you tell us more about what's going on? /GH"),
    ("thanks so much, it works now!",
     "You're welcome! Glad it's sorted. /IJ"),
]

N_PER_INTENT = 50


def main():
    rows = []
    tweet_id = 1000
    for i in range(N_PER_INTENT * len(EXAMPLES)):
        cust, agent = EXAMPLES[i % len(EXAMPLES)]
        cust_id, tweet_id = tweet_id, tweet_id + 1
        agent_id, tweet_id = tweet_id, tweet_id + 1
        text = f"{cust} (case {i}, some extra padding words to vary the text)"
        rows.append({
            "tweet_id": str(cust_id), "author_id": "12345", "inbound": "True",
            "created_at": "2018-01-01 00:00:00", "text": text,
            "response_tweet_id": str(agent_id), "in_response_to_tweet_id": "",
        })
        rows.append({
            "tweet_id": str(agent_id), "author_id": "SpotifyCares", "inbound": "False",
            "created_at": "2018-01-01 00:00:00", "text": agent,
            "response_tweet_id": "", "in_response_to_tweet_id": str(cust_id),
        })

    out_dir = "data_raw/archive-2/twcs"
    import os
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{out_dir}/twcs.csv", index=False)
    print(f"Wrote {len(df)} synthetic rows to {out_dir}/twcs.csv")


if __name__ == "__main__":
    main()
