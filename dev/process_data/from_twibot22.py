import os
import ijson
import numpy.random as nprand
import json
import polars as pl
from datetime import datetime


# Function to convert time format
def convert_time_format(timestamp):
    if not timestamp:
        return ""  # Handle missing timestamps
    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S%z")
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"
    except ValueError:
        return ""  # Handle incorrect formats gracefully
    
tweet_ds_num = 2
tweet_json_filename = f'tweet_{tweet_ds_num}.json'
output_json_filename = f'tweet_{tweet_ds_num}_processed.json'
cur_dir = os.path.dirname(__file__)
data_dir = os.path.join(cur_dir, '../data/twibot22')

labels_path = os.path.join(data_dir, 'label.csv')
tweet_json_path = os.path.join(data_dir, tweet_json_filename)
user_json_path = os.path.join(data_dir, 'user.json')
output_json_path = os.path.join(data_dir, 'processed', output_json_filename)

# Load labels CSV
labels_df = pl.read_csv(labels_path)

# Convert user_id to string for matching with JSON
labels_df = labels_df.with_columns(
    pl.col("id")
    .cast(pl.Utf8)
    .str.replace("^u", "")
    .alias("id"),
    pl.col("label")=="bot"
).rename({'label': 'is_bot'})

# Process tweet JSON
tweet_records = []
# max_entries = 50_000

with open(tweet_json_path, 'r', encoding='utf-8') as f:
    for idx, item in enumerate(ijson.items(f, 'item')):
        tweet_records.append({
            "author_id": str(item.get("author_id")),
            "text": item.get("text", ""),
            "lang": "en",
            "id": item.get("id", ""),
            "created_at": convert_time_format(item.get("created_at", ""))
        })
        # if idx + 1 >= max_entries:
        #     break

# Process user JSON
user_records = []
with open(user_json_path, 'r', encoding='utf-8') as f:
    for item in ijson.items(f, 'item'):
        user_records.append({
            "author_id": str(item.get("id"))[1:],  # Removing leading character
            "description": item.get("description", ""),
            "location": item.get("location", ""),
            "username": item.get("username", ""),
        })

rand_state = nprand.RandomState(seed=42)
# Create DataFrames
tweet_json_df = pl.DataFrame(tweet_records)
# keep first 10-120 tweets from each user
tweet_json_df = (
    tweet_json_df
    .sort("created_at")  # Sort by created_at before grouping
    .group_by("author_id")
    .head(rand_state.choice(range(10, 121)))
)
# Join user data with labels
users_json_df = pl.DataFrame(user_records)\
    .join(labels_df, left_on='author_id', right_on='id', how='left')


# keep only users from the tweets dataset
present_users_df = tweet_json_df\
    .join(users_json_df, on='author_id', how='left')\
    .unique(subset=['author_id'])\
        .drop(['text', 'created_at', 'lang', 'id'])\
        .rename({'author_id': 'user_id'})


# select humans and bots with   ~1:12 (bot:human) ratio
human_users_df = present_users_df.filter(
    is_bot = 0
)[:1000]

bot_users_df = present_users_df.filter(
    is_bot = 1
)[:82]

print("number of human users:", len(human_users_df))
print("number of bot users:", len(bot_users_df))

human_tweets_df = (tweet_json_df
                   .join(human_users_df, left_on="author_id", right_on="user_id", how="right")
                   .rename({"user_id": "author_id"})
                   .drop(["description", "location", "is_bot", "username"]))
bot_tweets_df = (tweet_json_df
                 .join(bot_users_df, left_on="author_id", right_on="user_id", how="right")
                 .rename({"user_id": "author_id"})
                 .drop(["description", "location", "is_bot", "username"]))


final_tweets_df = pl.concat([human_tweets_df, bot_tweets_df], rechunk=True)
final_users_df = pl.concat([human_users_df, bot_users_df], rechunk=True)

print("len tweet_json_df:", len(tweet_json_df))
print("len final_tweets_df:", len(final_tweets_df))
print("len human_tweets_df:", len(human_tweets_df))
print("len bot_tweets_df:", len(bot_tweets_df))
print("len final_users_df:", len(final_users_df))
print("len human_users_df:", len(human_users_df))
print("len bot_users_df:", len(bot_users_df))

# Convert to JSON format
output_data = {
    "id": f"tweet_{tweet_ds_num}_processed_test",
    "posts": final_tweets_df.to_dicts(),
    "users": final_users_df.to_dicts()
}

# Save JSON file
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2)

print(f"Saved processed data to {output_json_path}")
