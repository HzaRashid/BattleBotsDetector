import pandas as pd
import json

def process_data():
    with open("./data/session_10_results.json", "r", encoding="utf-8") as file:
        dataset = json.load(file)
    # print(dataset)
    users_df = pd.DataFrame(dataset["users"])
    posts_df = pd.DataFrame(dataset["posts"])

    # posts_df.drop(columns=['id', 'lang'])
    posts_df = posts_df[['author_id', 'created_at', 'text']]
    posts_df.sort_values('author_id', inplace=True)
    print(posts_df.head())

    users_df = users_df[['user_id', 'tweet_count', 'username', 'name', 'description', 'location', 'is_bot']]
    users_df.sort_values('user_id', inplace=True)

    print(users_df.head())

if __name__ == "__main__":
    process_data()