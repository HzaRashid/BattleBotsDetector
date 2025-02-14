import os
import json

def correct_tweet_count(json_dir):
    for file_name in ["session_10_results.json"]:
        if file_name.endswith('.json'):
            
            file_path = os.path.join(json_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            users = data.get('users', [])
            posts = data.get('posts', [])

            tweet_count_per_user = {}
            for post in posts:
                author_id = post.get('author_id')
                if author_id:
                    tweet_count_per_user[author_id] = tweet_count_per_user.get(author_id, 0) + 1

            updated = False
            for user in users:
                user_id = user.get('user_id')
                if user_id in tweet_count_per_user and user.get('tweet_count') != tweet_count_per_user[user_id]:
                    user['tweet_count'] = tweet_count_per_user[user_id]
                    updated = True

            if updated:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                print(f"Updated {file_name}")


if __name__ == "__main__":
    f'{os.path.dirname(__file__)}/data'
    json_directory = f'{os.path.dirname(__file__)}/../data'
    correct_tweet_count(json_directory)
