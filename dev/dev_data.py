import pandas as pd
import spacy
import os
import re

data_path = os.path.join(os.path.dirname(__file__), "../data/bot_detection_data.csv.zip")
# Load spaCy's English model
nlp = spacy.load("en_core_web_sm", enable=['tokenizer', 'stopwords', 'lemmatizer'])

def preprocess_text(text):
    text = re.sub(r"https?://\S+", "<URLURL>", text)
    text = re.sub(r"@\w+", "<UsernameMention>", text)
    text = re.sub(r"#\w+", "<HashtagMention>", text)
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def get_df():
    cols = ['User ID', 'Username', 'Created At', 'Tweet', 'Location', 'Bot Label']
    data = pd.read_csv(filepath_or_buffer=data_path, 
                    compression='zip', usecols=cols, )

    print("num rows:", len(data))

    print("num users:", len(data['Username'].unique()))

    return data


# def process_data():
#     data = get_df()
#     users_df = pd.DataFrame()
#     posts_df = pd.DataFrame(dataset1["posts"] + dataset2["posts"] + dataset3["posts"])

#     posts_df = posts_df[['author_id', 'text', 'created_at']]
#     users_df = users_df[['user_id', 'is_bot']]

#     posts_df["cleaned_text"] = posts_df["text"].apply(preprocess_text)
#     user_tweets = posts_df.groupby("author_id")["cleaned_text"].apply(lambda x: " ".join(x)).reset_index()
#     users_df = users_df.merge(user_tweets, left_on="user_id", right_on="author_id", how="left")
#     users_df.drop(columns=["author_id"], inplace=True)

#     return users_df

def main():
    data = get_df()
    # data["cleaned_text"] = data["Tweet"].apply(preprocess_text)
    # user_tweets = data.groupby(["Username", "Bot Label"])["cleaned_text"].apply(lambda x: " ".join(x)).reset_index()
    # user_tweets["Bot Label"] = 
    print(
       data[data['Username'] == 'aalvarez'][['User ID', 'Username', 'Tweet', 'Location', 'Bot Label']]
    )

if __name__ == "__main__":
    main()