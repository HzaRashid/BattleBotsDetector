import pandas as pd
import os



df = pd.read_csv(filepath_or_buffer=os.path.join(os.path.dirname(__file__), 'foo/check_features.csv'))

df = df[['user_id','most_common_topic','tweet_time_mean','tweet_time_std','time_bucket_duplicates','tweet_count','is_bot']]

df.to_csv(path_or_buf=os.path.join(os.path.dirname(__file__), 'foo/trunc.csv'))