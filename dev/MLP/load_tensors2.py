import os
import ijson
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from time_utils import generate_time_dna, encode_time_dna_batch_cnn
from content_utils import generate_content_dna, encode_content_dna_batch_cnn
from transformers import AutoTokenizer

# Few-shot examples for descriptions (updated to prevent data leakage).
bot_examples = [
    "Pastoralist(mulalo) by birth, interested in Microcredit & ICT. Treasures meaningful work and meaningful relationships",
    "Non-linear human in a complicated world. Currently PhD @EPFL. Previously @IITHyderabad .",
    "[no description]",
    "Engineer | An Introvert Nerd\n\nBengaluru - Barmer",
    "Trying with rthe lords faith  To heal the world for the future of all life. So we don't dread a future of leaving mother Earth.\n\ud83e\udd81CATMAN\n\ud83d\ude4f\u2764\ufe0f\ud83e\udd81\ud83c\udf0d",
]

human_examples = [
    "News and expert commentary on health & medicine, science, business, law and the arts from Washington University in St. Louis. #WashU",
    "Soy amiga de mis amigos, muy visceral, me gusta disfrutar de la vida y defensora de las causas perdidas",
    "Writer, journalist, anthropologist | Bylines in @TexasMonthly & @Harpers | Founder of Culture Concepts consulting | Former @UTAustin prof | Tejana & fronteriza",
    "Cryptographer. Implicit notation supporter."
]

# Simplified prompt for DNA modality.
dna_prompt_base = (
    "Task: Evaluate whether a Twitter account is operated by a human or a bot using two types of activity DNA—Content DNA and Time DNA—that capture the user's tweet characteristics.\n\n"
    "[Background & Definitions]\n"
    "Content DNA: Each letter represents a tweet's content characteristics.\n"
    " - N: No entities in tweet\n"
    " - H: Hashtag present in tweet\n"
    " - M: @mention present in tweet\n"
    " - X: Multiple entities present in tweet\n\n"
    "Time DNA: Each letter represents the interval between tweets.\n"
    " - n: User's first post\n"
    " - o: Less than 1 second since previous post\n"
    " - s: Less than 1 minute since previous post\n"
    " - m: Less than 1 hour since previous post\n"
    " - h: Less than 6 hours since previous post\n"
    " - u: Less than 12 hours since previous post\n"
    " - x: More than 12 hours since previous post\n\n"
    "[Examples]\n"
    "Example 1: Content DNA: NHMX, Time DNA: nsmhu => Likely human.\n"
    "Example 2: Content DNA: XXXX, Time DNA: ooooo => Likely bot.\n\n"
    "Now, here is the Content DNA for the user: "
)

# Simplified prompt for description modality.
desc_prompt_base = (
    "Task: Evaluate whether a Twitter account is operated by a human or a bot based solely on the user's self-written description.\n\n"
)

# Load the FLAN-T5 Small tokenizer.
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
BATCH_SIZE = 64

def sanitize_text(text):
    """
    Returns a UTF-8 safe version of the text by replacing invalid surrogate characters.
    """
    return text.encode('utf-8', 'replace').decode('utf-8')

def get_dna_prompt(content_dna, time_dna):
    """
    Constructs a DNA prompt by appending the content and time DNA information
    to the base prompt, then sanitizes the result.
    """
    prompt = (
        dna_prompt_base + str(content_dna) +
        ".\nHere is the Time DNA for the user: " + str(time_dna) +
        ".\nPlease provide your answer as either 'likely bot' or 'likely human'" 
    )
    return sanitize_text(prompt)

def get_desc_prompt(desc, add_examples=True):
    """
    Constructs a description prompt that first displays few-shot examples and
    then provides the user's actual description after the examples.
    If the description is empty, a default placeholder is used.
    The resulting prompt is sanitized.
    """
    desc = str(desc) if desc is not None else ""
    if not desc.strip():
        desc = "No description provided."
    
    # Start with the base instructions.
    prompt = desc_prompt_base
    # Append few-shot examples by default.
    if add_examples:
        prompt += "\n[Few-shot Examples]\nBot Descriptions:\n"
        for ex in bot_examples:
            prompt += " - " + (str(ex) if ex else "[empty]") + "\n"
        prompt += "\nHuman Descriptions:\n"
        for ex in human_examples:
            prompt += " - " + str(ex) + "\n"
    # Finally, provide the user's actual description.
    prompt += "\nNow, here is the user's actual description:\n" + desc
    prompt += "\n\nPlease provide your answer as either 'likely bot' or 'likely human'"
    return sanitize_text(prompt)

def load_data(data_dir, session_numbers=[], st_model=None, xnums=[]):
    """
    Loads user and post data from JSON files using ijson (streaming), extracts features,
    and computes text embeddings and tokenized prompts for each modality.
    
    Returns two dictionaries (train and test) containing:
      - "desc_embs"   : Description embeddings (from the sentence transformer model).
      - "dna_embs"    : DNA embeddings (from a CNN).
      - "time_embs"   : Time embeddings (from a CNN).
      - "desc_tokens" : Tokenized description prompts (via FLAN-T5).
      - "dna_tokens"  : Tokenized DNA prompts (via FLAN-T5).
      - "labels"      : NumPy array of binary labels (bot or human).
    """
    user_info_list = []
    user_posts_dict = {}
    datasets = []

    if session_numbers:
        datasets += [f"session_{num}_results.json" for num in session_numbers]
    if xnums:
        datasets += [f"twibot22/processed/tweet_{num}_processed.json" for num in xnums]

    # Stream JSON files to extract users and posts.
    for fname in datasets:
        json_file = os.path.join(data_dir, fname)
        with open(json_file, "r") as f:
            for key, items in ijson.kvitems(f, ""):
                if key == "users":
                    for user in items:
                        user_info_list.append(user)
                elif key == "posts":
                    for post in items:
                        uid = post.get("user_id") or post.get("author_id")
                        if uid is not None:
                            user_posts_dict.setdefault(uid, []).append(post)

    # Create a DataFrame for user metadata and perform a stratified train/test split.
    user_info_df = pd.DataFrame(user_info_list)[['user_id', 'is_bot']].drop_duplicates()
    train_users, test_users = train_test_split(
        user_info_df, test_size=0.2, random_state=42, stratify=user_info_df['is_bot']
    )
    train_user_ids = set(train_users['user_id'])
    test_user_ids = set(test_users['user_id'])

    # Prepare modality lists.
    train_desc_texts, test_desc_texts = [], []
    train_dna_list, test_dna_list = [], []
    train_time_list, test_time_list = [], []
    train_labels, test_labels = [], []
    train_desc_prompts, test_desc_prompts = [], []
    train_dna_prompts, test_dna_prompts = [], []

    for user in user_info_list:
        uid = user.get("user_id")
        if uid is None or uid not in user_posts_dict:
            continue

        description = str(user.get("description", ""))
        generated_desc_prompt = get_desc_prompt(description, add_examples=True)
        content_dna = str(generate_content_dna(user_posts_dict[uid]))
        time_dna = str(generate_time_dna(user_posts_dict[uid]))
        generated_dna_prompt = get_dna_prompt(content_dna, time_dna)

        label = int(user.get("is_bot", 0))

        if uid in train_user_ids:
            train_desc_texts.append(description)
            train_dna_list.append(content_dna)
            train_time_list.append(time_dna)
            train_desc_prompts.append(generated_desc_prompt)
            train_dna_prompts.append(generated_dna_prompt)
            train_labels.append(label)
        elif uid in test_user_ids:
            test_desc_texts.append(description)
            test_dna_list.append(content_dna)
            test_time_list.append(time_dna)
            test_desc_prompts.append(generated_desc_prompt)
            test_dna_prompts.append(generated_dna_prompt)
            test_labels.append(label)

    # Ensure all entries are strings.
    train_desc_prompts = [str(p) for p in train_desc_prompts]
    test_desc_prompts = [str(p) for p in test_desc_prompts]
    train_dna_prompts = [str(p) for p in train_dna_prompts]
    test_dna_prompts = [str(p) for p in test_dna_prompts]
    train_desc_texts = [str(t) for t in train_desc_texts]
    test_desc_texts = [str(t) for t in test_desc_texts]

    # Compute description embeddings using the sentence transformer.
    train_desc_embs = st_model.encode(train_desc_texts, batch_size=BATCH_SIZE)
    test_desc_embs = st_model.encode(test_desc_texts, batch_size=BATCH_SIZE)

    # Tokenize the prompts using the FLAN-T5 tokenizer.
    train_desc_tokenized = tokenizer(train_desc_prompts, padding=True, truncation=False, return_tensors="pt")
    train_dna_tokenized = tokenizer(train_dna_prompts, padding=True, truncation=False, return_tensors="pt")
    test_desc_tokenized = tokenizer(test_desc_prompts, padding=True, truncation=False, return_tensors="pt")
    test_dna_tokenized = tokenizer(test_dna_prompts, padding=True, truncation=False, return_tensors="pt")

    # Generate CNN embeddings for the DNA modalities.
    train_dna_embs = encode_content_dna_batch_cnn(train_dna_list)
    train_time_embs = encode_time_dna_batch_cnn(train_time_list)
    test_dna_embs = encode_content_dna_batch_cnn(test_dna_list)
    test_time_embs = encode_time_dna_batch_cnn(test_time_list)

    # Construct dictionaries for the train and test splits.
    train = {
        "desc_embs": train_desc_embs,
        "dna_embs": train_dna_embs,
        "time_embs": train_time_embs,
        "desc_tokens": train_desc_tokenized,
        "dna_tokens": train_dna_tokenized,
        "labels": np.array(train_labels)
    }
    
    test = {
        "desc_embs": test_desc_embs,
        "dna_embs": test_dna_embs,
        "time_embs": test_time_embs,
        "desc_tokens": test_desc_tokenized,
        "dna_tokens": test_dna_tokenized,
        "labels": np.array(test_labels)
    }
    
    return train, test

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer, models
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, "../data")

    # Initialize the transformer and pooling models.
    transformer_model = models.Transformer("Twitter/twhin-bert-base", model_args={'attn_implementation': 'eager'})
    pooling_model = models.Pooling(
        transformer_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )
    st_model = SentenceTransformer(modules=[transformer_model, pooling_model])

    # For a quick test, get the token count for a DNA prompt.
    prompt = get_desc_prompt("Your input description here...")
    token_ids = tokenizer.encode(prompt, truncation=False)
    print("Token count:", len(token_ids))
    
    # To load the data, uncomment the following lines:
    # train, test = load_data(data_dir=data_dir, session_numbers=[10], xnums=[], st_model=st_model)
    # print("Train description embeddings shape:", train["desc_embs"].shape)
    # ... (other print statements)
