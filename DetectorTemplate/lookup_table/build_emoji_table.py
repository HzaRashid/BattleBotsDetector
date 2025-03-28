import re

# Define a mapping from Unicode emoji groups to a single letter.
group_to_letter = {
    "Smileys & Emotion":    "S",
    "People & Body":        "P",
    "Animals & Nature":     "A",
    "Food & Drink":         "F",
    "Travel & Places":      "T",
    "Activities":           "R",   # R for recreation
    "Objects":              "O",
    "Symbols":              "Y",      # Y chosen arbitrarily for symbols
    "Flags":                "L"         # L for flags
}

# This will store our emoji-to-category letter lookup table.
emoji_lookup = {}
import os

current_group = None
emoji_file_path = os.path.join(os.path.dirname(__file__), "emoji-test.txt")
with open(emoji_file_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        # Look for a group header (e.g., "# group: Smileys & Emotion")
        if line.startswith("# group:"):
            current_group = line.split(":", 1)[1].strip()
        # Skip comments and blank lines
        elif line.startswith("#") or not line:
            continue
        else:
            # The expected format is:
            # Code points ; status # emoji E<version> description
            # We split on ";" to get the code points portion.
            parts = line.split(";")
            if len(parts) < 2:
                continue
            code_str = parts[0].strip()
            # Some emoji are sequences (e.g., with skin tone modifiers)
            codepoints = code_str.split()
            try:
                emoji_char = "".join(chr(int(cp, 16)) for cp in codepoints)
            except ValueError:
                continue
            # Look up the letter for the current group.
            if current_group in group_to_letter:
                letter = group_to_letter[current_group]
                emoji_lookup[emoji_char] = letter

# Now emoji_lookup is a dictionary mapping each emoji (as a string) to its category letter.
print(emoji_lookup)


import json
import os
save_path = os.path.join(os.path.dirname(__file__), "emoji_lookup.json")
# Assuming emoji_lookup is the dictionary built from the emoji-test.txt file:
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(emoji_lookup, f, ensure_ascii=False, indent=2)
