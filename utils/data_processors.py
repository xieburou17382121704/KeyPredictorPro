from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np
import os
import pandas as pd
import pickle
import re

from utils.constants import CLEANING_REGEX, NEGATIONS_DICT


def extend_data(sequences, labels, context_size):
    """
    Extend data to fit the context window size.

    Parameters:
    - sequences: List of input sequences
    - labels: List of output labels
    - context_size: Context window size

    Returns:
    - Extended input and output arrays
    """
    extended_sequences = []
    extended_labels = []
    for idx, sequence in enumerate(sequences):
        label = labels[idx]
        for idx2, word in enumerate(sequence):
            if idx2 < context_size:
                extended_sequences.append(
                    pad_sequences(
                        [sequence[: idx2 + 1]], maxlen=context_size, padding="post"
                    )[0]
                )
            else:
                extended_sequences.append(sequence[idx2 - context_size + 1 : idx2 + 1])
            extended_labels.append(np.array(label[idx2], dtype="int32"))
    extended_sequences = np.vstack(extended_sequences)
    extended_labels = np.array(extended_labels).astype(np.int32)
    return extended_sequences, extended_labels


def clean_text(text, stop_words, stemmer, apply_stemming=False):
    """
    Clean text by removing stopwords and applying stemming.

    Parameters:
    - text: Original text
    - stop_words: List of stopwords
    - stemmer: Stemmer object
    - apply_stemming: Boolean to apply stemming

    Returns:
    - Cleaned text
    """
    negation_pattern = re.compile(r"\b(" + "|".join(NEGATIONS_DICT.keys()) + r")\b")
    text = re.sub(CLEANING_REGEX, " ", str(text).lower()).strip()
    text = negation_pattern.sub(lambda x: NEGATIONS_DICT[x.group()], text)
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if apply_stemming:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    cleaned_text = " ".join(tokens)
    cleaned_text = re.sub("n't", "not", cleaned_text)
    return re.sub("'s", "is", cleaned_text)


def split_local_remote_data(data, local_share=0.1):
    """
    Split data into local and remote datasets.

    Parameters:
    - data: Dataset
    - local_share: Proportion of data to be local

    Returns:
    - Local data and remote data
    """
    unique_users = data.user.unique()
    split_index = int(local_share * unique_users.shape[0])
    local_users = unique_users[:split_index]
    remote_users = unique_users[split_index:]
    local_data = data[data.user.isin(local_users)]
    remote_data = data[data.user.isin(remote_users)]
    return local_data, remote_data


def index_data_by_date(data, timezone_str="PDT"):
    """
    Index data by date and localize timezone.

    Parameters:
    - data: Dataset
    - timezone_str: Timezone string

    Returns:
    - Indexed data
    """
    timezone = "US/Pacific" if "PDT" or "PT" in timezone_str else "UTC"
    data.date = data.date.str.replace(timezone_str, "")
    data.date = data.date.astype("datetime64[ns]")
    data.index = data.date
    data.drop(["date"], axis=1, inplace=True)
    data.index = data.index.tz_localize(timezone)
    return data


def merge_and_index_data(input_file, output_file, word_index_file, min_tweets=20):
    """
    Merge and index data, process text, and generate sequences.

    Parameters:
    - input_file: Path to input file
    - output_file: Path to output file
    - word_index_file: Path to word index file
    - min_tweets: Minimum number of tweets per user

    Returns:
    - Processed data and word index
    """
    if os.path.isfile(output_file) and os.path.isfile(word_index_file):
        data = pd.read_pickle(output_file)
        with open(word_index_file, "rb") as f:
            word_index = pickle.load(f)
        return data, word_index
    columns = ["target", "ids", "date", "flag", "user", "text"]
    stop_words = stopwords.words("english")
    stemmer = SnowballStemmer("english")
    data = pd.read_csv(input_file, encoding="ISO-8859-1", header=None, names=columns)
    data.drop(["target", "flag", "ids"], axis=1, inplace=True)
    valid_users = data.groupby(by="user").apply(len) > min_tweets
    data = data[data.user.isin(valid_users[valid_users].index)]
    data = index_data_by_date(data)
    data["cleaned_text"] = data.text.apply(lambda x: clean_text(x, stop_words, stemmer))
    data.drop_duplicates(subset=["cleaned_text"], keep=False, inplace=True)
    sequences, tokenizer = text_to_sequence(data.cleaned_text)
    data["sequence"] = sequences
    data = data[data.sequence.map(lambda x: len(x)) > 0]
    data = data.merge(
        data.sequence.apply(lambda x: split_sequences(x)),
        left_index=True,
        right_index=True,
    )
    data.to_pickle(output_file)
    with open(word_index_file, "wb") as f:
        pickle.dump(tokenizer.word_index, f, pickle.HIGHEST_PROTOCOL)
    return data, tokenizer.word_index


def split_sequences(sequence):
    """
    Split sequence into input and output.

    Parameters:
    - sequence: List of sequences

    Returns:
    - Series of input and output
    """
    inputs = [0]
    outputs = [sequence[0]]
    for idx, token in enumerate(sequence[:-1]):
        inputs.append(token)
        outputs.append(sequence[idx + 1])
    return pd.Series({"inputs": inputs, "outputs": outputs})


def text_to_sequence(texts):
    """
    Convert text to sequences.

    Parameters:
    - texts: List of texts

    Returns:
    - Sequences and Tokenizer object
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    return tokenizer.texts_to_sequences(texts), tokenizer
