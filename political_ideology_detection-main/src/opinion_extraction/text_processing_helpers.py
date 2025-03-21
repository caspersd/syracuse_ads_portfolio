import spacy

# Load SpaCy NLP model once to avoid redundant loads
nlp = spacy.load("en_core_web_sm")

def spacy_tokenizer(sentence):
    """
    Tokenizes a sentence into words.

    Args:
        sentence (str): Input sentence.

    Returns:
        list: List of tokenized words.
    """
    return [token.text for token in nlp(sentence)]

def spacy_pos_tokenizer(sentence):
    """
    Tokenizes a sentence and returns words with their POS tags.

    Args:
        sentence (str): Input sentence.

    Returns:
        list: List of words appended with POS tags (e.g., "word_VERB").
    """
    return [f"{token.text}_{token.pos_}" for token in nlp(sentence)]

def spacy_pos_count_tokenizer(sentence):
    """
    Tokenizes a sentence and extracts both words and POS tags separately.

    Args:
        sentence (str): Input sentence.

    Returns:
        list: List containing words and POS tags.
    """
    tokens = nlp(sentence)
    return [token.text for token in tokens] + [token.pos_ for token in tokens]
