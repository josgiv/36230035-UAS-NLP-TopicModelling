
import re
import html
import nltk
from functools import lru_cache
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure resources
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('omw-1.4', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
# Add common blog noise words
STOP_WORDS.update(['urlLink', 'urllink', 'nbsp', 'http', 'com'])
lemmatizer = WordNetLemmatizer()

# Contraction Map
CONTRACTIONS = {
    "ain't": "is not", "aren't": "is not", "can't": "cannot", "'cause": "because",
    "could've": "could have", "couldn't": "could not", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
    "how'd": "how did", "how'll": "how will", "how's": "how is", "i'd": "i would",
    "i'll": "i will", "i'm": "i am", "i've": "i have", "isn't": "is not",
    "it'd": "it would", "it'll": "it will", "it's": "it is", "let's": "let us",
    "ma'am": "madam", "might've": "might have", "must've": "must have", "needn't": "need not",
    "shan't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "that'd": "that would",
    "that's": "that is", "there'd": "there would", "there's": "there is", "they'd": "they would",
    "they'll": "they will", "they're": "they are", "they've": "they have",
    "wasn't": "was not", "we'd": "we would", "we'll": "we will", "we're": "we are",
    "we've": "we have", "weren't": "were not", "what'll": "what will", "what're": "what are",
    "what's": "what is", "what've": "what have", "where's": "where is", "who's": "who is",
    "who'll": "who will", "who've": "who have", "won't": "will not", "would've": "would have",
    "wouldn't": "would not", "y'all": "you all", "you'd": "you would", "you'll": "you will",
    "you're": "you are", "you've": "you have"
}

# Regex
re_url = re.compile(r'https?://\S+|www\.\S+')
re_html = re.compile(r'<.*?>')
re_alpha = re.compile(r'[^a-z\s]')
re_spaces = re.compile(r'\s+')

@lru_cache(maxsize=50000)
def cached_lemmatize(word):
    return lemmatizer.lemmatize(word)

def clean_text(text):
    if not isinstance(text, str):
        return ''

    # Decoding and Lowercase
    text = html.unescape(text).lower()

    # Remove URLs and HTML
    text = re_url.sub(' ', text)
    text = re_html.sub(' ', text)

    # Expand Contractions
    for k, v in CONTRACTIONS.items():
        if k in text:
            text = text.replace(k, v)

    # Keep only alphabets
    text = re_alpha.sub(' ', text)

    # Collapse spaces
    text = re_spaces.sub(' ', text).strip()
    return text

def process_batch(batch_texts):
    processed = []
    # Localize globals for speed loop
    local_stop = STOP_WORDS
    local_lem = cached_lemmatize

    for text in batch_texts:
        cleaned = clean_text(str(text))
        tokens = cleaned.split()
        final_tokens = [local_lem(w) for w in tokens if len(w) > 2 and w not in local_stop]
        processed.append(" ".join(final_tokens))
    return processed
