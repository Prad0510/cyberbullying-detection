import re
import html
import unicodedata


def basic_clean_text(text):
    """
    Performs basic cleaning of social-media text while
    preserving information useful for cyberbullying detection.
    """

    if not isinstance(text, str):
        return ""

    # Decode HTML entities such as &amp; and &lt;
    text = html.unescape(text)

    # Normalize Unicode representation
    text = unicodedata.normalize("NFKC", text)

    # Remove retweet marker at the beginning
    text = re.sub(r"^\s*RT\s+", "", text, flags=re.IGNORECASE)

    # Replace URLs with a placeholder
    text = re.sub(r"https?://\S+|www\.\S+", " <URL> ", text)

    # Replace user mentions with a placeholder
    text = re.sub(r"@\w+", " <USER> ", text)

    # Remove unnecessary space before punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    # Normalize repeated whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text