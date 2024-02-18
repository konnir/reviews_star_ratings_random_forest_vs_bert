import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def category_to_int(category: str) -> int:
    """ Convert a category string to an integer. """
    category_dict = {'Prime_Pantry': 0,
                     'Grocery_and_Gourmet_Food': 1,
                     'Pet_Supplies': 2,
                     'Arts_Crafts_and_Sewing': 3,
                     'Office_Products': 4,
                     'Cell_Phones_and_Accessories': 5,
                     'Video_Games': 6,
                     'Patio_Lawn_and_Garden': 7,
                     'Software': 8,
                     'Musical_Instruments': 9,
                     'Industrial_and_Scientific': 10,
                     'Luxury_Beauty': 11,
                     'All_Beauty': 12,
                     'AMAZON_FASHION': 13,
                     'Digital_Music': 14,
                     'Appliances': 15}
    return category_dict[category]

def price_to_float(price: str) -> float:
    """ Convert a price string to a float. """
    # Use regex to extract numeric values correctly and handle cases where conversion might fail
    if price is None:
        price = "9.99$"
    match = re.search(r'(\d+\.\d+|\d+)', price)
    if match:
        # If a match is found, convert the matched string to float
        return float(match.group(0))
    else:
        # If no numeric pattern is found, handle it as needed (e.g., return None or raise an error)
        return 9.99

def verified_to_int(verified: bool) -> int:
    """ Convert a verified string to an integer. """
    if verified:
        return 1
    else:
        return 0

# Function to clean text
def review_text_to_clean(text: str) -> str:
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercasing
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text

