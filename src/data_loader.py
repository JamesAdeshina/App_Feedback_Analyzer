import pandas as pd

def load_data(filepath):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(filepath)
    print(f"Loaded data with {len(data)} rows and {len(data.columns)} columns.")
    return data

def clean_data(data):
    """Perform basic cleaning of the dataset."""
    data.dropna(subset=['review', 'star'], inplace=True)  # Remove rows with missing reviews or stars
    data.drop_duplicates(subset=['review'], inplace=True)  # Remove duplicate reviews
    print(f"Data cleaned. Remaining rows: {len(data)}")
    return data

if __name__ == "__main__":
    filepath = "../data/raw_reviews.csv"
    data = load_data(filepath)
    cleaned_data = clean_data(data)
    cleaned_data.to_csv("../data/processed_reviews.csv", index=False)
