import os
import pandas as pd
from dotenv import load_dotenv
from comet_ml import Experiment
from query import generate_embeddings

# Load environment variables
load_dotenv()
COMET_API_KEY = os.getenv("COMET_API_KEY")

DATA_FILE_PATH = 'data/nyc_food_scrap_drop_off_sites.csv'
SAMPLE_OUTPUT_PATH = "data/data_wrangling_sample.csv"
EMBEDDINGS_OUTPUT_PATH = 'embeddings.csv'


def create_experiment() -> Experiment:
    """Create and return a Comet experiment."""
    return Experiment(api_key=COMET_API_KEY)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file and return a DataFrame."""
    return pd.read_csv(file_path)


def save_sample(df: pd.DataFrame, output_path: str) -> None:
    """Save the first 20 rows of the DataFrame to a CSV file."""
    df.head(20).to_csv(output_path, index=False)


def main():
    df = load_data(DATA_FILE_PATH)
    df['text'] = df['text'].str.replace(r'^\d+,', '', regex=True)
    save_sample(df, SAMPLE_OUTPUT_PATH)

    df = pd.read_csv('data/data_wrangling_sample.csv')
    # Generate embeddings
    df['embedding'] = df.text.apply(lambda x: generate_embeddings(x))
    df.to_csv(EMBEDDINGS_OUTPUT_PATH, index=False)

    # Log dataset to Comet
    experiment = create_experiment()
    experiment.log_dataset_hash(df)


if __name__ == '__main__':
    main()
