from comet_ml import Experiment
import os
import pandas as pd
from dotenv import load_dotenv
from query import get_completion, custom_prompt

# Load environment variables
load_dotenv('my_config.env')
COMET_API_KEY = os.getenv("COMET_API_KEY")
EMBEDDINGS_OUTPUT_PATH = 'embeddings.csv'


def create_experiment() -> Experiment:
    """Create and return a Comet experiment."""
    return Experiment(api_key=COMET_API_KEY)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file and return a DataFrame."""
    return pd.read_csv(file_path)


def log_query_response(query: str, response: str) -> None:
    """Log the query and response to Comet."""
    experiment = create_experiment()
    experiment.log_text(query, metadata={"type": "query"})
    experiment.log_text(response, metadata={"type": "response"})


def main():
    df = load_data(EMBEDDINGS_OUTPUT_PATH)
    df['embedding'] = df['embedding'].apply(
        lambda x: [float(val) for val in x.strip('[]').split()])
    question = "What is the food scrap drop-off site in Brooklyn?"
    response = get_completion(custom_prompt(question, df))
    log_query_response(question, response)


if __name__ == '__main__':
    main()
