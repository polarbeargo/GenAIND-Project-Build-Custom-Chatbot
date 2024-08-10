from comet_ml import Experiment
import os
import pandas as pd
from dotenv import load_dotenv
from query import get_completion, custom_prompt, simple_prompt, questions, chain_of_thoughts

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


def main():
    df = load_data(EMBEDDINGS_OUTPUT_PATH)
    df['embedding'] = df['embedding'].apply(
        lambda x: [float(val) for val in x.strip('[]').split()])

    experiment = create_experiment()
    for question in questions:
        custom_response = get_completion(custom_prompt(question, df))
        basic_response = get_completion(simple_prompt(question))
        chain_of_thoughts(question)
        experiment.log_text(question, metadata={"type": "question"})
        experiment.log_text(
            basic_response, metadata={
                "type": "basic_response"})
        experiment.log_text(
            custom_response, metadata={
                "type": "custom_response"})


if __name__ == '__main__':
    main()
