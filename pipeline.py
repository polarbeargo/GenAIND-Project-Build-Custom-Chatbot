import openai
import os
import pandas as pd
from dotenv import load_dotenv
from comet_ml import Experiment
import kfp
from kfp import dsl
from custom_query_completion import generate_embeddings

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
COMET_API_KEY = os.getenv("COMET_API_KEY")

# Constants for file paths
DATA_FILE_PATH = 'data/nyc_food_scrap_drop_off_sites.csv'
SAMPLE_OUTPUT_PATH = "data/data_wrangling_sample.csv"
EMBEDDINGS_OUTPUT_PATH = 'embeddings.csv'


def create_experiment():
    return Experiment(api_key=COMET_API_KEY)


def data_wrangling_op():
    def data_wrangling():
        # Load dataset and perform data wrangling
        with open(DATA_FILE_PATH, 'r') as file:
            next(file)  # Skip the header
            lines = [line.strip() for line in file]

        df = pd.DataFrame(lines, columns=["text"])
        df['text'] = df['text'].str.replace(r'^\d+,', '', regex=True)
        df.head(20).to_csv(SAMPLE_OUTPUT_PATH)

        # Generate embeddings
        df = pd.read_csv(SAMPLE_OUTPUT_PATH)
        df['embedding'] = df.text.apply(generate_embeddings)
        df.to_csv(EMBEDDINGS_OUTPUT_PATH, index=False)

        # Log dataset to Comet
        experiment = create_experiment()
        experiment.log_dataset_hash(df)
        return df

    return dsl.ContainerOp(
        name='Data Wrangling',
        image='python:3.8',
        command=['python', '-c'],
        arguments=[data_wrangling]
    )


def custom_query_op(df):
    def custom_query():
        custom_queries_responses = {}
        experiment = create_experiment()
        for query, response in custom_queries_responses.items():
            experiment.log_text(query, metadata={"type": "query"})
            experiment.log_text(response, metadata={"type": "response"})

    return dsl.ContainerOp(
        name='Custom Query',
        image='python:3.8',
        command=['python', '-c'],
        arguments=[custom_query]
    )


def compare_prompts_op():
    def compare_prompts():
        questions_responses = {}
        experiment = create_experiment()
        for question, (basic_response,
                       custom_response) in questions_responses.items():
            experiment.log_text(question, metadata={"type": "question"})
            experiment.log_text(
                basic_response, metadata={
                    "type": "basic_response"})
            experiment.log_text(
                custom_response, metadata={
                    "type": "custom_response"})

    return dsl.ContainerOp(
        name='Compare Prompts',
        image='python:3.8',
        command=['python', '-c'],
        arguments=[compare_prompts]
    )


@dsl.pipeline(
    name='Generative AI Project Pipeline',
    description='A pipeline for the Generative AI project with Comet-ML integration'
)
def generative_ai_pipeline():
    df = data_wrangling_op()
    custom_query_op(df)
    compare_prompts_op()


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(
        generative_ai_pipeline,
        'generative_ai_pipeline.yaml'
    )
