import openai
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import Union
import numpy as np


MODEL_NAME = 'paraphrase-MiniLM-L6-v2'


def generate_embeddings(input_data: Union[str, list[str]]) -> np.ndarray:
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(input_data)
    return embeddings


def get_completion(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=100,
    )
    return response.choices[0].message["content"]


def simple_prompt(question):
    return [
        {"role": "user", "content": question}
    ]


def custom_prompt(question, df):
    return [{"role": "system",
             "content": """You are a helpful assistant that provides information about food scrap drop-off sites. Answer the question base on context below. Context:
                {}
            """.format('\n\n'.join(custom_query(question, df)))}, {"role": "user", "content": question}]


def custom_query(question, df):
    embeddings_array = generate_embeddings([question])
    df_copy = df.copy()
    df_copy["similarity"] = df_copy["embedding"].apply(
        lambda emb: cosine_similarity([emb], embeddings_array))
    df_copy.sort_values("similarity", ascending=True, inplace=True)
    return df_copy.iloc[:5]['text'].tolist()
