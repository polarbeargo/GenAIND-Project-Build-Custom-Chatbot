import openai
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import Union
import numpy as np
from comet_llm import Span, end_chain, start_chain
from dotenv import load_dotenv
import os

load_dotenv('my_config.env')
COMET_API_KEY = os.getenv("COMET_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
prompt = """
Property 1 : Food Scrap Drop-off Site at South Beach

Neighborhood : Grasmere-Arrochar-South Beach-Dongan Hills
Location : 21 Robin Road, Staten Island NY
Hosted By : Snug Harbor Youth
Open Months : Year Round
Operation Hours : Friday (Start Time: 1:30 PM - End Time: 4:30 PM)
Website : snug-harbor.org(opens in a new tab)
Coordinates : Latitude 40.595579, Longitude -74.062991
Notes : This site accepts all food scraps. Please compost responsibly.
Property 2 : Food Scrap Drop-off Site at Inwood

Neighborhood : Inwood
Location : SE Corner of Broadway & Academy Street
Hosted By : Department of Sanitation
Open Months : Year Round
Operation Hours : 24/7
Website : www.nyc.gov/smartcomposting(opens in a new tab)
Coordinates : Not specified
Notes : Download the app to access bins. Accepts all food scraps, including meat and dairy. Do not leave food scraps outside of bin!
Property 3 : Food Scrap Drop-off Site at Old Stone House Brooklyn

Neighborhood : Park Slope
Location : 336 3rd St, Brooklyn, NY 11215
Hosted By : Old Stone House Brooklyn
Open Months : Year Round
Operation Hours : 24/7
Website : Not specified
Coordinates : Latitude 40.6727118, Longitude -73.984731
Notes : This site accepts all food scraps. Please compost responsibly.
Property 4 : Food Scrap Drop-off Site at East Harlem

Neighborhood : East Harlem (North)
Location : SE Corner of Pleasant Avenue & E 116 Street
Hosted By : Department of Sanitation
Open Months : Year Round
Operation Hours : 24/7
Website : www.nyc.gov/smartcomposting(opens in a new tab)
Coordinates : Not specified
Notes : Download the app to access bins. Accepts all food scraps, including meat and dairy. Do not leave food scraps outside of bin!
Property 5 : Food Scrap Drop-off Site at Malcolm X FSDO

Neighborhood : Corona
Location : 111-26 Northern Blvd, Flushing, NY 11368
Hosted By : NYC Compost Project Hosted by Big Reuse
Open Months : Year Round
Operation Hours : Tuesdays (Start Time: 12:00 PM - End Time: 2:00 PM)
Website : Not specified
Coordinates : Latitude 40.7496855, Longitude -73.8630721
Notes : This site accepts all food scraps. Please compost responsibly.
Property 6 : Food Scrap Drop-off Site at Astoria Pug

Neighborhood : Astoria (North)-Ditmars-Steinway
Location : Ditmars Boulevard and 41st Street
Hosted By : Astoria Pug
Open Months : Year Round
Operation Hours : Mondays (Start Time: 8:00 AM - End Time: 2:00 PM)
Website : Instagram(opens in a new tab)
Coordinates : Latitude 40.7724122, Longitude -73.9053388
Notes : Not accepted: meat, bones, or dairy. Please compost responsibly.
Property 7 : Food Scrap Drop-off Site at Norwood

Neighborhood : Norwood
Location : SE Corner of Kings College Place & Gun Hill Rd.
Hosted By : Department of Sanitation
Open Months : Year Round
Operation Hours : 24/7
Website : www.nyc.gov/smartcomposting(opens in a new tab)
Coordinates : Not specified
Notes : Download the app to access bins. Accepts all food scraps, including meat and dairy. Do not leave food scraps outside of bin!
Property 8 : Food Scrap Drop-off Site at Bedford-Stuyvesant (East)

Neighborhood : Bedford-Stuyvesant (East)
Location : NW Corner of Malcolm X Boulevard & Bainbridge Street
Hosted By : Department of Sanitation
Open Months : Year Round
Operation Hours : 24/7
Website : www.nyc.gov/smartcomposting(opens in a new tab)
Coordinates : Not specified
Notes : Download the app to access bins. Accepts all food scraps, including meat and dairy. Do not leave food scraps outside of bin!
"""

questions = [
    "What is the food scrap drop-off site in Brooklyn?",
    "Where can I drop off food scraps in Manhattan?",
    "Are there any food scrap drop-off sites in Queens?",
    "What are the hours of operation for food scrap drop-off sites in the Bronx?",
    "Can I drop off food scraps in Staten Island?",
    "Is there a food scrap drop-off site near me?",
]


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


def get_only_response(response):
    messages = [
        {
            "role": "system",
            "content": "Your task is to extract only the response to the user in the following full chatbot response: {response}".format(
                response=response)}]

    return get_completion(messages)


def chain_of_thoughts(questions):
    chatbot_responses = []
    for question in questions:
        messages = [
            {
                "role": "system",
                "content": "Your task is to answer questions factually about a nyc food scrap drop off sites, provided below and delimited by +++++. The user request is provided here: {request}\n\nStep 1: The first step is to check if the user is asking a question related to any type of food scrap drop off sites (even if that food scrap drop off sites is not on the list). If the question is about any type of food scrap drop off sites, we move on to Step 2 and ignore the rest of Step 1. If the question is not about food scrap drop off sites, then you send a response: \"Sorry! I cannot help with that. Please let me know if you have a question about our food scrap drop off sites.\"\n\nStep 2: In this step, you check that the user question is relevant to any of the items on the food scrap drop off sites. You should check that the food scrap drop off site exists in the food scrap drop off sites. If it doesn't exist then send a kind response to the user that the item doesn't exist in the exsisting food scrap drop off sites and then include a list of available but similar food scrap drop off sites without any other details (e.g., location). The food scrap drop off sites available are provided below and delimited by +++++: {Location}+++++\n\nStep 3: If the item exists in the food scrap drop off sites and the user is requesting specific information, provide that relevant information to the user using the food scrap drop off sites. Make sure to use a friendly tone and keep the response concise.\n\nPerform the following reasoning steps to send a response to the user:\nStep 1: <Step 1 reasoning>\nStep 2: <Step 2 reasoning>\nResponse to the user (only output the final response): <response to user>".format(
                    request=question,
                    food_scrap_drop_off_sites=prompt,
                    Location=prompt)}]

        response = get_completion(messages)
        chatbot_responses.append(response)

        start_chain(
            inputs={"question": question},
            api_key=COMET_API_KEY,
        )

        with Span(
            category="reasoning",
            name="chain-of-thought",
            inputs={"user_question": question},
        ) as span:
            span.set_outputs(outputs={"full_response": response})

        with Span(
            category="response-extraction",
            inputs={
                "user_question": question,
                "full_response": response,
            },
        ) as span:
            final_response = get_only_response(response)
            span.set_outputs(outputs={"final_response": final_response})

        end_chain(outputs={"final_response": final_response})
