# Project-Build-Custom-Chatbot

[image1]: ./images/chainOfThought.png

## Project Overview

- The chatbot is trained on the GPT-3 model using the OpenAI API which is able to answer questions related to food scrap drop-off sites in New York City and can provide information about the location, address, city, state, ZIP code, latitude, longitude, borough, days, hours, food scraps accepted, location, start date, end date, website, and notes of the drop-off sites.

## Dataset

- `nyc_food_scrap_drop_off_sites.csv` - The dataset contains the location of food scrap drop-off sites in New York City. The dataset contains the following columns:
  - `site_name` - The name of the drop-off site.
  - `address` - The address of the drop-off site.
  - `city` - The city where the drop-off site is located.
  - `state` - The state where the drop-off site is located.
  - `zip` - The ZIP code of the drop-off site.
  - `latitude` - The latitude of the drop-off site.
  - `longitude` - The longitude of the drop-off site.
  - `borough` - The borough where the drop-off site is located.
  - `days` - The days of the week when the drop-off site is open.
  - `hours` - The hours when the drop-off site is open.
  - `food_scrap_materials_accepted` - The type of food scraps that are accepted at the drop-off site.
  - `location` - The location of the drop-off site.
  - `start_date` - The date when the drop-off site started accepting food scraps.
  - `end_date` - The date when the drop-off site stopped accepting food scraps.
  - `website` - The website of the drop-off site.
  - `notes` - Additional notes about the drop-off site.

## Developing A Pipeline

 Integrating data processing, machine learning, and experiment tracking using Comet ML in `project.ipynb` and into `pipline.py`.

#### Benefits of Using a Pipeline

- `Modularity`: Each function has a specific purpose, making the code easier to read, maintain, and debug.
- `Experiment Tracking`: By integrating Comet ML, we can track experiments, log metrics, and visualize results, which is crucial for iterative development in AI.
- `Scalability`: Using Kubeflow Pipelines allows us to scale our operations and manage complex workflows efficiently.

### 1. Libraries

The code imports several libraries:

- `openai`: For interacting with OpenAI's API.
- `os` and `dotenv`: For managing environment variables securely.
- `pandas`: For data manipulation and analysis.
- `comet_ml`: For logging experiments and tracking metrics.
- `kfp` and `dsl`: For defining and managing Kubeflow Pipelines.

### 2. Constants for File Paths

The `pipline.py` defines constants for file paths to make it easier to manage and reference data files throughout the script.

### 3. Functions

- `create_experiment()`: Initializes a new Comet experiment for tracking.

- `load_data(file_path)`: Loads a CSV file into a Pandas DataFrame. This is essential for data wrangling.

- `save_sample(df, output_path)`: Saves the first 20 rows of the DataFrame to a CSV file, which can be useful for quick inspections or debugging.

- `log_query_response(query, response)`: Logs queries and their responses to Comet, which is useful for tracking how well our model is performing.

- `data_wrangling_op()`: This function defines a data wrangling operation. It loads the data, processes it (e.g., cleaning text), generates embeddings, and logs the dataset to Comet. The embeddings are crucial for many AI applications, as they convert text into numerical representations that models can understand.

- `custom_query_op()`: Defines a custom query operation that asks a specific question about the data and logs the response. By using `cosine_similarity` we can effectively compare the similarity between the embedding of each text in the DataFrame (emb) and the embedding of the input question (embeddings_array).

- `compare_prompts_op()`: This function is intended to compare responses from different prompts such as from `the chain of thoughts`, allowing us to evaluate the effectiveness of various queries.

### 4. Pipeline Definition

`generative_ai_pipeline()`: This is the main function that ties everything together. It defines the sequence of operations in the pipeline, ensuring that data wrangling occurs before custom queries and comparisons.

### 5. Pipeline Compilation

The last part of the code compiles the pipeline into a YAML file, which can be deployed in a Kubeflow environment.

## Project Setup

### 1. Create a new virtual environment

```bash
python3 -m venv env
```

### 2. Activate the virtual environment

```bash
source env/bin/activate
```

### 3. Install the required packages (can operate in the jupyter notebook code cell)

```bash
pip install -r requirements.txt
```

- Once the packages are installed, we can run the Jupyter notebook.

### 4. Deploy it using Kubernetes

- Create a Kubernetes cluster using Minikube on MacOS

```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-darwin-amd64
sudo install minikube-darwin-amd64 /usr/local/bin/minikube
minikube start
minikube dashboard
```

- Generate the `generative_ai_pipeline.yaml` file

```bash
python pipline.py
```

- Upload the `generative_ai_pipeline.yaml` file to the Kubeflow Pipelines dashboard. 

- Build and push your own custom docker image to docker hub. 

- After uploading, we should see the pipeline listed in the dashboard.
Click on the pipeline name to open it, and then click on the "Create Run" button.

## Demo

- Data Wrangling  
  - Results: [embeddings.csv](embeddings.csv)
- Custom Query Completion and Performance Demostation
  - Results: [project.ipynb](project.ipynb)
- Chain of thought

  - [project.ipynb](project.ipynb)

  - Results:
[https://www.comet.com/polarbeargo/llm-general/prompts](https://www.comet.com/polarbeargo/llm-general/prompts)

![Chain of thought][image1]

- Run on Kubernetes:
  - [pipeline.py](pipeline.py)
  - [generative_ai_pipeline.yaml](generative_ai_pipeline.yaml)  
  - Compare Prompts:  
  [![YouTube Video](https://img.youtube.com/vi/1-3un7hQVWY/0.jpg)](https://youtu.be/1-3un7hQVWY)


