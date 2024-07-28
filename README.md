# Project-Build-Custom-Chatbot
[image1]: ./images/chainOfThought.png

## Project Overview
- The chatbot is trained on the GPT-3 model using the OpenAI API.
- The chatbot is able to answer questions related to food scrap drop-off sites in New York City and can provide information about the location, address, city, state, ZIP code, latitude, longitude, borough, days, hours, food scraps accepted, location, start date, end date, website, and notes of the drop-off sites.
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

## Demo

- Data Wrangling  
  - Results: [embeddings.csv](embeddings.csv)
- Custom Query Completion and Performance Demostation
    - Results: [project.ipynb](project.ipynb)
- Chain of thought

    - Results:
[https://www.comet.com/polarbeargo/llm-general/prompts](https://www.comet.com/polarbeargo/llm-general/prompts)

![Chain of thought][image1]
