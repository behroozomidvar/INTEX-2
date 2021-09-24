# INTEX: Interactive Text-based Exploration

Exploratory Data Analysis (EDA) is a hot research topic in database research. Existing work has focused on providing guidance to users to help them refine their needs and find items of interest in large volumes of structured data. Proposed approaches are attribute-based and focus on structured content. In this paper, we focus on providing guidance for exploring databases that contain a mix of structured (attribute-based) and unstructured (text-based) data. We refer to that as Text-based Data Exploration (TDE) and develop **INTEX**, a framework for guided TDE. **INTEX** formalizes text dimensions such as sentiment and topics, and defines new text-based operators that are seamlessly integrated with traditional EDA operators. To provide guidance, **INTEX** develops a Deep Reinforcement Learning (DRL) approach to train policies that provide exploration guidance. To do that, **INTEX** formalizes multi-rewards that capture different textual dimensions, and extends the Deep Q-Networks (DQN) architecture with multi-objective optimization.

## Requirements: Python dependencies
Install the required packages with 

    pip install -r requirements.txt

## Requirements: The databases

### Amazon dataset

The Postgres SQL dump for the electronic retail dataset can be found here: https://www.dropbox.com/s/j04z1gx4tt51q20/intex_amazon.backup?dl=0. 

The database must be called `amazon_reviews` .

### IMDb dataset

The dataset for the IMDb movies can be found here: https://www.dropbox.com/s/7n0v4drk351p872/intex_imdb.backup?dl=0.

The database must be called `imdb_reviews`.

## Offline process: Learn text-based exploration policies

The script `train_deep_agent.py` employs DRL using the the PyTorch-based [PFRL library](https://github.com/pfnet/pfrl) to learn text-based exploration policies. It depcreated two previously developped libraries, `agent` and `environment`.

The **INTEX** environment is also modeled using the [Gym library](https://gym.openai.com). The necessary files for the environment is in the folder `intex_env`. To work with the library, it has to be first installed: `pip install -e .` (run inside the `intex_env` folder).

## Online process: Execute the text-based exploration GUI
To execute the GUI of **INTEX**, first execute `start-server.py`, and then open `localhost/8080` on a browser.

The code provides 3 different handlers for creating/loading an exploration session, starting an exploration, and proceeding the exploration process.
