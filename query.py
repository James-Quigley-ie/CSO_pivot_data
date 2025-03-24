import faiss, pickle
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
from datetime import timedelta
from cso_ireland_data import CSODataSession
from groq import Groq
import re
import numpy as np
import json

# Pandas options
pd.set_option("display.max_colwidth", None)    # no truncation
pd.set_option("display.width", None)    

# Initialize client with API key
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Load metadata and models
table_meta = pd.read_pickle("cso_table.pkl")
CORPUS_DIR = "my_corpus"
INDEX_PATH  = "vector_index.faiss"
META_PATH   = "metadata.pkl"
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"

index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)
model = SentenceTransformer(MODEL_NAME)

cso = CSODataSession(
    cached_session_params={
        "use_cache_dir": True,      # Save files in the default user cache dir
        "cache_control": True,      # Use Cache-Control response headers for expiration, if available
        "expire_after": timedelta(days=1),  # Otherwise expire responses after one day
    }
)

# Global variables (initially set to None)
query = table = table_title = fields = None
table_select_result = ""
clean_aggregate_table_result = ""
llm_title = ""
df = None
supertitle = ""

def ensure_dataframe(obj):
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    if isinstance(obj, (int, float, np.number)):
        return pd.DataFrame([obj], columns=['Value'])
    try:
        return pd.DataFrame(obj)
    except Exception as e:
        raise ValueError(f"Conversion to DataFrame failed for object of type {type(obj)}: {e}")




# Define a simple custom error for our purposes.
class CustomError(Exception):
    def __init__(self, message, context=None):
        super().__init__(message)
        self.context = context

def search_text_only(query: str, k: int = 25):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for idx in indices[0]:
        if idx < len(metadata):
            item = metadata[idx]
            if isinstance(item, dict):
                result_entry = {key: item[key] for key in ['table_name', 'table_id', 'fields'] if key in item}
            else:
                result_entry = {"table_id": item, "table_name": None, "fields": None}
            results.append(result_entry)
    return results

def get_table_info(table_id: str, table_list: list):
    for table in table_list:
        if table.get('table_id') == table_id:
            table_name = table.get('table_name')
            fields = table.get('fields') or ""
            return table_name, fields
    return None, None

def extract_python_code(s):
    match = re.search(r'```python(.*?)```', s, re.DOTALL)
    if match:
        code = match.group(1)
        return code.replace('\n', '')
    return ""

# def ensure_dataframe(obj):
#     if isinstance(obj, pd.DataFrame):
#         return obj
#     if isinstance(obj, pd.Series):
#         return obj.to_frame()
#     if isinstance(obj, (int, float, np.number)):
#         return pd.DataFrame([obj], columns=['Value'])
#     try:
#         df_obj = pd.DataFrame(obj)
#         return df_obj
#     except Exception as e:
#         yield f"Conversion to DataFrame failed for object of type {type(obj)}: {e}"
#         return None


def get_table(exception=None):
    global query, table, table_title, fields, table_select_result
    results = search_text_only(query)
    first_result = results[0]
    yield "\nStep 0: Retrieved 25 matching datasets successfully."
    
    message_content = (
        "You will be given a list of documents in the form of python dictionaries consisting of statistics tables from Ireland's Central Statistics Office. "
        "For each table you will see the \"table_name\", \"table_id\" and the \"fields\" provided in the table. "
        "For example, the first table is " + json.dumps(first_result) +
        ". You must return the table \"table_id\" which most likely contains the relevant information to answer the following user query. "
        "Do not attempt to answer the question and do not ask for follow up information. Only return the table_id. "
        "You must answer with a table ID. If you do not find a matching table, make your best guess. The user query is: " 
        + query + " The documents are " + json.dumps(results) + 
        ". Do not attempt to answer the question and do not ask for follow up information. Only return the table_id. "
        "Return the table_id now."
    )
    
    messages = [{
        "role": "user",
        "content": message_content
    }]
    if exception is not None:
        messages.append({
            "role": "system",
            "content": "The previous attempt resulted in this exception: " + str(exception) + "Try another table_id"
        })
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-specdec",
        temperature=0,
    )
    table_select_result = chat_completion.choices[0].message.content.strip()
    yield "\nStep 1: Selected Table ID: " + table_select_result
    
    try:
        table = cso.get_table(table_select_result)
        yield "Step 2: Retrieved table successfully. Table ID: " + table_select_result
    except Exception as e:
        error_message  = "Step 2 Error: Unable to get the table " + table_select_result
        yield error_message
        raise Exception(error_message) from e

    
    try:
        table_title, fields = get_table_info(table_select_result, results)
        yield ("Step 2: Retrieved table metadata successfully. Table Name: " + str(table_title) +
               " | Table fields: " + str(fields))
    except Exception as e:
        table_title = fields = ""
        yield "\n(partial failure): Was unable to get table Metadata"

def pivot_table(exception=None):
    global table, table_title, fields, table_select_result, clean_aggregate_table_result, df, supertitle
    field_analysis = ""
    if hasattr(table, "index") and table.index.names:
        for i, level_name in enumerate(table.index.names):
            val_counts = table.index.get_level_values(i).value_counts()
            top_index_names = str(list(val_counts.head(7).index))
            field_analysis += (f"The 7 most frequent values for index '{level_name}' are {top_index_names} "
                               f"from a total of {len(val_counts)} unique level values. ")
    
    field_analysis_numeric = ""
    if hasattr(table, "index") and table.index.names:
        for level in table.index.names:
            try:
                aggregated = table.groupby(level=level).sum()
                if isinstance(aggregated, pd.Series):
                    sorted_aggregated = aggregated.sort_values(ascending=False)
                    top_values = list(sorted_aggregated.head(7).index)
                    field_analysis_numeric += (f"The top values for index '{level}' are {top_values} "
                                               f"from a list of {len(sorted_aggregated.index)} values. ")
                else:
                    top_values = list(table.index.get_level_values(level).unique())[:3]
                    field_analysis_numeric += (f"The first three unique values for index '{level}' are {top_values}. ")
            except Exception:
                continue

    message_content = (
        "You will be given a pandas DataFrame from Ireland's Central Statistics Office. "
        "The table title is " + str(table_title) + ". "
        "You must return the one-liner pandas DataFrame aggregation command for the DataFrame called \"table\" which is most likely to answer the user query. "
        "The user query is " + query + " "
        "You might need to perform aggregation and selection across multiple indices and fields to help the user understand. "
        "Do not attempt to answer the query and do not ask for follow up information. "
        "Do not explain your reasoning. "
        "Do not make up fields or MultiIndices. Only aggregate and filter on the provided field and MultiIndices or their most frequent values. "
        "If a MultiIndex or field is present but not relevant, it must be filtered or aggregated appropriately. "
        "For example, if the field 'sex' contains ['Both sexes', 'Female', 'Male'] do not sum over all values as each sex will be counted twice. "
        "If the index 'county' contains the values ['State', 'Dublin', 'Cork'] then the 'State' value clearly represents the total count and must be handled carefully. "
        "When aggregating data that measures a continuous value in time, using mean may be preferred over sum. "
        "If an important key, index, or field is not present, return the most important information for the user's query. "
        "The user query is **" + query + "**. "
        "The table title is " + str(table_title) + ". "
        "The fields in the table are " + str(list(table.columns)) + ". "
        "The DataFrame MultiIndices are " + str(table.index.names) + ". "
        + field_analysis + field_analysis_numeric +
        " Return the data aggregation command now."
    )
    
    messages = [{
        "role": "user",
        "content": message_content
    }]
    if exception is not None:
        err_dict = {
            "role": "user",
            "content": str(exception) + ". Be careful to produce a command now which will not throw an error."
        }
        messages.insert(0, err_dict)
        messages.append(err_dict)

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )
    model_output = chat_completion.choices[0].message.content
    clean_aggregate_table_result = model_output.replace("python", "").replace("`", "").replace("\n", "").strip()
    
    yield "\nStep 3: Pivoting Data with aggregation command: " + clean_aggregate_table_result
    
    supertitle = table_select_result + " " + clean_aggregate_table_result[5:]
    
    local_vars = {"table": table}
    saved_cmd = "temp = " + clean_aggregate_table_result
    try:
        exec(saved_cmd, globals(), local_vars)
        df = ensure_dataframe(local_vars["temp"])
    except Exception as e:
        error_message = "WARNING! The command " + clean_aggregate_table_result + " threw the error: " + str(e)
        yield error_message
        raise Exception(error_message) from e


def get_title(exception=None):
    global table, table_title, clean_aggregate_table_result, llm_title, df
    message_content = (
        "Given a DataFrame from Ireland's Central Statistics Office. "
        "The table title is " + str(table_title) + ". "
        "The fields in the DataFrame are " + str(list(table.columns)) + ". "
        "The DataFrame MultiIndices are " + str(table.index.names) + ". "
        "The DataFrame is then manipulated by the command " + clean_aggregate_table_result + ". "
        "The first row in the DataFrame is " + df.head(1).to_string() + " "
        "The last row in the DataFrame is " + df.tail(1).to_string() + ". "
        "The DataFrame is then plotted using df.plot(kind='bar'). "
        "You must return a short title for the generated plot. "
        "Do not attempt to answer the query and do not ask for follow up information. "
        "Return the title of the plot now."
    )
    
    messages = [{
        "role": "user",
        "content": message_content
    }]
    if exception is not None:
        messages.append({
            "role": "system",
            "content": "Previous exception during title generation: " + str(exception)
        })
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-specdec",
        temperature=0,
    )
    
    llm_title = chat_completion.choices[0].message.content.strip()
    yield "\nStep 4: Chart Title Built: " + llm_title

def call_with_retry(generator_func, max_retries=3):
    """
    Calls the generator function (which accepts an optional keyword 'exception') and yields
    its messages. If an exception is thrown, it is caught and passed into the next call.
    """
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            gen = generator_func(exception=last_exception) if last_exception is not None else generator_func()
            for status in gen:
                yield status
                if status.strip() == "Aborting after maximum attempts.":
                    # Abort immediately if we reached the abort message.
                    return
            return  # Successful completion; exit the retry loop.
        except Exception as e:
            yield f"Attempt {attempt} failed with exception: {e}"
            last_exception = e
    yield "Aborting after maximum attempts."

def AskCSO(user_query: str):
    global query
    query = user_query  # set the global query variable
    
    # Execute each step, checking for the abort message after each.
    for status in call_with_retry(get_table, max_retries=3):
        yield status
        if status.strip() == "Aborting after maximum attempts.":
            return
    for status in call_with_retry(pivot_table, max_retries=3):
        yield status
        if status.strip() == "Aborting after maximum attempts.":
            return
    for status in call_with_retry(get_title, max_retries=3):
        yield status
        if status.strip() == "Aborting after maximum attempts.":
            return
    
    yield df
    yield llm_title
    yield supertitle
    yield table_title
