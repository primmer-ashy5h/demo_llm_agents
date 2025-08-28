
from pyspark.sql import functions as F, Window
from langchain_core.tools import tool
from typing import List
import os
from pyspark.sql import SparkSession
from dotenv import load_dotenv


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

@tool
def generate_base_aggregations(
        agg_func: str, 
        cols_to_agg: List[str]=None, 
        group_by_cols: List[str]=None,
        round_dec_points: int = 2,
    ) -> str:
    """
    Generate pyspark code for based aggregations on a dataframe using df.transform() construct.
    Args:
        agg_func (str): Aggregation function to apply (e.g., "sum", "avg").
        cols_to_agg (List[str]): List of columns to aggregate.
        group_by_cols (List[str]): List of columns to group by.
        round_dec_points (int): Number of decimal points to round the result.
    """
    return (
        f'.transform(base_aggregations,'
        f'agg_func="{agg_func}", '
        f'cols_to_agg={cols_to_agg}, '
        f'group_by_cols={group_by_cols})'
        f'round_dec_points={round_dec_points})'
    )

@tool
def generate_rolling_window_code(
    partition_cols: List[str],
    order_col: str,
    agg_func: str,
    num_cols_to_agg: List[str],
    window_size_in_days: List[int],
    window_offset: int = 0,
    round_dec_points: int = 2
) -> str:
    """
    Generate PySpark code for rolling window aggregations using df.transform() construct.
    Args:
        partition_cols (List[str]): List of columns to partition by.
        order_col (str): Column to order by within each partition.
        agg_func (str): Aggregation function to apply (e.g., "sum", "avg").
        num_cols_to_agg (List[str]): List of numerical columns to apply aggregations to.
        window_size_in_days (int): List of window sized (in days) to apply the aggregation (inclusive).
        window_offset (int): Lower window row offset (inclusive).
        round_dec_points (int): Number of decimal points to round the result.
    
    """
    return (
        f'.transform(rolling_window_aggregations,'
        f'agg_func="{agg_func}",'
        f'partition_cols={partition_cols},'
        f'order_col="{order_col}",'
        f'num_cols_to_agg={num_cols_to_agg},'
        f'window_size_in_days={window_size_in_days},'
        f'window_offset={window_offset})'
        f'round_dec_points={round_dec_points}'
    )

# --- load api key  ---

# Load variables from .env into environment
load_dotenv()

# Retrieve the key
LLM_API_KEY = os.getenv("GOOGLE_API_KEY")

print("Loaded API Key:", LLM_API_KEY is not None)


# --- LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=LLM_API_KEY,
    temperature=0
)
# --- Tools ---
tools = [generate_base_aggregations, generate_rolling_window_code]


# --- Prompt ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """
               You are a feature engineering code assistant.
               You generate PySpark pipeline code using .transform(function, param-value, ...) constructs
                based on the provided data schema and the user feaure requests.
               You do not generate any other code.
               Return only valid PySpark code snippets that can be directly used to run the pipeline like below:
                
                df_output = df.transform(
                                apply_base_aggregations,
                                agg_func='sum',
                                cols_to_agg=['sales'],
                                group_by_cols=['region'])
                                round_dec_points=2
                            )
                            .transform(
                                apply_rolling_window,
                                agg_func='sum',
                                partition_cols=['region'],
                                order_col='date',
                                num_cols_to_agg=['sales'],
                                window_size_in_days=[2, 3],
                                window_offset=0,
                                round_dec_points=2
                        )
                Do return the output code as in string format with parameters in new lines as shown above in the example.
                Do not return any other text, comments, or explanations.   
     """),
    ("human", "Schem:\n {schema}\n\n Feature request:\n {feature_request}"),
    ("placeholder", "{agent_scratchpad}")
])

# --- Agent + Executor ---
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
                    agent=agent, 
                    tools=tools, 
                    verbose=True, 
                    return_intermediate_steps=True, 
                    handle_parsing_errors=True,
                    max_iterations=3,
                    max_execution_time=60,
                    early_stopping_method="generate")


# # --- Example Run ---
# schema = """
# root
#     |-- cutomer_id: string (nullable = true)
#     |-- edi_business_date: date (nullable = true)
#     |-- event_cnt: double (nullable = true)
#     |-- event_type_login_cnt: double (nullable = true)
#     |-- event_type_payment_cnt: double (nullable = true)
# """

# feature_request = """
# create below features for given data:
# 1. Average event count per customer.
# 2. Rolling 7-day and 2-day sum, average and count of event_type_login_cnt and event_type_payment_cnt per customer.
# """

# result = agent_executor.invoke({
#     "schema": schema,
#     "feature_request": feature_request
# })


# ----------------------------
# Streamlit UI
# ----------------------------
import streamlit as st 
st.set_page_config(page_title="PySpark Feature Pipeline Generator", layout="wide")

st.title("🔧 Rapido Feature Pipeline Generator")
st.markdown("Generate feature pipelines using Rapido based on data schema and feature requests from the user.")

schema = st.text_area("Enter Your DataFrame Schema", height=150, placeholder="root\n |-- region: string\n |-- date: date\n |-- sales: double")
feature_request = st.text_area("Enter Your Feature Requests", height=100, placeholder="Compute 7-day rolling avg of sales per region and total sales per region")

if st.button("Generate Feature Pipeline"):
    if not schema.strip() or not feature_request.strip():
        st.error("Please enter both schema and feature request.")
    else:
        with st.spinner("Generating pipeline..."):
            try:
                result = agent_executor.invoke({"schema": schema, "feature_request": feature_request})
                st.success("Pipeline generated successfully!")
                st.code(result["output"], language="python")
            except Exception as e:
                st.error(f"Error: {e}")