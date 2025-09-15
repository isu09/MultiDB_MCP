import os
import pandas as pd
from sqlalchemy import create_engine,text
from dotenv import load_dotenv
from openai import OpenAI
import re
from fastmcp import FastMCP
import sys

postgres_mcp = FastMCP("POSTGRES_MCP")

load_dotenv()
db_type ="postgres"

# -----------------------------
# CONFIG
# -----------------------------
postgres_conn = os.getenv("postgres_url")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Set your API key here

# -----------------------------
# HELPER FUNCTIONS (from your code)
# -----------------------------
def get_file_schema(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == "csv":
        df = pd.read_csv(file_path)
    elif ext in ["xls", "xlsx"]:
        df = pd.read_excel(file_path)
    elif ext == "json":
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file type")
    
    # Clean column names: strip whitespace
    df.columns = df.columns.str.strip()
    # Optionally, replace spaces with underscores
    df.columns = df.columns.str.replace(" ", "_")

    schema = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
    return df, schema

def extract_sql_from_llm_response(response_text):
    match = re.search(r"```sql(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1).strip()
        return sql
    else:
        sql = re.sub(r"```", "", response_text).strip()
        return sql


def generate_sql(query_type: str, table_name: str, schema: dict = None,upsert:bool=False):
    """
    Fully LLM-driven SQL generation: CREATE, INSERT, SELECT.
    No SQL is hardcoded in Python.
    
    LLM-driven SQL generation: CREATE, INSERT, SELECT, optionally UPSERT.
    No SQL is hardcoded in Python.

    Parameters:
    - query_type: "create", "insert", or "select"
    - table_name: target table name
    - schema: dict of column names and types (required for create/insert)
    - upsert: if True, generate INSERT ... ON DUPLICATE KEY UPDATE for Postgres
    """
    # Build rules dynamically
    rules = [
        "Quote all column names using double quotes to handle reserved words",
        "For INSERT, generate a template with placeholders (like :colname), no actual data.",
        "For SELECT, return all columns by default.",
        "For CREATE, include IF NOT EXISTS and use best-fit Postgres datatypes.",
        "If possible, generate INSERT ... ON DUPLICATE KEY UPDATE to handle existing rows."
        "For UPSERT (if upsert=True), generate INSERT ... ON DUPLICATE KEY UPDATE with placeholders, no actual data values."
    ]
    rules = [r for r in rules if r]  # Remove None values

    # Build the prompt
    prompt = f"""
    You are an expert SQL generator for {db_type}.
    Task: Generate a {query_type.upper()} query.
    Table: {table_name}
    Schema: {schema}

    Rules:
    - {'\n- '.join(rules)}
    """
    # """
    # prompt = f"""
    # You are an expert SQL generator for {db_type}.
    # Task: Generate a {query_type.upper()} query.
    # Table: {table_name}
    # Schema: {schema}

    # Rules:
    # - Always Quote table/column names with backticks.
    # - For INSERT, generate a template with placeholders (like :colname), no actual data.
    # - For SELECT, return all columns by default.
    # - For CREATE, include IF NOT EXISTS and use best-fit Postgres datatypes.
    # """
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return extract_sql_from_llm_response(response.choices[0].message.content)

def execute_query(sql: str, params=None, fetch=False):
    engine = create_engine(postgres_conn)
    with engine.begin() as conn:
        if fetch:
            result = conn.execute(text(sql))
            return [dict(row._mapping) for row in result]
        else:
            conn.execute(text(sql), params or {})
            return {"status": "success", "query": sql}


def create_table_in_db(create_query: str, df: pd.DataFrame, table_name: str,schema):
    engine = create_engine(postgres_conn)
    with engine.begin() as conn:  # connection open & auto-commit
        # Create table
        
        conn.execute(text(create_query))
        

    
    # LLM generates INSERT template
        insert_sql = generate_sql("insert", table_name, schema)
        
        
        # Insert rows
        for _, row in df.iterrows():
            conn.execute(text(insert_sql), row.to_dict())
    
    return {"status": "success", "message": f"Table '{table_name}' created and data inserted successfully!"}
# -----------------------------
# MCP TOOLS
# -----------------------------
#------get_schema_file---------------
@postgres_mcp.tool(
    name="get_schema_file",
    description="Get the schema from the file"
)
def get_schema(file_path: str):
    df,schema = get_file_schema(file_path)
    return schema


#-----create_table-------------
@postgres_mcp.tool(
    name="create_table",
    description="Generate and execute a CREATE TABLE query from file schema."
)
def create_table(file_path: str, table_name: str):
    df, schema = get_file_schema(file_path)
    create_query = generate_sql("create", table_name, schema)
    message = create_table_in_db(create_query,df,table_name,schema)
    return message,schema


#------insert_data------------
@postgres_mcp.tool(
    name="insert_data",
    description="Generate an INSERT query template (no values) and insert new rows."
)
def insert_data(file_path: str, table_name: str):
    df, schema = get_file_schema(file_path)
    
    # LLM generates INSERT template
    insert_sql = generate_sql("insert", table_name, schema)
    
    # LLM generates SELECT to check existing rows
    select_sql = generate_sql("select", table_name)
    
    engine = create_engine(postgres_conn)
    with engine.begin() as conn:
        existing_df = pd.read_sql(select_sql, conn)
        new_rows = df[~df.apply(tuple, axis=1).isin(existing_df.apply(tuple, axis=1))]
        
        if new_rows.empty:
            return {"status": "exists", "message": f"No new rows to insert into '{table_name}'."}
        
        # Execute INSERT template with real data
        for _, row in new_rows.iterrows():
            conn.execute(text(insert_sql), row.to_dict())
        
        return {"status": "updated", "message": f"Inserted {len(new_rows)} new rows into '{table_name}'."}

#---------select_data------
@postgres_mcp.tool(
    name="select_data",
    description="Generate and execute a SELECT query to return all rows."
)
def select_data(table_name: str):
    select_sql = generate_sql("select", table_name)
    return execute_query(select_sql, fetch=True)


#------generate_upsert_query------------
@postgres_mcp.tool(
    name="upsert_data",
    description="Generate an UPSERT (INSERT ... ON DUPLICATE KEY UPDATE) SQL template from a file schema. No actual data inserted."
)
def upsert_data(file_path: str, table_name: str):
    # Step 1: Read file and get schema
    df, schema = get_file_schema(file_path)
    
    # Step 2: Generate upsert query template using LLM
    upsert_sql_template = generate_sql("insert", table_name, schema, upsert=True)
    
    # Step 3: Connect to database and execute UPSERT for each row
    engine = create_engine(postgres_conn)  # or MySQL connection string
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(text(upsert_sql_template), row.to_dict())
    
    return {
        "status": "success",
        "message": f"All rows from '{file_path}' upserted into table '{table_name}' successfully!"
    }
# -----------------------------
# START MCP SERVER
# -----------------------------
if __name__ == "__main__":
    print("Starting Database MCP server...", file=sys.stderr)
    postgres_mcp.run()