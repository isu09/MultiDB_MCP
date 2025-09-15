import os
import pandas as pd
from sqlalchemy import create_engine,text
from dotenv import load_dotenv
from openai import OpenAI
import re
from fastmcp import FastMCP
import sys

mysql_mcp = FastMCP("MYSQL_MCP")

load_dotenv()
db_type ="mysql"

# -----------------------------
# CONFIG
# -----------------------------
mysql_conn = os.getenv("mysql_url")
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


def generate_sql(query_type: str, table_name: str, schema: dict = None):
    """
    Fully LLM-driven SQL generation: CREATE, INSERT, SELECT.
    No SQL is hardcoded in Python.
    """
    prompt = f"""
    You are an expert SQL generator for {db_type}.
    Task: Generate a {query_type.upper()} query.
    Table: {table_name}
    Schema: {schema}

    Rules:
    - Always Quote table/column names with backticks.
    - For INSERT, generate a template with placeholders (like :colname), no actual data.
    - For SELECT, return all columns by default.
    - For CREATE, include IF NOT EXISTS and use best-fit MySQL datatypes.
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return extract_sql_from_llm_response(response.choices[0].message.content)

def execute_query(sql: str, params=None, fetch=False):
    engine = create_engine(mysql_conn)
    with engine.begin() as conn:
        if fetch:
            result = conn.execute(text(sql))
            return [dict(row._mapping) for row in result]
        else:
            conn.execute(text(sql), params or {})
            return {"status": "success", "query": sql}


def create_table_in_db(create_query: str, df: pd.DataFrame, table_name: str):
    engine = create_engine(mysql_conn)
    with engine.begin() as conn:  # connection open & auto-commit
        # Create table
        
        conn.execute(text(create_query))
        
        # Insert data row by row
        for _, row in df.iterrows():
            columns_formatted = ', '.join(f'`{col}`' for col in df.columns)
            placeholders = ', '.join(f':{col}' for col in df.columns)
            insert_sql = f'INSERT INTO `{table_name}` ({columns_formatted}) VALUES ({placeholders})'
            conn.execute(text(insert_sql), row.to_dict())

    return {"status": "success", "message": f"Table '{table_name}' created and data inserted successfully!"}
# -----------------------------
# MCP TOOLS
# -----------------------------
#-----create_table-------------
@mysql_mcp.tool(
    name="create_table",
    description="Generate and execute a CREATE TABLE query from file schema."
)
def create_table_tool(file_path: str, table_name: str):
    df, schema = get_file_schema(file_path)
    create_query = generate_sql("create", table_name, schema)
    message = create_table_in_db(create_query,df,table_name)
    return message


#------insert_data------------
@mysql_mcp.tool(
    name="insert_data",
    description="Generate an INSERT query template (no values) and insert new rows."
)
def insert_data_tool(file_path: str, table_name: str):
    df, schema = get_file_schema(file_path)
    
    # LLM generates INSERT template
    insert_sql = generate_sql("insert", table_name, schema)
    
    # LLM generates SELECT to check existing rows
    select_sql = generate_sql("select", table_name)
    
    engine = create_engine(mysql_conn)
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
@mysql_mcp.tool(
    name="select_data",
    description="Generate and execute a SELECT query to return all rows."
)
def select_data_tool(table_name: str):
    select_sql = generate_sql("select", table_name)
    return execute_query(select_sql, fetch=True)

# -----------------------------
# START MCP SERVER
# -----------------------------
if __name__ == "__main__":
    print("Starting Database MCP server...", file=sys.stderr)
    mysql_mcp.run()
