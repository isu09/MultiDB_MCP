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
    
    schema = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
    return df,schema

def extract_sql_from_llm_response(response_text):
    match = re.search(r"```sql(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1).strip()
        return sql
    else:
        sql = re.sub(r"```", "", response_text).strip()
        return sql

def generate_create_table_query(schema: dict, table_name: str):
    prompt = f"""
You are an expert SQL developer.
I have the following table schema from a file:

{schema}

Please generate a CREATE TABLE query for {db_type} with table name {table_name}.
Quote all table and column names using backticks ` to handle reserved words.
Use appropriate SQL types and proper syntax.
"""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    query = response.choices[0].message.content.strip()
    sql_query = extract_sql_from_llm_response(query)
    return sql_query

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
# MCP TOOL
# -----------------------------

#------CREATE TABLE---------
@mysql_mcp.tool(
    name="create_table",
    description="Create a MySQL table based on file schema and table name."
)
def create_table_tool(file_path: str, table_name: str):
    df,schema = get_file_schema(file_path)
    create_query = generate_create_table_query(schema, table_name)
    message = create_table_in_db(create_query,df,table_name)
    return message


#------INSERT TABLE----------
#@mysql_mcp.tool()
# -----------------------------
# START MCP SERVER
# -----------------------------
if __name__ == "__main__":
    print("Starting Database MCP server...", file=sys.stderr)
    mysql_mcp.run()
