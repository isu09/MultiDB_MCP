import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import pandas as pd
from fastmcp import FastMCP
import json
import sys
import re
from openai import OpenAI

# Load environment variables
load_dotenv()

snowflake_mcp = FastMCP("SNOWFLAKE_MCP")

# Snowflake connection URL (from env variable)
# Example format:
# snowflake://<user>:<password>@<account>/<database>/<schema>?warehouse=<warehouse>&role=<role>
snowflake_conn = os.getenv("snowflake_url")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
db_type = "snowflake"
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


def generate_sql(query_type: str, table_name: str, schema: dict = None,upsert:bool=False , db_type = "snowflake"):
    """
        Fully LLM-driven SQL generation for Snowflake:
    CREATE, INSERT, SELECT, COUNT, DESCRIBE, MERGE (UPSERT).
    No MySQL-specific syntax will be used.
    """
    rules = [
        'Always quote table and column names with double quotes (").',
        'For INSERT, generate template with placeholders (:colname), no actual data.',
        'For SELECT, return all columns by default.',
        'For CREATE, include IF NOT EXISTS with proper Snowflake types.',
        'For COUNT, generate SELECT COUNT(*) query.',
        'For DESCRIBE, generate query to return column names and data types.',
        'For UPSERT in Snowflake, always generate a MERGE INTO statement.',
        'Do not use MySQL-specific syntax like ON DUPLICATE KEY UPDATE.',
        'Ensure all generated SQL is compatible with Snowflake.'
    ]

    prompt = f"""
    You are an expert Snowflake SQL generator.
    Task: Generate a {query_type.upper()} query.
    Table: {table_name}
    Schema: {schema}

    Rules:
    - {'\n- '.join(rules)}
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return extract_sql_from_llm_response(response.choices[0].message.content)

def execute_query(sql: str, params=None, fetch=False):
    engine = create_engine(snowflake_conn)
    with engine.begin() as conn:
        if fetch:
            result = conn.execute(text(sql))
            return [dict(row._mapping) for row in result]
        else:
            conn.execute(text(sql), params or {})
            return {"status": "success", "query": sql}


def create_table_in_db(create_query: str, df: pd.DataFrame, table_name: str,schema):
    engine = create_engine(snowflake_conn)
    with engine.begin() as conn:  # connection open & auto-commit
        # Create table
        
        conn.execute(text(create_query))
        
        # LLM generates INSERT template
        insert_sql = generate_sql("insert", table_name, schema)
        
        rows_inserted =0
        # Insert rows
        for _, row in df.iterrows():
            conn.execute(text(insert_sql), row.to_dict())
            rows_inserted +=1
    
    return insert_sql, rows_inserted
# -----------------------------
# MCP TOOLS
# -----------------------------
#------get_schema_file---------------
@snowflake_mcp.tool(
    name="get_schema_file",
    description="Get the schema from the file"
)
def get_schema(file_path: str):
    schema = get_file_schema(file_path)
    return schema


#-----create_table-------------
@snowflake_mcp.tool(
    name="create_table",
    description="Generate and execute a CREATE TABLE query from file schema."
)
def create_table(file_path: str, table_name: str , db_type = "snowflake"):
    df, schema = get_file_schema(file_path)
    engine = create_engine(snowflake_conn)
    # Step 1: Check if table exists dynamically using LLM
    describe_sql = generate_sql("describe", table_name)  # LLM-driven "DESCRIBE"
    try:
        with engine.begin() as conn:
            conn.execute(text(describe_sql))
        # If execution succeeds, table exists
        return {
            "status": "exists",
            "message": f"Table '{table_name}' already exists. No table created or rows inserted."
        }
    except:
        # Table does not exist → proceed
        pass
    create_query = generate_sql("create", table_name, schema,db_type)
    insert_query, rows_inserted = create_table_in_db(create_query,df,table_name,schema)
    result =  {
        "status": "success",
        "create_query": create_query.strip().split("\n"),
        "insert_query": insert_query.strip().split("\n"),
        "rows_inserted": rows_inserted,
        "message": f"Table '{table_name}' created and {rows_inserted} rows inserted successfully!"
    }
    return json.dumps(result, indent=4)


#------insert_data------------
@snowflake_mcp.tool(
    name="insert_data",
    description="Generate an INSERT query template (no values) and insert new rows."
)
def insert_data(file_path: str, table_name: str,db_type = "snowflake"):
    df, schema = get_file_schema(file_path)
    
    # LLM generates INSERT template
    insert_sql = generate_sql("insert", table_name, schema,db_type)
    
    # LLM generates SELECT to check existing rows
    select_sql = generate_sql("select", table_name)
    
    engine = create_engine(snowflake_conn)
    with engine.begin() as conn:
        rows_inserted = 0
        existing_df = pd.read_sql(select_sql, conn)
        new_rows = df[~df.apply(tuple, axis=1).isin(existing_df.apply(tuple, axis=1))]
        
        if new_rows.empty:
            return {"status": "exists", "message": f"No new rows to insert into '{table_name}'."}
        
        # Execute INSERT template with real data
        for _, row in new_rows.iterrows():
            conn.execute(text(insert_sql), row.to_dict())
            rows_inserted += 1
        
        result= {
            "status": "updated",
            "message": f"Inserted {rows_inserted} new rows into '{table_name}'.",
            "insert_query": insert_sql.strip(),
            "rows_inserted": rows_inserted
        }
        return json.dumps(result, indent=4)

#---------select_data------
@snowflake_mcp.tool(
    name="select_data",
    description="Generate and execute a SELECT query to return all rows."
)
def select_data(table_name: str):
    # Generate schema dynamically using LLM
    # You can ask LLM to give column names/types for the table
    schema_sql = generate_sql("describe", table_name)  # LLM-driven "DESCRIBE" equivalent
    engine = create_engine(snowflake_conn)
    with engine.begin() as conn:
        schema_result = conn.execute(text(schema_sql))
        schema = [dict(row._mapping) for row in schema_result]  # List of dicts with column info

    # Generate SELECT query dynamically using LLM
    select_sql = generate_sql("select", table_name, schema={col["name"]: col["type"] for col in schema})
    sample_rows = execute_query(select_sql, fetch=True)  # execute_query will limit to 10 rows

    # Generate COUNT query dynamically using LLM
    count_sql = generate_sql("count", table_name)
    with engine.begin() as conn:
        total_rows = conn.execute(text(count_sql)).scalar()

    return {
        "status": "success",
        "select query": select_sql,
        "count query" : count_sql,
        "total_rows": total_rows,
        "returned_rows": len(sample_rows),
        "data": sample_rows
    }

#-----upsert_data------------
@snowflake_mcp.tool(
    name="upsert_data",
    description="Generate an UPSERT (INSERT ... ON DUPLICATE KEY UPDATE) SQL template from a file schema. No actual data inserted."
)
def upsert_data(file_path: str, table_name: str ):
    # Step 1: Read file and get schema
    df, schema = get_file_schema(file_path)
    
    # Step 2: Generate upsert query template using LLM
    upsert_sql_template = generate_sql("merge", table_name, schema, db_type="snowflake")
    
   # Step 3: Connect to database and execute UPSERT for each row
    engine = create_engine(snowflake_conn)  # or snowflake connection string
    rows_inserted= 0 
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(text(upsert_sql_template), row.to_dict())
            rows_inserted +=1
    
    result = {
        "status": "success",
        "upsert_sql_template": upsert_sql_template,
        "rows inserted " : rows_inserted,
        "message": f"All rows from '{file_path}' upserted into table '{table_name}' successfully!"
    }
    return json.dumps(result, indent=4)
# -----------------------------
# START MCP SERVER
# -----------------------------
if __name__ == "__main__":
    print("Starting Database MCP server...", file=sys.stderr)
    snowflake_mcp.run()