import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from fastmcp import FastMCP
import sys
import json
from motor.motor_asyncio import AsyncIOMotorClient
import re
from typing import Optional
mongodb_mcp = FastMCP("MONGODB_MCP",instructions="""
        MongoDB MCP Server
        Tools:
        
        - create_collection
        - list_collection
        - upsert_document (insert new documents only)
        - get_document (fetch documents)
    """)

load_dotenv()

conn = os.getenv("mongo_url")
client = AsyncIOMotorClient(conn)
mongodb_conn = client["stocks"] 

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Set your API key here



#----------get_schema-----------
@mongodb_mcp.tool(
    name="get_schema",
    description="""
    Get the schema from a file.

    Parameters:
    - file_path: Path to uploaded file.

    Returns:
    - dict of column_name: dtype for the file.
    """
    )
async def get_schema(file_path: str):
    ext = file_path.split('.')[-1].lower()
    
    if ext == "json":
        df = pd.read_json(file_path)
    elif ext == "csv":
        df = pd.read_csv(file_path)
    elif ext =="tsv":
        df = pd.read_csv(file_path , sep = '\t')
    else:
        raise ValueError("Unsupported file type")

    df.columns = df.columns.str.strip()

    df.columns = df.columns.str.replace(" ", "_")

    schema = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
    
    return schema



# ---------------- Tools ----------------

# Create a new collection
@mongodb_mcp.tool(name="create_collection", description="Create a collection in MongoDB")
async def create_collection(collection_name: str, file_path: Optional[str] = None):
    try:
        df = pd.DataFrame()  # empty by default

        # 1️⃣ Read file into DataFrame if provided
        if file_path:
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file_path.endswith((".xls", ".xlsx")):
                df = pd.read_excel(file_path)
            elif file_path.endswith(".json"):
                df = pd.read_json(file_path)
            else:
                return {"error": "Unsupported file type. Use CSV, XLSX, or JSON."}

        # 2️⃣ Check if collection exists
        existing_collections = await mongodb_conn.list_collection_names()
        if collection_name in existing_collections:
            return {"message": f"Collection '{collection_name}' already exists."}

        # 3️⃣ Create collection
        collection = await mongodb_conn.create_collection(collection_name)

        # 4️⃣ Insert data if file was provided and has rows
        if not df.empty:
            await collection.insert_many(df.to_dict(orient="records"))

        return {"message": f"Collection '{collection_name}' created successfully with {len(df)} documents."}

    except Exception as e:
        return {"error": str(e)}
# List all collections
@mongodb_mcp.tool(name="list_collection", description="List all collections in MongoDB")
async def list_collection():
    try:
        collections = await mongodb_conn.list_collection_names()
        if not collections:
            return {"message": "No collections found in MongoDB."}
        return {"collections": collections}
    except Exception as e:
        return {"error": str(e)}

@mongodb_mcp.tool(
    name="upsert_document",
    description="Insert or update documents from JSON/CSV/TSV/Excel files into MongoDB (auto-detect unique keys)"
)
async def upsert_document(collection_name: str, file_path: str, unique_keys: list = None):
    """
    collection_name: MongoDB collection to insert/update into
    file_path: Path to CSV/TSV/Excel/JSON file
    unique_keys: Optional list of column names to identify existing documents
    """
    try:
        collection = mongodb_conn[collection_name]

        # ---------- Read file ----------
        ext = file_path.split('.')[-1].lower()
        if ext == "json":
            df = pd.read_json(file_path)
        elif ext == "csv":
            df = pd.read_csv(file_path)
        elif ext == "tsv":
            df = pd.read_csv(file_path, sep='\t')
        elif ext in ["xls", "xlsx"]:
            df = pd.read_excel(file_path)
        else:
            return {"error": "Unsupported file type."}

        df.columns = df.columns.str.strip().str.replace(" ", "_")
        documents = df.to_dict(orient="records")

        if not documents:
            return {"message": "No documents to insert."}

        # ---------- Determine unique keys ----------
        if not unique_keys or not all(k in df.columns for k in unique_keys):
            # Default: first column as unique key
            unique_keys = [df.columns[0]]

        # ---------- Upsert documents ----------
        inserted = 0
        updated = 0
        for doc in documents:
            filter_query = {k: doc[k] for k in unique_keys if k in doc}
            if not filter_query:
                # fallback: use entire document as filter
                filter_query = doc

            result = await collection.update_one(
                filter_query,
                {"$set": doc},
                upsert=True
            )
            if result.matched_count:
                updated += 1
            else:
                inserted += 1

        return {
            "message": f"{inserted} new document(s) inserted, {updated} document(s) updated in '{collection_name}'."
        }

    except Exception as e:
        return {"error": str(e)}

# Fetch documents from a collection
@mongodb_mcp.tool(name="get_document", description="Fetch documents from MongoDB collection")
async def get_document(collection_name: str, limit: int = 10):
    try:
        if collection_name not in await mongodb_conn.list_collection_names():
            return {"error": f"Collection '{collection_name}' does not exist."}

        collection = mongodb_conn[collection_name]
        cursor = collection.find().limit(int(limit))

        documents = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            documents.append(doc)

        if not documents:
            return {"message": f"No documents found in '{collection_name}'."}

        return {"collection": collection_name, "documents": documents}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print("Starting Database MCP server...", file=sys.stderr)
    mongodb_mcp.run()