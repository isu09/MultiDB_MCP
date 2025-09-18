import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_mcp_adapters.client import MultiServerMCPClient
import json

# # --- Load environment variables ---
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# --- Setup LLM (OpenAI GPT model) ---
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


async def init_mongodb():
    
    # ---- MCP client setup ----
    client = MultiServerMCPClient(
        {
            "database": {
                "command": "python",
                "args": ["mongodb_mcp.py"],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
# ----------------------------- Create React Agent -----------------------------
    mongodb_agent = create_react_agent(name= "mongodb_agent",model=model, 
                               tools=[t for t in tools if t.name in ["create_collection","list_collection","upsert_document","get_document"]],
                               prompt = """
You are a MongoDB assistant agent. Your job is to help the user interact with a MongoDB database
using the available tools. Only use the tools listed below to answer user queries, and do not
make assumptions beyond the database contents.

Available tools:

1. get_schema(file_path)
   - Use this to get the schema of a file (JSON, CSV, TSV, Excel) provided by the user.
   - Input: Path to the file.
   - Output: Dictionary of column_name: data_type.

2. create_collection(collection_name)
   - Use this to create a new collection in MongoDB.
   - Input: Name of the collection.

3. list_collection()
   - Use this to list all collections in the current MongoDB database.
   - Input: None.
   - Output: List of collection names.

4. upsert_document(collection_name, file_path)
   - Use this to insert data from a file into a MongoDB collection.
   - It will only insert new documents; existing documents are not modified.
   - Input: Collection name and path to file.

5. get_document(collection_name, limit=10)
   - Use this to fetch documents from a MongoDB collection.
   - Input: Collection name and optional limit (default 10).

Rules:

- Always answer using the tools when possible. 
- If a query requires file data, always ask the user to provide the file path.
- For queries about schema, collections, or documents, use the appropriate tool.
- Do not attempt to access or modify data directly; always go through the tools.
- Keep your responses clear and concise.
- If unsure about which tool to use, ask clarifying questions first.

Example interactions:

User: "Show me all collections in the database."
AI: "I will use the `list_collection` tool to get all collections."

User: "Create a new collection called 'users'."
AI: "I will use the `create_collection` tool with collection_name='users'."

User: "Insert new data from 'users.json' into the 'users' collection."
AI: "I will use the `upsert_document` tool with collection_name='users' and file_path='users.json'."

User: "Get the first 5 documents from 'users'."
AI: "I will use the `get_document` tool with collection_name='users' and limit=5."

Now respond to the user's queries using the tools appropriately.


""")
    # ----------------------------- CLI Loop -----------------------------
   
    # --- Simple CLI Loop ---
    chat_history = []
    while True:
        query = input("\nYour query: ")
        if query.lower() in ["exit", "quit"]:
            break

        chat_history.append(("user", query))

        result = await mongodb_agent.ainvoke({"messages": chat_history})

        # # Debug trace
        # print("\n=== Conversation Trace ===")
        tool_outputs = [msg for msg in result["messages"] if msg.type == "tool"]
        if tool_outputs:
            print("\n=== Tool Output (DB Response) ===")
            for t in tool_outputs:
                print(t.content)

        # Final AI message
        final_ai = [msg for msg in result["messages"] if msg.type == "ai"]
        if final_ai:
            response = final_ai[-1].content
            print("\n=== Final Agent Reply ===")
            print(response)

        chat_history.append(("ai", response))

if __name__ == "__main__":
    asyncio.run(init_mongodb())