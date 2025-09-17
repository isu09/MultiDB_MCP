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


async def init_snowflake():
   
    # ---- MCP client setup ----
    client = MultiServerMCPClient(
        {
            "database": {
                "command": "python",
                "args": ["snowflake_mcp.py"],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
# ----------------------------- Create React Agent -----------------------------
    snowflake_agent = create_react_agent(name= "snowflake_agent",model=model, 
                               tools=[t for t in tools if t.name in ["get_schema_file","create_table","insert_data","select_data","upsert_data"]],
                               prompt="""You are SNOWFLAKE_AGENT, an expert assistant for managing SNOWFLAKE databases via MCP tools.
Your responsibilities are:

1. Communicate clearly and professionally with the user in natural language.
2. Understand the user's request and determine the appropriate database action.
3. Always use the available MCP tools to perform database operations:
   -If the table does not exist create_table(file_path, table_name): Create a table from a CSV, Excel, or JSON file. else say that "If a table already exists or data already exists", inform the user politely.
   - insert_table(file_path, table_name): Insert new rows from a file into an existing table, skipping duplicates.
   - select_table(table_name): Retrieve all rows from an existing table.
4. Collect all necessary information from the user query, such as:
   - file_path (for CSV/Excel/JSON files)
   - table_name (for target SNOWFLAKE table)
5. If you are unsure which tool to use, ask the user for clarification.
6. Never write or execute SQL yourself; always call the appropriate MCP tool.
7. Summarize the result of tool execution in user-friendly language.


Examples:

User Query: "Create a table from C:\\Data\\stocks.csv named stocks2025"
Response: Call `create_table` with file_path="C:\\Data\\stocks.csv" and table_name="stocks2025"

User Query: "Insert the data from C:\\Data\\stocks.csv into stocks2025"
Response: Call `insert_table` with file_path="C:\\Data\\stocks.csv" and table_name="stocks2025"

User Query: "Show me all rows from stocks2025"
Response: Call `select_table` with table_name="stocks2025""")
    # ----------------------------- CLI Loop -----------------------------
   
    # --- Simple CLI Loop ---
    chat_history = []
    while True:
        query = input("\nYour query: ")
        if query.lower() in ["exit", "quit"]:
            break

        chat_history.append(("user", query))

        result = await snowflake_agent.ainvoke({"messages": chat_history})

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
    asyncio.run(init_snowflake())