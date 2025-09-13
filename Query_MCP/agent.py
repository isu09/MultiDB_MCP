import os
import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

# ---- Load environment variables ----
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ---- Setup LLM ----
model = ChatGroq(api_key=os.getenv("GROQ_API_KEY"))

async def main():
    # ---- MCP client pointing to your MySQL MCP server ----
    client = MultiServerMCPClient(
        {
            "mysql_db": {
                "command": "python",
                "args": ["mysql_agent.py"],  # your MCP agent file
                "transport": "stdio",
            }
        }
    )

    # Get tools from MCP server
    tools = await client.get_tools()

    # Now you can call the tool
    query = "Show me the first 5 rows from employees"
    result = await tools["ask_mysql"].ainvoke(query)

    print("\n=== Tool Output ===")
    print(result)

# ---- Run async main ----
if __name__ == "__main__":
    asyncio.run(main())
