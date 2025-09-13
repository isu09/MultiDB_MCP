import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

# # --- Load environment variables ---
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# --- Setup LLM (OpenAI GPT model) ---
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

async def main():
    # ---- MCP client setup ----
    client = MultiServerMCPClient(
        {
            "database": {
                "command": "python",
                "args": ["postgres_mcp.py"],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
# ----------------------------- Create React Agent -----------------------------
    agent = create_react_agent(name= "postgres_agent",model=model, 
                               tools=[t for t in tools if t.name in ["create_table"]],
                               prompt="""
You are A PostgreSQL_AGENT """
)
    # ----------------------------- CLI Loop -----------------------------
   
    # --- Simple CLI Loop ---
    chat_history = []
    while True:
        query = input("\nYour query: ")
        if query.lower() in ["exit", "quit"]:
            break

        chat_history.append(("user", query))

        result = await agent.ainvoke({"messages": chat_history})

        # # Debug trace
        # print("\n=== Conversation Trace ===")
        # for msg in result["messages"]:
        #     print(f"{msg.type.upper()}: {msg.content}")

        # Tool outputs
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
    asyncio.run(main())


