import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from langchain.chat_models import init_chat_model

# Load environment variables from .env file
load_dotenv()

model = init_chat_model("openai:gpt-4.1")

async def main():
    client = MultiServerMCPClient(
        {
            "server": {
                "command": "python",
                # Make sure to update to the full absolute path to your math_server.py file
                "args": ["./server.py"],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()

    def call_model(state: MessagesState):
        response = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_node(ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    builder.add_edge("tools", "call_model")
    graph = builder.compile()
    dice_response = await graph.ainvoke({"messages": "roll a 6 sided die once"})
    upc_response = await graph.ainvoke({"messages": "Lookup upc: 028400596008"})
    print("Dice response: ", dice_response["messages"][-1].content)
    print("UPC response: ", upc_response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())