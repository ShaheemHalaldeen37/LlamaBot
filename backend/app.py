from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.messages import messages_to_dict
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logging

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PAGE_HTML = os.path.join(_PROJECT_ROOT, "page.html")
import time
import json
from datetime import datetime
from agents.react_agent.nodes import build_workflow
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import CheckpointTuple
from psycopg_pool import ConnectionPool
from run_logger import RunLogger
from llm_provider import build_llm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chat_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories
app.mount("/assets", StaticFiles(directory="../assets"), name="assets")
app.mount("/examples", StaticFiles(directory="../examples"), name="examples")

# Initialize the LLM provider chain (gpt-oss → gemini fallback)
llm = build_llm()

# Pydantic model for chat request
class ChatMessage(BaseModel):
    message: str
    thread_id: str = None  # Optional thread_id parameter
    agent: str = None  # Optional agent parameter

# Application state to hold persistent checkpointer, important for session-based persistence.
app.state.checkpointer = None

def get_or_create_checkpointer():
    """Get persistent checkpointer, creating once if needed"""
    if app.state.checkpointer is None:
        db_uri = os.getenv("DB_URI")
        if db_uri:
            try:
                # Create connection pool and PostgresSaver directly
                pool = ConnectionPool(db_uri)
                app.state.checkpointer = PostgresSaver(pool)
                app.state.checkpointer.setup()
                logger.info("Using PostgreSQL persistence.")
            except Exception as e:
                logger.warning(f"Failed to connect to PostgreSQL: {e}. Using MemorySaver.")
                app.state.checkpointer = MemorySaver()
        else:
            logger.info("No DB_URI found. Using MemorySaver for session-based persistence.")
            app.state.checkpointer = MemorySaver()
    
    return app.state.checkpointer

@app.get("/", response_class=HTMLResponse)
async def root():
    # Serve the home.html file
    with open("home.html") as f:
        return f.read()

@app.post("/chat-message")
async def chat_message(chat_message: ChatMessage):
    request_id = f"req_{int(time.time())}_{hash(chat_message.message)%1000}"
    logger.info(f"[{request_id}] New chat message received: {chat_message.message[:50]}...")
    
    # Get the existing HTML content from page.html
    try:
        with open(_PAGE_HTML, "r") as f:
            existing_html_content = f.read()
    except FileNotFoundError:
        existing_html_content = ""

    # Define a generator function to stream the response
    async def response_generator():
        thread_id = chat_message.thread_id or "5"
        run_log = RunLogger(request_id, chat_message.message, thread_id)
        run_log.log_existing_html(existing_html_content)

        try:
            logger.info(f"[{request_id}] Starting streaming response")
            yield json.dumps({"type": "start", "request_id": request_id}) + "\n"

            logger.info(f"[{request_id}] Using thread_id: {thread_id}")

            checkpointer = get_or_create_checkpointer()
            graph = build_workflow(checkpointer=checkpointer)
            stream = graph.stream(
                {
                    "messages": [HumanMessage(content=chat_message.message)],
                    "initial_user_message": chat_message.message,
                    "existing_html_content": existing_html_content,
                },
                config={"configurable": {"thread_id": thread_id, "request_id": request_id}},
                stream_mode=["updates", "messages"],
            )

            final_state = None

            for chunk in stream:
                if chunk is None:
                    continue

                is_llm_message = isinstance(chunk, tuple) and len(chunk) == 2 and chunk[0] == "messages"
                is_update = isinstance(chunk, tuple) and len(chunk) == 2 and chunk[0] == "updates"

                if is_llm_message:
                    message_from_llm = chunk[1][0]
                    langgraph_node_info = chunk[1][1]
                    node_name = langgraph_node_info["langgraph_node"]

                    logger.info(f"[{request_id}] token from {node_name}: {str(message_from_llm.content)[:80]}")

                    yield json.dumps({
                        "type": "update",
                        "node": node_name,
                        "value": str(message_from_llm.content),
                    }) + "\n"

                elif is_update:
                    updated_state = chunk[1]

                    for node_name, node_output in updated_state.items():
                        messages = node_output.get("messages", []) if isinstance(node_output, dict) else []

                        for msg in messages:
                            msg_type = type(msg).__name__

                            if msg_type == "AIMessage":
                                # LLM finished a full response — log its output
                                run_log.log_llm_output(msg)

                            elif msg_type == "ToolMessage":
                                # Tool finished — log the result
                                run_log.log_tool_result(
                                    tool_name=getattr(msg, "name", "unknown_tool"),
                                    result=str(msg.content),
                                )

                        logger.info(f"[{request_id}] update from {node_name}: {str(node_output)[:100]}")

                    if "messages" in (list(updated_state.values()) or [{}])[0] if updated_state else False:
                        final_state = updated_state
                    # Track final state from any node that has messages
                    for node_output in updated_state.values():
                        if isinstance(node_output, dict) and "messages" in node_output:
                            final_state = node_output
                            break

                else:
                    logger.info(f"[{request_id}] unknown chunk format: {type(chunk)}")

        except Exception as e:
            err_str = str(e)
            logger.error(f"[{request_id}] Error in stream: {err_str}", exc_info=True)

            if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                run_log.log_gemini_error(err_str)
            else:
                run_log.log_general_error(err_str)

            yield json.dumps({"type": "error", "error": err_str, "request_id": request_id}) + "\n"

        finally:
            run_log.log_completion()
            run_log.close()
            logger.info(f"[{request_id}] Stream completed. Log → {run_log.log_path}")
            serializable_messages = messages_to_dict(
                final_state.get("messages", []) if final_state else []
            )
            yield json.dumps({
                "type": "final",
                "node": "final",
                "value": "final",
                "messages": serializable_messages,
            }) + "\n"

    # Return a streaming response
    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream"
    )

@app.get("/chat", response_class=HTMLResponse)
async def chat():
    with open("chat.html") as f:
        return f.read()

@app.get("/page", response_class=HTMLResponse)
async def page():
    with open(_PAGE_HTML) as f:
        return f.read()
    
@app.get("/conversations", response_class=HTMLResponse)
async def conversations():
    with open("conversations.html") as f:
        return f.read()

@app.get("/threads", response_class=JSONResponse)
async def threads():
    checkpointer = get_or_create_checkpointer()
    config = {}
    checkpoint_generator = checkpointer.list(config=config)
    all_checkpoints :list[CheckpointTuple] = list(checkpoint_generator) #convert to list
    
    # reduce only to the unique thread_ids  
    unique_thread_ids = list(set([checkpoint[0]["configurable"]["thread_id"] for checkpoint in all_checkpoints]))
    state_history = []
    for thread_id in unique_thread_ids:
        graph = build_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        state_history.append({"thread_id": thread_id, "state": graph.get_state(config=config)})
    return state_history

@app.get("/chat-history/{thread_id}")
async def chat_history(thread_id: str):
    checkpointer = get_or_create_checkpointer()
    graph = build_workflow(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    state_history = graph.get_state(config=config)
    print(state_history)
    return state_history

@app.get("/available-agents", response_class=JSONResponse)
async def available_agents():
    # map from langgraph.json to a list of agent names
    with open("../langgraph.json", "r") as f:
        langgraph_json = json.load(f)
    return {"agents": list(langgraph_json["graphs"].keys())}