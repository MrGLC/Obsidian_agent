from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional, Literal
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from datetime import datetime


load_dotenv()


TipoOptions = Literal["Resolver duda", "Elaborar un plan de accion", "Realizar una tarea"]
tiempoOptions = Literal["Pasado", "Presente", "Futuro"]
UrgenciaOptions = Literal["baja", "media", "alta", "crítica"]
endgoalOptions = Literal["Desarrollar informacion sobre un tema", "Resolver un problema inmediato"]

class pregunta(BaseModel):
    """Information about a person."""
    tipo: Optional[TipoOptions] = Field(..., description="Que tipo de accion requiere la consulta?")
    urgencia: Optional[UrgenciaOptions] = Field(
        ..., description="Requiere una respuesta inmediata y corta o puede esperar una respuesta mas larga y detallada?"
    )
    tiempo: Optional[tiempoOptions] = Field(..., description="En que tiempo se esta refiriendo, si se consulta informacion pasada o planes para futuro.")
    end_goal: Optional[endgoalOptions] = Field(..., description="Que es lo que se quiere lograr con la consulta?")


# Define the state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    query_intent: pregunta
    VaultFindings: List[Document]
    WebResults: Optional[List[Dict]]
    processed_vault_docs: List[Dict]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

query_analyzer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the query "
            "If you do not know the value of an attribute asked "
            "to extract, return null for the attribute's value.",
        ),
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        MessagesPlaceholder("examples"),  # <-- EXAMPLES!
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        ("human", "{text}"),
    ]
)

import uuid
from typing import Dict, List, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field


class Example(TypedDict):
    """A representation of an example consisting of text input and expected tool calls.

    For extraction, the tool calls are represented as instances of pydantic model.
    """

    input: str  # This is the example text
    tool_calls: List[BaseModel]  # Instances of pydantic model that should be extracted


def tool_example_to_messages(example: Example) -> List[BaseMessage]:
    """Convert an example into a list of messages that can be fed into an LLM.

    This code is an adapter that converts our example to a list of messages
    that can be fed into a chat model.

    The list of messages per example corresponds to:

    1) HumanMessage: contains the content from which content should be extracted.
    2) AIMessage: contains the extracted information from the model
    3) ToolMessage: contains confirmation to the model that the model requested a tool correctly.

    The ToolMessage is required because some of the chat models are hyper-optimized for agents
    rather than for an extraction use case.
    """
    messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
    tool_calls = []
    for tool_call in example["tool_calls"]:
        tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "args": tool_call.model_dump(),
                # The name of the function right now corresponds
                # to the name of the pydantic model
                # This is implicit in the API right now,
                # and will be improved over time.
                "name": tool_call.__class__.__name__,
            },
        )
    messages.append(AIMessage(content="", tool_calls=tool_calls))
    tool_outputs = example.get("tool_outputs") or [
        "You have correctly called this tool."
    ] * len(tool_calls)
    for output, tool_call in zip(tool_outputs, tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    return messages

query_examples = [
    (
        "Quiero aprender sobre matematicas desde donde me habia quedado",
        pregunta(
            tipo="Resolver duda",
            urgencia="baja", 
            tiempo="Presente",
            end_goal="Desarrollar informacion sobre un tema"
        ),
    ),
    (
        "Se puede hacer agua mineral de forma casera?",
        pregunta(
            tipo="Resolver duda",
            urgencia="alta",
            tiempo="Presente", 
            end_goal="Resolver un problema inmediato"
        ),
    ),
    (
        "Necesito que recuerdes esto y lo escribas en una nota nueva",
        pregunta(
            tipo="Realizar una tarea",
            urgencia="alta",
            tiempo="Presente",
            end_goal="Resolver un problema inmediato"
        ),
    ),
    (
        "Quiero que me ayudes a planificar mi proximo viaje a Japon",
        pregunta(
            tipo="Elaborar un plan de accion",
            urgencia="baja",
            tiempo="Futuro",
            end_goal="Desarrollar informacion sobre un tema"
        ),
    ),
    (
        "Am I Cute?",
        pregunta(
            tipo="Resolver duda",
            urgencia="baja",
            tiempo="Presente",
            end_goal="Resolver un problema inmediato"
        ),
    )
]

messages = []

for text, tool_call in query_examples:
    messages.extend(
        tool_example_to_messages({"input": text, "tool_calls": [tool_call]})
    )


query_analyzer = query_analyzer_prompt | llm.with_structured_output(
    schema=pregunta,
    method="function_calling",
    include_raw=False,
)

# Setup vector store connection
# Replace this with your actual vector store setup
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
OBSIDIAN_VAULT_PATH = "/Users/luisgg/tryingAI"  # Update this path
vector_store = Chroma(
    collection_name="obsidian_jarvis",
    embedding_function=embeddings,
    persist_directory="./obsidian_jarvis_db",  # Where to save data locally, remove if not necessary
)

# Define tools
@tool
def search_vault(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
    """Search the Obsidian vault for information relevant to the query."""
    # Search vector store
    results = vector_store.similarity_search_with_score(query, k=3)
    
    # Format results
    findings = []
    for doc, score in results:
        findings.append(f"* Document: {doc.metadata.get('title', 'Untitled')} (Relevance: {score:.2f})")
    
    findings_str = "\n".join(findings) if findings else "No relevant documents found."
    
    # Return message and update state with the actual document objects
    return Command(update={
        "VaultFindings": [doc for doc, _ in results],
        "messages": [ToolMessage(f"Found {len(results)} relevant documents in vault:\n{findings_str}", tool_call_id=tool_call_id)]
    })


@tool
def human_assistance(
    vault_summary: str, web_summary: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human for reviewing findings."""
    human_response = interrupt(
        {
            "question": "Are these findings accurate and complete?",
            "vault_findings": vault_summary,
            "web_findings": web_summary,
        },
    )
    
    # Process human feedback
    if human_response.get("approval", "").lower().startswith("y"):
        response = "Human confirmed findings are accurate."
    else:
        response = f"Human provided feedback: {human_response.get('feedback', 'No specific feedback')}"
    
    return Command(update={
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)]
    })

### where i am right now ###
@tool
def update_vault_document(
    title: str, content: str, tool_call_id: Annotated[str, InjectedToolCallId], existing_uuid: Optional[str] = None
) -> str:
    """Update or create a document in the Obsidian vault.
    
    Args:
        title: The title of the document
        content: The content to write to the document
        tool_call_id: Tool call ID
        existing_uuid: Optional UUID of an existing document to update. If provided, the old document will be deleted from the vector store.
    """
    print(f"\nDEBUG: update_vault_document called with:")
    print(f"  - title: {title}")
    print(f"  - content length: {len(content)} characters")
    print(f"  - existing_uuid: {existing_uuid}")
    
    # Clean title for filename
    import re
    filename = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '-').lower()
    
    # Create file path
    file_path = os.path.join(OBSIDIAN_VAULT_PATH, f"{filename}.md")
    print(f"  - file_path: {file_path}")
    
    # Generate a new UUID if none was provided
    doc_uuid = existing_uuid if existing_uuid else str(uuid.uuid4())
    
    # If we have an existing UUID, delete the old document from the vector store
    if existing_uuid:
        try:
            vector_store.delete(ids=[existing_uuid])
        except Exception as e:
            print(f"Warning: Failed to delete document with UUID {existing_uuid}: {e}")
    
    try:
        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  - Successfully wrote file to: {file_path}")
        
        # Create document directly without using a loader
        from langchain_core.documents import Document
        
        # Create a document with the content and metadata
        document = Document(
            page_content=content,
            metadata={
                'uuid': doc_uuid,
                'title': title,
                'last_modified': datetime.now().isoformat(),
                'source_path': f"{filename}.md"
            }
        )
        
        # Add to vector store with explicit ID
        vector_store.add_documents([document], ids=[doc_uuid])
        print(f"  - Successfully added document to vector store with UUID: {doc_uuid}")
        
        return Command(update={
            "updated_file_path": file_path,
            "document_uuid": doc_uuid,  # Return the UUID for future reference
            "messages": [ToolMessage(f"Document '{title}' has been updated in the vault at {file_path}", tool_call_id=tool_call_id)]
        })
    except Exception as e:
        error_msg = f"Error in update_vault_document: {str(e)}"
        print(f"ERROR: {error_msg}")
        return Command(update={
            "messages": [ToolMessage(error_msg, tool_call_id=tool_call_id)]
        })

@tool
def delete_vault_document(
    uuid: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Delete a document from the Obsidian vault and vector store by UUID.
    
    Args:
        uuid: The UUID of the document to delete
        tool_call_id: Tool call ID
    """
    try:
        # Search for documents with this UUID in metadata
        results = vector_store.similarity_search(
            "", 
            k=10,
            filter={"uuid": uuid}
        )
        
        if not results:
            return Command(update={
                "messages": [ToolMessage(f"No document found with UUID {uuid}", tool_call_id=tool_call_id)]
            })
        
        # Get the document title and file path
        doc = results[0]
        title = doc.metadata.get('title', 'Unknown')
        
        # Get the filename from metadata or construct it
        if 'source_path' in doc.metadata:
            file_path = os.path.join(OBSIDIAN_VAULT_PATH, doc.metadata['source_path'])
        else:
            # Clean title for filename (same as in update_vault_document)
            import re
            filename = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '-').lower()
            file_path = os.path.join(OBSIDIAN_VAULT_PATH, f"{filename}.md")
        
        # Delete from vector store
        vector_store.delete(ids=[uuid])
        
        # Delete the file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
            file_deleted = True
        else:
            file_deleted = False
        
        # Return success message
        if file_deleted:
            return Command(update={
                "messages": [ToolMessage(f"Document '{title}' (UUID: {uuid}) has been deleted from the vault and vector store.", tool_call_id=tool_call_id)]
            })
        else:
            return Command(update={
                "messages": [ToolMessage(f"Document with UUID {uuid} has been deleted from the vector store, but no corresponding file was found in the vault.", tool_call_id=tool_call_id)]
            })
            
    except Exception as e:
        return Command(update={
            "messages": [ToolMessage(f"Error deleting document with UUID {uuid}: {str(e)}", tool_call_id=tool_call_id)]
        })

# Define nodes
def query_analyzer_node(state: State):
    """Analyze the user query to determine intent."""
    # Get the latest user message
    latest_msg = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            latest_msg = msg.content
            break
    
    if not latest_msg:
        return {"messages": [AIMessage(content="I couldn't find a query to analyze.")]}
    
    # Use examples from your training data
    # Convert examples to messages format for the query_analyzer
    examples_messages = []
    for text, tool_call in query_examples:
        examples_messages.extend(
            tool_example_to_messages({"input": text, "tool_calls": [tool_call]})
        )
    
    # Analyze intent using the correctly defined query_analyzer
    intent = query_analyzer.invoke({"text": latest_msg, "examples": examples_messages})
    
    return {
        "query_intent": intent,
        "messages": [AIMessage(content=f"I understand that your query is a {intent.tipo} with {intent.urgencia} urgency.")]
    }


def vault_processor(state: State):
    """Process vault findings to extract relevant information."""
    # Get documents and intent from state
    vault_findings = state.get("VaultFindings", [])
    query_intent = state.get("query_intent", None)
    
    if not vault_findings:
        return {
            "processed_vault_docs": [],
            "messages": [AIMessage(content="No vault documents to process.")]
        }
    
    # Process documents
    processed_docs = []
    for doc in vault_findings:
        # Extract content and metadata, ensuring UUID is preserved
        uuid_value = doc.metadata.get("uuid", "")
        
        processed_docs.append({
            "uuid": uuid_value,
            "title": doc.metadata.get("title", "Untitled"),
            "content": doc.page_content,
            "metadata": doc.metadata,
            "last_modified": doc.metadata.get("last_modified", ""),
            "tags": doc.metadata.get("tags", ""),
            "concepts": doc.metadata.get("concepts", "")
        })
    
    # Prepare a summary for the model
    summary = ""
    if processed_docs:
        summary = "Vault findings summary:\n"
        for i, doc in enumerate(processed_docs):
            summary += f"{i+1}. {doc['title']} (UUID: {doc['uuid'][:8]}...): {doc['content'][:100]}...\n"
    
    return {
        "processed_vault_docs": processed_docs,
        "messages": [AIMessage(content=summary)]
    }


def document_creator(state: State):
    """Create or update documents based on findings and intent."""
    # Get processed docs and intent
    processed_docs = state.get("processed_vault_docs", [])
    query_intent = state.get("query_intent", None)
    web_results = state.get("WebResults", [])
    
    # Get the latest user message for context
    latest_query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            latest_query = msg.content
            break
    
    # Determine if we're updating or creating
    is_updating = False
    primary_doc = None
    
    # Check if we have documents to update and if the intent suggests updating
    if processed_docs and query_intent:
        # Check if the intent is to update an existing document
        if query_intent.tipo in ["Actualizar", "Modificar", "Realizar una tarea"]:
            is_updating = True
            # Use the most relevant document (first in the list) as the primary document to update
            primary_doc = processed_docs[0]
    
    # Format web results if available
    web_content = ""
    if web_results:
        web_content = "\n\n## Web Research Results\n"
        for i, result in enumerate(web_results):
            title = result.get("title", f"Result {i+1}")
            snippet = result.get("snippet", "No snippet available")
            url = result.get("link", "No URL available")
            web_content += f"### {title}\n{snippet}\nSource: {url}\n\n"
    
    # Prepare content based on intent
    if is_updating and primary_doc:
        title = primary_doc["title"]
        content = primary_doc["content"]
        uuid_to_update = primary_doc["uuid"]
        
        # If we have web results, suggest incorporating them
        if web_content:
            content_with_web = f"{content}\n\n{web_content}"
            
            # Add message about updating with UUID reference and web content
            return {
                "document_to_update": {
                    "title": title,
                    "content": content_with_web,
                    "uuid": uuid_to_update,
                    "has_web_content": True
                },
                "messages": [AIMessage(content=f"I'll update the document '{title}' (UUID: {uuid_to_update[:8]}...) with the latest web research results.")]
            }
        else:
            # Add message about updating with UUID reference
            return {
                "document_to_update": {
                    "title": title,
                    "content": content,
                    "uuid": uuid_to_update,
                    "has_web_content": False
                },
                "messages": [AIMessage(content=f"I'll update the document '{title}' (UUID: {uuid_to_update[:8]}...) based on your request.")]
            }
    else:
        # We're creating a new document
        # Create more specific titles for different types of content
        if "dieta" in latest_query.lower() or "alimentación" in latest_query.lower() or "nutrición" in latest_query.lower():
            title = f"Plan de Alimentación: {latest_query[:30]}"
        elif "plan" in latest_query.lower():
            title = f"Plan de Acción: {latest_query[:30]}"
        else:
            title = f"Note on: {latest_query[:50]}"
        
        # If we have web results, include them in the new document
        initial_content = ""
        if web_content:
            initial_content = f"# {title}\n\nCreated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n{web_content}"
            
            return {
                "document_to_create": {
                    "title": title,
                    "initial_content": initial_content,
                    "has_web_content": True
                },
                "messages": [AIMessage(content=f"I'll create a new document titled '{title}' with web research results.")]
            }
        else:
            initial_content = f"# {title}\n\nCreated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            return {
                "document_to_create": {
                    "title": title,
                    "initial_content": initial_content,
                    "has_web_content": False
                },
                "messages": [AIMessage(content=f"I'll create a new document titled '{title}' based on your request.")]
            }


# Web search tool
web_search = TavilySearchResults(max_results=3)
answer_llm = ChatOpenAI(model="gpt-4", temperature=0)
# Combine all tools
tools = [search_vault, human_assistance, update_vault_document, delete_vault_document, web_search]
llm_with_tools = answer_llm.bind_tools(tools)


def chatbot(state: State):
    """Main chatbot node with tool-calling capabilities."""
    print("\nDEBUG: Chatbot node called")
    
    # Check if we have a document to update from the document_creator
    document_to_update = state.get("document_to_update", None)
    document_to_create = state.get("document_to_create", None)
    web_results = state.get("WebResults", [])
    vault_findings = state.get("VaultFindings", [])
    query_intent = state.get("query_intent", None)
    
    print(f"  - document_to_update: {document_to_update is not None}")
    print(f"  - document_to_create: {document_to_create is not None}")
    print(f"  - web_results: {len(web_results) if web_results else 0} results")
    print(f"  - vault_findings: {len(vault_findings) if vault_findings else 0} findings")
    
    # Get the latest user message for context
    latest_query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            latest_query = msg.content
            break
    
    # If we don't have web results yet and we're creating a diet plan, search the web first
    if not web_results and "dieta" in latest_query.lower():
        print("\nDEBUG: Searching web for diet information")
        return Command(update={
            "messages": [
                AIMessage(content="", tool_calls=[{
                    "id": str(uuid.uuid4()),
                    "name": "tavily_search_results_json",
                    "args": {
                        "query": f"1200 calorie diet for irritable bowel syndrome IBS foods to eat and avoid"
                    }
                }])
            ]
        })
    
    # If we have search results but no document creation/update in progress,
    # and the intent is for document creation, create a new document
    if not document_to_update and not document_to_create and (web_results or vault_findings):
        if query_intent and query_intent.tipo in ["Actualizar", "Modificar", "Crear", "Elaborar un plan de accion", "Realizar una tarea"]:
            # Create a new document
            title = ""
            if "dieta" in latest_query.lower() or "alimentación" in latest_query.lower() or "nutrición" in latest_query.lower():
                title = f"Plan de Alimentación: {latest_query[:50]}"
            elif "plan" in latest_query.lower():
                title = f"Plan de Acción: {latest_query[:50]}"
            else:
                title = f"Note on: {latest_query[:50]}"
            
            # Format web findings
            web_content = ""
            if web_results:
                web_content = "\n\n## Web Research Findings\n"
                for result in web_results:
                    web_content += f"### {result.get('title', 'Untitled')}\n"
                    web_content += f"{result.get('content', '')}\n"
                    web_content += f"Source: {result.get('url', '')}\n\n"
            
            # Format vault findings
            vault_content = ""
            if vault_findings:
                vault_content = "\n\n## Related Vault Documents\n"
                for doc in vault_findings:
                    vault_content += f"### {doc.metadata.get('title', 'Untitled')}\n"
                    vault_content += f"{doc.page_content[:500]}...\n\n"
            
            # Create initial content
            initial_content = f"""# {title}

Created on: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Overview
This document was created based on your request for information about {latest_query}

{web_content}
{vault_content}
"""
            
            # Call update_vault_document directly
            print("\nDEBUG: Creating new document")
            print(f"  - Title: {title}")
            print(f"  - Content length: {len(initial_content)} characters")
            
            # Set document_created flag to prevent recursion
            state["document_created"] = True
            
            return Command(update={
                "messages": [
                    AIMessage(content="", tool_calls=[{
                        "id": str(uuid.uuid4()),
                        "name": "update_vault_document",
                        "args": {
                            "title": title,
                            "content": initial_content,
                        }
                    }])
                ]
            })
    
    # If we have a document to update, add a system message with instructions
    if document_to_update:
        has_web_content = document_to_update.get("has_web_content", False)
        web_content_msg = "The document has been updated with web research results." if has_web_content else ""
        
        system_message = SystemMessage(content=f"""
        You are updating an existing document with title: '{document_to_update['title']}' and UUID: {document_to_update['uuid']}.
        When using the update_vault_document tool, make sure to pass this UUID as the existing_uuid parameter.
        {web_content_msg}
        Current content: {document_to_update['content'][:200]}...
        
        IMPORTANT: You MUST use the update_vault_document tool to save the changes to the document.
        """)
        state["messages"].append(system_message)
    
    # If we have a document to create, add a system message with instructions
    elif document_to_create:
        has_web_content = document_to_create.get("has_web_content", False)
        initial_content = document_to_create.get("initial_content", "")
        title = document_to_create.get("title", "")
        
        if "Plan de Alimentación" in title:
            system_message = SystemMessage(content=f"""
            You are creating a new diet plan document with title: '{title}'. 
            
            IMPORTANT: Your next action MUST be to use the update_vault_document tool to create this document.
            Do not respond with any other message or perform any other action first.
            
            Create a comprehensive diet plan based on the user's request. Include:
            1. A brief introduction about the diet's purpose
            2. Daily meal plans with specific foods and portions
            3. Nutritional information (calories, macros)
            4. Special considerations for the user's health conditions
            5. References or sources if applicable
            
            Start with this template:
            {initial_content}
            
            Remember: Use the update_vault_document tool IMMEDIATELY to create the document.
            Do not respond with the content in the chat.
            """)
        else:
            system_message = SystemMessage(content=f"""
            You are creating a new document with title: '{title}'. 
            
            IMPORTANT: Your next action MUST be to use the update_vault_document tool to create this document.
            Do not respond with any other message or perform any other action first.
            
            Start with this template:
            {initial_content}
            
            Remember: Use the update_vault_document tool IMMEDIATELY to create the document.
            Do not respond with the content in the chat.
            """)
        
        state["messages"].append(system_message)
    
    # Invoke the LLM with tools
    print("\nDEBUG: Invoking LLM with tools")
    message = llm_with_tools.invoke(state["messages"])
    
    # Check for tool calls
    if hasattr(message, "tool_calls") and message.tool_calls:
        print(f"  - Number of tool calls: {len(message.tool_calls)}")
        for i, tool_call in enumerate(message.tool_calls):
            print(f"  - Tool call {i+1}:")
            print(f"    - Name: {tool_call.get('name', 'unknown')}")
            print(f"    - Args: {tool_call.get('args', {})}")
    else:
        print("  - No tool calls in message")
    
    # Ensure we don't have too many tool calls
    if hasattr(message, "tool_calls") and len(message.tool_calls) > 1:
        message.tool_calls = message.tool_calls[:1]
    
    return {"messages": [message]}


def route_after_analysis(state: State):
    """Determine next step after query analysis."""
    intent = state.get("query_intent")
    
    if not intent:
        return "chatbot"
    
    # For any intent type, first gather information through the chatbot
    return "chatbot"

def route_after_search(state: State):
    """Determine if we should move to document creation after search."""
    intent = state.get("query_intent")
    vault_findings = state.get("VaultFindings", [])
    web_results = state.get("WebResults", [])
    document_created = state.get("document_created", False)
    
    # If we've already created a document, end the process
    if document_created:
        return END
    
    # If we have search results and the intent is for document creation
    if intent and intent.tipo in ["Actualizar", "Modificar", "Crear", "Elaborar un plan de accion", "Realizar una tarea"]:
        if vault_findings or web_results:
            return "document_creator"
    
    return END

def should_continue_processing(state: State):
    """Determine if we should continue processing or end."""
    # Check if we have completed a document creation/update
    if "updated_file_path" in state:
        return END
    
    # Check if we've hit the document creation phase
    document_to_create = state.get("document_to_create")
    document_to_update = state.get("document_to_update")
    if document_to_create or document_to_update:
        state["document_created"] = True
        return "chatbot"
    
    # Check if we have search results to process
    vault_findings = state.get("VaultFindings", [])
    web_results = state.get("WebResults", [])
    if vault_findings or web_results:
        return route_after_search(state)
    
    return END

def tools_condition(state: State):
    """Determine whether to use tools or continue to vault processor."""
    # If we have tool calls, process them
    latest_message = state["messages"][-1] if state.get("messages") else None
    if latest_message and hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
        return "tools"
    
    # If we've already created a document, end
    if state.get("document_created", False):
        return END
        
    # Otherwise, continue to vault processor
    return "vault_processor"

# Build the graph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("query_analyzer", query_analyzer_node)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("vault_processor", vault_processor)
graph_builder.add_node("document_creator", document_creator)

# Add tool node
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_edge(START, "query_analyzer")
graph_builder.add_conditional_edges(
    "query_analyzer",
    route_after_analysis
)
graph_builder.add_edge("document_creator", "chatbot")

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "vault_processor")
graph_builder.add_conditional_edges(
    "vault_processor",
    should_continue_processing
)

# Compile the graph
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Run the graph
def process_query(user_input):
    """Process a user query through the graph."""
    config = {
        "configurable": {
            "thread_id": "1",
            "recursion_limit": 10  # Set a lower recursion limit
        }
    }
    
    events = graph.stream(
        {
            "messages": [HumanMessage(content=user_input)],
            "document_created": False,  # Initialize state
            "WebResults": [],  # Initialize empty web results
            "VaultFindings": [],  # Initialize empty vault findings
        },
        config,
        stream_mode="values",
    )
    
    # Process and display events
    for event in events:
        if "messages" in event and event["messages"]:
            latest_message = event["messages"][-1]
            print(f"{latest_message.__class__.__name__}: {latest_message.content}")
            
            # Check for tool calls and display them
            if hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
                for tool_call in latest_message.tool_calls:
                    tool_name = tool_call.get("name", "unknown_tool")
                    tool_args = tool_call.get("args", {})
                    print(f"Tool Call: {tool_name}")
                    print(f"Tool Args: {tool_args}")
                    
                    # If it's an update_vault_document call, print additional info
                    if tool_name == "update_vault_document":
                        title = tool_args.get("title", "Untitled")
                        print(f"Creating/Updating document: {title}")
                        
        # Check for other important state updates
        if "updated_file_path" in event:
            print(f"Document updated at: {event['updated_file_path']}")
        if "document_uuid" in event:
            print(f"Document UUID: {event['document_uuid']}")

# Example usage
if __name__ == "__main__":
    process_query("me puedes dar una dieta de 1200 calorias?tengo el colon irritado por lo que alimentos especificos que ayuden serian buenos. no hay prisa tienes tiempo pero asegurate de crear las notas adecuadas")