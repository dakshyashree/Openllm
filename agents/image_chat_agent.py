import os
from dotenv import load_dotenv # Import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_openai import OpenAI
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory

# Import your image tools (these are already BaseTool instances)
from tools.image_tools import ImageCaptionTool, ObjectDetectionTool

# 1. Instantiate the underlying Image Tools
# These are already instances of BaseTool, ready to be used.
caption_tool = ImageCaptionTool()
detector_tool = ObjectDetectionTool()

# Directly use the instantiated tool objects in the 'tools' list
tools = [
    caption_tool,
    detector_tool,
]

# 2. Set up the LLM
# Retrieve the API key from environment variables (loaded from .env)
openai_api_key = os.getenv("OPENAI_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables or .env file.")

llm = OpenAI(temperature=0, openai_api_key=openai_api_key) # Pass the retrieved API key

# 3. Optional: add conversational memory for follow-ups
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 4. Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    memory=memory,
    verbose=True,
)

# 5. Helper function to run the agent


def run_image_agent(image_path: str, question: str) -> str:
    """
    Given the file path to an image and a user question, this function
    invokes the LangChain agent which will call the caption and detection
    tools as needed and return a natural-language answer.
    """
    prompt = (
        f"The user has uploaded an image at: {image_path}\n"
        f"Question: {question}\n"
        "Use the available tools to analyze the image and answer comprehensively."
    )
    return agent.run(prompt)