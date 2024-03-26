from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    template = """
                Given the full name {name_of_person} I want you to get me the link to their linkedin profile page.
                Your answer should only contain the URL of the linkedin profile page.   
                """
    tools_for_agent = [Tool(name="Crawl google for linkedin profile page", func="?", description="Useful for when you need to find a linkedin profile page")]

    return "Linkedin Profile URL"