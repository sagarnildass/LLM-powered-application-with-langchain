import os
from dotenv import load_dotenv, find_dotenv
from os.path import join, dirname
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

load_dotenv(find_dotenv())

if __name__ == "__main__":
    # print("Hello, Langchain")
    linkedin_profile_url = linkedin_lookup_agent(name="ashley amakoh")
    summary_template = """
        given the information {information} about a person I want you to create:
        1. A short summary
        2. Two interesting facts about them
        """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-4")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    res = chain.invoke(input={"information": linkedin_data})
    # print(res.keys())

    print(res["text"])
