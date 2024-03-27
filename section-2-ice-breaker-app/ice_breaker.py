from dotenv import load_dotenv, find_dotenv
from os.path import join, dirname
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from output_parsers import (
    summary_parser,
    topics_of_interest_parser,
    ice_breaker_parser,
    Summary,
    IceBreaker,
    TopicOfInterest,
)
from chains.custom_chains import (
    get_summary_chain,
    get_interests_chain,
    get_ice_breaker_chain,
)
from typing import Tuple

load_dotenv(find_dotenv())


def ice_break_with(name: str) -> Tuple[Summary, IceBreaker, TopicOfInterest, str]:

    linkedin_profile_url = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    summary_chain = get_summary_chain()
    summary_and_facts = summary_chain.invoke({"information":linkedin_data})
    summary_and_facts = summary_parser.parse(summary_and_facts["text"])

    interests_chain = get_interests_chain()
    interests = interests_chain.invoke({"information":linkedin_data})
    interests = topics_of_interest_parser.parse(interests["text"])

    ice_breaker_chain = get_ice_breaker_chain()
    ice_breakers = ice_breaker_chain.invoke({"information":linkedin_data})
    ice_breakers = ice_breaker_parser.parse(ice_breakers["text"])

    return (
        summary_and_facts,
        interests,
        ice_breakers,
        linkedin_data.get("profile_pic_url"),
    )


if __name__ == "__main__":
    pass
    # print("Hello, Langchain")
    # result, profile_pic_url = ice_break_with(name="Sagarnil Das")
    # print(result)
    # print(profile_pic_url)
    # print(person_intel_parser.parse(result))
    # print(scrape_user_tweets("@elonmusk", num_tweets=5))
