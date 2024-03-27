import os
import requests
from linkedin_scraper import Person, actions
from selenium import webdriver


def scrape_linkedin_profile(linkedin_profile_url: str):
    """
    Scrapes a linkedin profile for information
    Manually scrape the information from the LinkedIn Profile
    """
    # Commenting out actual scraping code
    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    headers = {"Authorization": "Bearer " + os.environ["PROXYCURL_API_KEY"]}

    params = {
        "url": linkedin_profile_url,
    }
    response = requests.get(api_endpoint, params=params, headers=headers)

    # For github gist, we will return a dummy response
    # response = requests.get(
    #     "https://gist.githubusercontent.com/sagarnildass/3a7cd4d167af9eae1007565241cfc072/raw/204d13b43d78bf71dfbf8b4d1e4849c0112c7a9b/sagarnil-das-linkedin.json"
    # )
    data = response.json()

    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
        and k not in ["people_also_viewed", "certifications"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data
