import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

@dataclass
class MineryConfig:
    base_url:str
    api_token:str

mineru_config = MineryConfig(
    base_url=os.getenv("MINERU_BASE_URL",""),
    api_token=os.getenv("MINERU_API_TOKEN","")
)
