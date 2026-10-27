from langchain_openai import ChatOpenAI
from config.llm_config import lm_config

_llm_client_cache = {}

def get_llm_client(model: str|None = None,json_mode:bool=False):
    m = model or lm_config.llm_model

    key = (m,json_mode)

    if key in _llm_client_cache:
        return _llm_client_cache[key]

    model_kwargs = {}
    if json_mode:
        model_kwargs["response_format"] = {"type":"json_object"}
    # 返回模型
    client = ChatOpenAI(
        model=m,
        temperature=lm_config.llm_temperature,  # 模型的发散度
        base_url=lm_config.base_url,
        api_key=lm_config.api_key,
        model_kwargs=model_kwargs
    )

    _llm_client_cache[key] = client

    return client
