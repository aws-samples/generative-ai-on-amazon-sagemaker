import re

def append_unique_name(prefix: str, suffix: str) -> str:
    return str((prefix + re.sub(r'[^0-9a-z]+', '_', suffix.lower())).strip()[:30])

PATH_TO_SAMPLE_RANKING_CSV = 'assets/search_ranking_sample.csv'
PATH_TO_EVENTS_RANKING_CSV = 'assets/search_ranking_prod.csv'
PATH_TO_SAMPLE_CHATBOT_CSV = 'assets/llm_events.parquet'

PATH_TO_LLM_CHARTS = 'assets/charts_llm.yaml'
PATH_TO_ML_CHARTS = 'assets/charts_ml.yaml'

PROJECT_NAME_PREFIX = 'ai_travel_agent_'

LLM_MODEL_NAME = 'assistant_chatbot'
RANKING_MODEL_NAME = 'search_ranking'
