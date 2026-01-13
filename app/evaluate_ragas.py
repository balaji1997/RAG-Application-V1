import os
from dotenv import load_dotenv
from typing import List

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_utilization

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


def run_ragas_evaluation(
    query: str,
    answer: str,
    retrieved_contexts: List[str],
):
    """
    RAGAS evaluation using OpenAI as judge.
    Gemini is NOT used here.
    """

    dataset = Dataset.from_dict(
        {
            "question": [query],
            "answer": [answer],
            "contexts": [retrieved_contexts],
        }
    )

    # -------------------------------
    # OpenAI Judge LLM
    # -------------------------------
    judge_llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # -------------------------------
    # Embeddings
    # -------------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/paraphrase-MiniLM-L3-v2",
        )
    )

    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_utilization],
        llm=judge_llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    return scores.to_pandas()
