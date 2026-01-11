import os
import json
import re
from dotenv import load_dotenv
from typing import List, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def _safe_json_extract(text: str) -> Dict:
    """
    Extract JSON object from LLM output safely.
    """
    try:
        # Try direct parse
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract JSON block using regex
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Absolute fallback
    return {
        "faithfulness_score": 0.0,
        "hallucination": "Yes",
        "explanation": "Gemini did not return valid JSON.",
    }


def evaluate_with_gemini(
    question: str,
    answer: str,
    contexts: List[str],
) -> Dict:
    """
    Gemini-based LLM evaluation for RAG answers.
    Always returns a valid dictionary.
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    context_text = "\n\n".join(contexts)

    prompt = ChatPromptTemplate.from_template(
        """
You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems.

Evaluate the answer strictly using the provided context.

Context:
{context}

Question:
{question}

Answer:
{answer}

Tasks:
1. Give a faithfulness score between 0 and 1.
2. State whether the answer contains hallucinations (Yes/No).
3. Briefly explain your reasoning.

Respond ONLY in valid JSON.
No markdown.
No explanations outside JSON.

JSON format:
{{
  "faithfulness_score": 0.0,
  "hallucination": "Yes or No",
  "explanation": "short explanation"
}}
"""
    )

    response = llm.invoke(
        prompt.format(
            context=context_text,
            question=question,
            answer=answer,
        )
    )

    return _safe_json_extract(response.content)
