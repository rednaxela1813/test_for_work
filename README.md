# AI-Powered News Intelligence (Case Study)

## What it does
- Scrapes latest articles from a public news source by keyword/topic
- Cleans and enriches data (dedup, keyword extraction, simple categorization)
- Uses a free LLM API (Hugging Face Inference API) to generate:
  - 2–3 sentence summaries
  - insight headlines
  - sentiment (Positive/Neutral/Negative)
- Produces charts and an HTML report

## Setup
Python 3.11 recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"


# Hugging Face Inference Providers
export HF_TOKEN="hf_..."
export HF_PROVIDER="cerebras"          # or hf-inference / nscale / nebius if available
export HF_GEN_MODEL="meta-llama/Llama-3.1-8B-Instruct"

python main.py --topic "AI" --max-articles 25

or

python main.py --topic "AI" --max-articles 25 --no-llm