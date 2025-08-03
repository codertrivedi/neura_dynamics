from transformers import pipeline

# Load summarizer on first use (can cache this globally)
summarizer = pipeline("summarization", model="google/flan-t5-small")

def summarize_text(text: str) -> str:
    if len(text.strip()) == 0:
        return "No content to summarize."

    summary = summarizer(text, max_length=60, min_length=20, do_sample=False)
    return summary[0]["summary_text"]
