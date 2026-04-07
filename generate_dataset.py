import os
import json
import random

# --- SETUP API KEY ---
os.environ["OPENROUTER_API_KEY"] = "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"

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

def main():
    print("🤖 Initializing DeepSeek-R1 via OpenRouter...")
    
    # OpenRouter requires a specific base_url and model name format
    llm = ChatOpenAI(
        model="deepseek/deepseek-r1", # You can also try "meta-llama/llama-3.1-405b"
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost:3000", # Required by OpenRouter
            "X-Title": "University RAG Project",      # Optional dashboard title
        },
        temperature=0.1
    )

    # 1. Load PDFs
    print("📄 Loading papers from folder...")
    loader = PyPDFDirectoryLoader("./papers")
    docs = loader.load()
    
    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks.")

    # 3. Prompt Logic
    prompt = PromptTemplate.from_template("""
    You are a Research Professor writing a certification exam.
    Read this academic text:
    {text}

    TASK:
    1. Write one complex question that requires logical reasoning based on the text.
    2. Provide a 'ground_truth' answer.
    
    Output ONLY as raw JSON:
    {{
        "question": "your question here",
        "ground_truth": "your answer here"
    }}
    """)

    chain = prompt | llm
    golden_dataset = []
    
    # Generate 100 questions (or as many as you have chunks for)
    num_to_generate = min(100, len(chunks))
    sample_chunks = random.sample(chunks, num_to_generate)

    print(f"⚙️ Generating {num_to_generate} Q&A pairs. This uses reasoning, so it may take a bit...")
    
    for i, chunk in enumerate(sample_chunks):
        try:
            response = chain.invoke({"text": chunk.page_content})
            raw_json = response.content.strip()
            
            # Remove reasoning block if the model outputs it (common with DeepSeek-R1)
            if "</thought>" in raw_json:
                raw_json = raw_json.split("</thought>")[-1].strip()
            
            # Basic JSON cleanup
            if raw_json.startswith("```json"): raw_json = raw_json[7:]
            if raw_json.endswith("```"): raw_json = raw_json[:-3]
            
            qa_pair = json.loads(raw_json)
            qa_pair["contexts"] = [chunk.page_content] # Store context for Recall testing
            
            golden_dataset.append(qa_pair)
            print(f"  [{i+1}/{num_to_generate}] Question generated.")
            
        except Exception as e:
            print(f"  [!] Skipping chunk {i+1} due to logic error: {e}")

    # 4. Save Final Dataset
    with open("golden_dataset.json", "w", encoding="utf-8") as f:
        json.dump(golden_dataset, f, indent=4)

    print("\n🎉 Done! 'golden_dataset.json' is ready for evaluation.")

if __name__ == "__main__":
    main()