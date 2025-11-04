'''
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall
from datasets import Dataset
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- Load a local model instead of OpenAI ---
model_name = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
hf_llm = HuggingFacePipeline(pipeline=generator)

# --- Example dataset ---
data = {
    "question": ["How can we improve CTR in Europe?"],
    "contexts": [["Europe region has best CTR but needs CPA improvement."]],
    "answer": ["Focus on mobile ads and optimize cost per click."],
    "ground_truth": ["Europe has high CTR but high cost per acquisition; focus on optimizing spend."]
}
dataset = Dataset.from_dict(data)

# --- Evaluate using Hugging Face model ---
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, context_precision, context_recall],
    llm=hf_llm
)

print("\n RAGAS Evaluation Results (Offline Model):")
for metric, score in results.items():
    print(f"{metric}: {score:.3f}")
'''