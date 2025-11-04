import os
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

# --- API Key Setup ---
os.environ["OPENAI_API_KEY"] = ""

# --- Load Campaign Data ---
data_path = "data/adtech_campaign_performance.csv"
df = pd.read_csv(data_path, parse_dates=["date"])

summary = (
    df.groupby(["region", "device"])
    .agg({"impressions": "sum", "clicks": "sum", "conversions": "sum", "cost": "sum"})
    .reset_index()
)
summary["CTR"] = (summary["clicks"] / summary["impressions"]) * 100
summary["CVR"] = (summary["conversions"] / summary["clicks"].replace(0, pd.NA)) * 100
summary["CPA"] = summary["cost"] / summary["conversions"].replace(0, pd.NA)

print(" Campaign performance summary loaded.")
print(summary.head())

# --- Define Prompt ---
prompt_template = PromptTemplate(
    input_variables=["campaign_data"],
    template=(
        "You are an AdTech optimization assistant.\n"
        "Given this campaign summary:\n"
        "{campaign_data}\n\n"
        "Analyze and provide:\n"
        "1 Key insights (CTR, CVR, CPA)\n"
        "2 Optimization recommendations\n"
        "3 A/B test ideas\n"
        "4 Warning on anomalies or poor segments\n"
        "Format as clear bullet points."
    )
)

# --- Initialize LLM ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
chain = LLMChain(prompt=prompt_template, llm=llm)

# --- Run Analysis ---
sample_summary = summary.head(8).to_string(index=False)
response = chain.run(campaign_data=sample_summary)

print("\n AI-Generated Optimization Insights:\n")
print(response)
