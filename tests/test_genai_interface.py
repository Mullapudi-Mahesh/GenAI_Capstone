import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.genai_interface import df, summary, prompt_template, chain

def test_genai_chain_runs():
    """Ensure GenAI prompt returns text output"""
    sample = summary.head(3).to_string(index=False)
    result = chain.run(campaign_data=sample)
    assert isinstance(result, str)
    assert len(result) > 20 