import torch
from core import loader

def generate_natural_language_summary(system_prompt: str, data_context: dict) -> str:
    if not loader.llm_model:
        return "LLM model is currently loading or unavailable."
        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the data output:\n{data_context}\n\nPlease provide a clear, professional summary based ONLY on this data."}
    ]
    
    text = loader.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = loader.tokenizer([text], return_tensors="pt").to(loader.llm_model.device)
    
    with torch.no_grad():
        outputs = loader.llm_model.generate(**inputs, max_new_tokens=200, temperature=0.7)
        
    generated = outputs[0][inputs.input_ids.shape[1]:]
    return loader.tokenizer.decode(generated, skip_special_tokens=True).strip()