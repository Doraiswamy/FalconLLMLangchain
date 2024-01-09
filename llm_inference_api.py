import json
import os
from pprint import pprint

import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from flask import Flask, request, jsonify

app = Flask(__name__)

global_model = None
global_tokenizer = None


def generateResponse(question: str, tokenizer, model, generation_config, DEVICE) -> str:
    prompt = f"""
<human>: {question}
<assistant>:
""".strip()
    encoding = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    assistant_start = "<assistant>:"
    response_start = response.find(assistant_start)
    response = response[response_start + len(assistant_start):].strip()

    if len(response.split()) > generation_config.max_new_tokens:
        sentences = response.split('.')
        generated_tokens = 0
        generated_response = ''
        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            if generated_tokens + sentence_tokens <= generation_config.max_new_tokens:
                generated_response += sentence + '.'
                generated_tokens += sentence_tokens
            else:
                break

        if not generated_response.endswith('.'):
            generated_response += '.'

        return generated_response.strip()
    else:
        return response


def loadLLMModel():
    PEFT_MODEL = "NECLLM"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    config = PeftConfig.from_pretrained(PEFT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, PEFT_MODEL)
    return model, tokenizer


def inference(model, tokenizer, prompt):
    generation_config = model.generation_config
    generation_config.max_new_tokens = 110
    generation_config.temperature = 0.7
    generation_config.top_p = 0.7
    generation_config.num_return_sequences = 1
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    DEVICE = "cuda:0"
    llmResponse = generateResponse(prompt, tokenizer, model, generation_config, DEVICE)
    return llmResponse


@app.route('/inference', methods=['POST'])
def getLLMResponse():
    print(request.json['prompt'])
    if request.method == 'POST':
        global global_model
        global global_tokenizer
        print(global_model)
        print(global_tokenizer)
        if global_model is None:
            global_model, global_tokenizer = loadLLMModel()
        llmResponse = inference(global_model, global_tokenizer, request.json['prompt'])
        response_data = {'status': 200, 'data': llmResponse}
        return jsonify(response_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
