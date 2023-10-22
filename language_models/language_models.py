# Importing the necessary libraries
import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain.chat_models import ChatOpenAI
from langchain.llms import CTransformers, HuggingFacePipeline
from langchain.llms.huggingface_hub import HuggingFaceHub
from loguru import logger as log
from transformers import AutoTokenizer, TextStreamer, pipeline

# %%: Configuration for the language models
model_config = {
    'max_new_tokens': 4096, 'temperature': 0.1, 'top_p': 0.95,
    'repetition_penalty': 1.15, 'context_length': 4096
}

# %%: Device to use
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# %%: Get language models
def get_huggingface_model():
    """Get the HuggingFace model from the HuggingFace Hub"""
    return HuggingFaceHub(repo_id="google/flan-t5-xxl", config=model_config)


def get_openai_model():
    """Get the OpenAI model from the OpenAI API"""
    return ChatOpenAI(config=model_config)


def get_mistral_model():
    """Get the Mistral model with CTransformers"""
    model_path = 'models/mistral-7b-instruct-v0.1.Q4_K_M.gguf'
    return CTransformers(model=model_path, model_type='mistral', device=DEVICE, do_sample=True, config=model_config)


def get_gguf_model():
    """Get the GGUF model with CTransformers"""
    model_path = 'models/llama-2-13b-chat.Q4_K_M.gguf'
    return CTransformers(model=model_path, model_type='llama', device=DEVICE, do_sample=True, config=model_config)


def get_gptq_model():
    """Get the GPTQ model with AutoGPTQForCausalLM"""
    model_name = 'TheBloke/Llama-2-13B-chat-GPTQ'
    model_basename = "model"
    model = AutoGPTQForCausalLM.from_quantized(model_name, revision="main", model_basename=model_basename,
                                               use_safetensors=True, trust_remote_code=True,
                                               inject_fused_attention=False,
                                               device=DEVICE, quantize_config=None)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    text_pipeline = pipeline("text2text-generation",
                             model=model,
                             tokenizer=tokenizer,
                             max_new_tokens=4096,
                             temperature=0.1,
                             top_p=0.95,
                             do_sample=True,
                             repetition_penalty=1.15,
                             streamer=streamer)
    return HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.1})


def get_language_model(model_name):
    """
    Get a language model based on the given name.

    Parameters
    ----------
    model_name: str
        The name of the model to get.

    Returns
    -------
    HuggingFacePipeline, CTransformers, or ChatOpenAI instance:
        The loaded language model.

    Warnings
    - log (loguru): If the model name is unknown.
    """
    if model_name == 'Llama-2-13B-chat-GPTQ (GPU required)':
        return get_gptq_model()
    elif model_name == 'Llama-2-13B-chat-GGML (CPU only)':
        return get_gguf_model()
    elif model_name == 'HuggingFace Hub (Online)':
        return get_huggingface_model()
    elif model_name == 'OpenAI API (Online)':
        return get_openai_model()
    elif model_name == 'Mistral-7B (CPU only)':
        return get_mistral_model()
    else:
        log.error(f'Unknown model name: {model_name}')
