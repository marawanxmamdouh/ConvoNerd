# Downloading Models

You can download your desired model from the following links. The default models are:

- **Mistral**: `mistral-7b-instruct-v0.1.Q4_K_M.gguf`
- **LLAMA 2**: `llama-2-13b-chat.Q4_K_M.gguf`
- **GPTQ**: `TheBloke/Llama-2-13B-chat-GPTQ`
- **HuggingFace**: `google/flan-t5-xxl`

You can choose the model size that best suits your needs and your device's capabilities.

1. If you want to work with **Mistrial**, you can download it from this link:
   - [Download Mistral](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main)

2. If you want to work with **Zephyr**, you can download it from this link:
   - [Download Zephyr](https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/tree/main)

3. If you want to work with **Llama-2-7b**, you can download it from this link:
   - [Download Llama-2-7b](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/tree/main)

4. If you want to work with **Llama-2-13B**, you can download it from this link:
   - [Download Llama-2-13B](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/tree/main)

5. Otherwise, you can download any model you want from [HuggingFace](https://huggingface.co/models?pipeline_tag=text-generation).

After downloading the model, make sure to place it in the `models` folder.

If you decide to use any other size or model that is not one of defaults, don't forget to update the `model_name` or `path` in the `conf/language_models.yaml` file.
