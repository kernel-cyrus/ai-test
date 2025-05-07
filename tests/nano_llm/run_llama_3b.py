# 1. Install Jetson Containers
# 2. jetson-containers run --volume /home/cyrus:/home/cyrus $(autotag nano_llm)
# 3. Run

from nano_llm import NanoLLM

model = NanoLLM.from_pretrained(
       "TinyLlama/TinyLlama-1.1B-Chat-v1.0",                # HuggingFace repo/model name, or path to HF model checkpoint
       api='mlc',                                           # supported APIs are: mlc, awq, hf
       api_token='hf_!FUOFgBvGMdHKyWiCaktKJMTitksNaMZmtV!',  # HuggingFace API key for authenticated models ($HUGGINGFACE_TOKEN)
       quantization='q4f16_ft'                              # q4f16_ft, q4f16_1, q8f16_0 for MLC, or path to AWQ weights
)

while True:

    prompt = input("prompt: ")

    if prompt == 'exit':
        break

    response = model.generate(prompt, max_new_tokens=1024)

    for token in response:
        print(token, end='', flush=True)
