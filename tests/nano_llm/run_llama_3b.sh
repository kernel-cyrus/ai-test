# Requirements
# 1. Install Docker
# 2. Install Jetson-Containers
# 3. Create Huggingface token

jetson-containers run							\
        --env HUGGINGFACE_TOKEN=hf_!FUOFgBvGMdHKyWiCa!ktKJMTitksNaMZmtV!\
	--env HF_ENDPOINT=https://hf-mirror.com                         \
	$(autotag nano_llm) 						\
	python3 -m nano_llm.chat --api mlc 				\
	--model deepseek-ai/DeepSeek-R1-Distill-Llama-8B 		\
	--prompt "Can you tell me a joke about llamas?"
