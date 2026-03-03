FROM vllm/vllm-openai:latest

RUN pip install transformers==4.57.1 \
    peft==0.17.1 \
    accelerate==1.11.0 \
    bitsandbytes==0.48.2 \
    datasets==4.0.0 \
    huggingface_hub[hf_transfer] \
    safetensors>=0.6.2 \
    sentencepiece==0.2.1 \
    tokenizers>=0.22.1 \
    vllm==0.11.0 \
    qwen-vl-utils==0.0.14

COPY serve /usr/bin/serve
RUN chmod 777 /usr/bin/serve

ENTRYPOINT [ "/usr/bin/serve" ]
