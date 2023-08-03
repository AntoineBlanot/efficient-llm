from transformers import LlamaConfig, LlamaForCausalLM
from huggingface_hub import snapshot_download
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

checkpoint = "Mikael110/llama-2-7b-guanaco-fp16"
weights_location = snapshot_download(repo_id=checkpoint)

config = LlamaConfig.from_pretrained(checkpoint)
with init_empty_weights():
    model = LlamaForCausalLM(config)

print(model)

model = load_checkpoint_and_dispatch(model, checkpoint=weights_location)
print(model)