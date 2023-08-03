from transformers import LlamaConfig, LlamaForCausalLM
from huggingface_hub import snapshot_download
from accelerate import init_empty_weights, load_checkpoint_in_model


def classic_load(path: str, device: str = "cpu"):
    config = LlamaConfig.from_pretrained(checkpoint)
    model = LlamaForCausalLM.from_pretrained(path, torch_dtype=config.torch_dtype, device_map={"": device})
    return model

def efficient_load(path: str, device: str = "cpu"):
    config = LlamaConfig.from_pretrained(checkpoint)
    weights_location = snapshot_download(repo_id=path)

    with init_empty_weights():
        model = LlamaForCausalLM(config)
    
    load_checkpoint_in_model(model, checkpoint=weights_location, dtype=config.torch_dtype, device_map={"": device})
    return model


if __name__ == "__main__":

    checkpoint = "Mikael110/llama-2-7b-guanaco-fp16"
    model = efficient_load(path=checkpoint, device="cuda")
    print(model)
    print(model.model.layers[0].self_attn.q_proj.weight)
    input()
