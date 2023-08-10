import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast
from huggingface_hub import snapshot_download
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.hooks import attach_align_device_hook, add_hook_to_module, AlignDevicesHook

torch.manual_seed(0)

def load(path: str, dtype: torch.dtype = torch.float16, device_map = "auto", exec_device: str = "cuda") -> LlamaForCausalLM:
    config = LlamaConfig.from_pretrained(path)
    config.torch_dtype = dtype

    with init_empty_weights():
        print("Loading empty model ...")
        LlamaForCausalLM._set_default_torch_dtype(config.torch_dtype)
        model = LlamaForCausalLM(config)

    print("Loading weights and dispatch ...")
    weights_location = snapshot_download(repo_id=path)
    model = load_checkpoint_and_dispatch(model, checkpoint=weights_location, dtype=config.torch_dtype, device_map=device_map)
    attach_align_device_hook(model, execution_device=exec_device, offload=True)
    add_hook_to_module(model, AlignDevicesHook(execution_device=exec_device, io_same_device=True), append=True)

    print("Model loaded !")
    return model

if __name__ == "__main__":

    checkpoint = "Mikael110/llama-2-13b-guanaco-fp16"
    model = load(path=checkpoint, device_map={"": "cpu"}, exec_device="cuda")
    print(model)
    print(model.hf_device_map)
    
    
    tok = LlamaTokenizerFast.from_pretrained(checkpoint)
    inputs = tok(["Llama is the best"], return_tensors="pt")
    
    outputs = model.generate(inputs.input_ids, max_new_tokens=20)
    
    print(tok.decode(outputs[0], skip_special_tokens=True))
    