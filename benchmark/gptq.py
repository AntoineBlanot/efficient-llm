from argparse import ArgumentParser
import time
from tqdm import tqdm

import torch
import transformers
from transformers import LlamaTokenizerFast, LlamaForCausalLM
from accelerate.hooks import add_hook_to_module, AlignDevicesHook
from datasets import load_dataset

import quant
from utils import find_layers

def load_quant(model, checkpoint, wbits, groupsize=-1, fused_mlp=True, eval=True, warmup_autotune=True):
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)

    if eval:
        quant.make_quant_attn(model)
        quant.make_quant_norm(model)
        if fused_mlp:
            quant.make_fused_mlp(model)
    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)
    model.seqlen = 2048
    print('Done.')

    return model

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Llama model to load")
    parser.add_argument("-q", "--quant", type=str, help="Quantitized model to load")
    parser.add_argument("-d", "--dataset", type=str, choices=["oasst1"], help="Llama model to load")
    args = parser.parse_args()
    
    model = load_quant(args.model, args.quant, 4, 128).to("cuda")
    tokenizer = LlamaTokenizerFast.from_pretrained(args.model)
    add_hook_to_module(model, AlignDevicesHook(execution_device="cuda", io_same_device=True), append=True)
    print(model)

    dataset = load_dataset("timdettmers/openassistant-guanaco", split="test")
    dataset = dataset.map(lambda x: tokenizer(x["text"], return_tensors="pt"), remove_columns=["text"])
    dataset = dataset.add_column("labels", dataset["input_ids"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    losses = []
    start_time = time.time()
    max_memory = torch.cuda.memory_allocated() / 1024 / 1024

    with torch.no_grad():
        for inputs in tqdm(dataset, desc="Benchmarking {}".format(args.dataset)):
            outputs = model(**inputs)
            max_memory = max(max_memory, torch.cuda.memory_allocated() / 1024 / 1024)
            losses.append(outputs.loss)
    
    run_time = time.time() - start_time
    loss = torch.stack(losses).mean()
    ppl = torch.exp(loss)
    print("Loss: {}\t\tPPL: {}\t\tTime: {} s\t\tMax memory: {} MiB".format(loss, ppl, run_time, max_memory))