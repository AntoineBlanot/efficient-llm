from argparse import ArgumentParser
import time
from tqdm import tqdm

import torch
from transformers import LlamaTokenizerFast, LlamaForCausalLM, BitsAndBytesConfig
from accelerate.hooks import add_hook_to_module, AlignDevicesHook
from datasets import load_dataset


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Llama model to load")
    parser.add_argument("-d", "--dataset", type=str, choices=["oasst1"], help="Llama model to load")
    args = parser.parse_args()

    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = LlamaForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16, quantization_config=quantization_config).eval()
    tokenizer = LlamaTokenizerFast.from_pretrained(args.model)
    add_hook_to_module(model, AlignDevicesHook(execution_device="cuda", io_same_device=True), append=True)
    print(model)
    print("Model device map: {}".format(model.hf_device_map))

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