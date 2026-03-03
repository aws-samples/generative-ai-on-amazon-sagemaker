import base64
import torch
import json
import os
from pathlib import Path
from io import BytesIO
from diffusers import StableDiffusion3Pipeline

def model_fn(model_dir):
    pipe = StableDiffusion3Pipeline.from_pretrained(model_dir, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    adapter_list = []
    adapter_root_path = Path('./adapters')

    if adapter_root_path.exists():
        print("Found adapters directory, attempting to load LoRA Adapters...")
        for adapter_dir in adapter_root_path.iterdir():
            if adapter_dir.is_dir():
                safetensors_files = list(adapter_dir.glob("*.safetensors"))
                if safetensors_files:
                    adapter_file = safetensors_files[0]  # Take first .safetensors file
                    print(f"Loading Adapter: {adapter_file.name}")
                    pipe.load_lora_weights(
                        str(adapter_file),
                        weight_name=adapter_file.name,
                        adapter_name=adapter_dir.name,
                        prefix="unet",
                        local_files_only=True
                    )
                    adapter_list.append(adapter_dir.name)
    
    os.environ["ADAPTER_LIST"] = json.dumps(adapter_list)
    return pipe

def predict_fn(data, pipe):
    # Default parameters
    defaults = {
        "num_inference_steps": 30,
        "guidance_scale": 0.0,
        "num_images_per_prompt": 1
    }
    
    adapters = json.loads(os.getenv('ADAPTER_LIST', '[]'))
    prompt = data.pop("inputs", data)
    
    pipe.disable_lora()
    
    parameters = data.get("parameters", {})
    
    # Handle adapter
    if "adapter" in parameters:
        if parameters["adapter"] in adapters:
            pipe.enable_lora()
            pipe.set_adapters([parameters["adapter"]])
        else:
            return {"error": f"unknown adapter: {parameters['adapter']}"}
    
    # Extract generation parameters
    gen_params = {k: parameters.get(k, v) for k, v in defaults.items()}
    
    # Generate images
    generated_images = pipe(prompt, **gen_params)["images"]
    
    # Encode images
    encoded_images = []
    for image in generated_images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode())
    
    return {"generated_images": encoded_images}