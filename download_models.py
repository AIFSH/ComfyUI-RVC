import os
from pathlib import Path
from huggingface_hub import hf_hub_download

now_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(now_dir, "rvc")


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_DIR ,"assets","weights"), exist_ok=True)
    weights_path = os.path.join(BASE_DIR ,"assets")
    print("Downloading hubert_base.pt...")
    hf_hub_download(repo_id="lj1995/VoiceConversionWebUI",
                    filename="hubert_base.pt",
                    subfolder= "",
                    local_dir= os.path.join(weights_path, "hubert"))
    print("Downloading rmvpe.pt...")
    hf_hub_download(repo_id="lj1995/VoiceConversionWebUI",
                    filename="rmvpe.pt",
                    subfolder= "",
                    local_dir= os.path.join(weights_path, "rmvpe"))
    
    
    print("Downloading pretrained models:")

    model_names = [
        "D40k.pth",
        "D48k.pth",
        "G32k.pth",
        "G40k.pth",
        "G48k.pth",
        "f0D32k.pth",
        "f0D40k.pth",
        "f0D48k.pth",
        "f0G32k.pth",
        "f0G40k.pth",
        "f0G48k.pth",
    ]
    for model in model_names:
        print(f"Downloading {model}...")
        hf_hub_download(repo_id="lj1995/VoiceConversionWebUI",
                    filename=model,
                    subfolder= "pretrained",
                    local_dir= weights_path)

   
    print("Downloading pretrained models v2:")

    for model in model_names:
        print(f"Downloading {model}...")
        hf_hub_download(repo_id="lj1995/VoiceConversionWebUI",
                    filename=model,
                    subfolder= "pretrained_v2",
                    local_dir= weights_path)

    print("All models downloaded!")
