import os
import sys
import json

data_json = sys.argv[1]
out_dir = sys.argv[2]

os.makedirs(out_dir, exist_ok=True)

wav_scp = open(f"{out_dir}/wav.scp", "w")
img_scp = open(f"{out_dir}/img.scp", "w")

with open(data_json) as f:
    data = json.load(f)
img_path = data["image_base_path"]
audio_base_path = data["audio_base_path"]

for v in data["data"]:
    img = f"{img_path}/{v['image']}"
    wav = f"{audio_base_path}/{v['wav']}"
    print(f"{v['uttid']} {wav}", file=wav_scp)
    print(f"{v['uttid']} {img}", file=img_scp)
