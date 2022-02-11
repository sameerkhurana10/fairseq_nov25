import sys
import json

data_json = sys.argv[1]
xlsr_scp = sys.argv[2]

with open(data_json) as f:
    data = json.load(f)
img_path = data["image_base_path"]
audio_base_path = data["audio_base_path"]

xlsr_dict = {}
with open(xlsr_scp) as f:
    for line in f:
        splits = line.rstrip().split()
        xlsr_dict[splits[0]] = splits[1]

for v in data["data"]:
    img = f"{img_path}/{v['image']}"
    wav = f"{audio_base_path}/{v['wav']}"
    if wav in list(xlsr_dict.keys()):
        print(f"{xlsr_dict[wav]}\t{img}")
