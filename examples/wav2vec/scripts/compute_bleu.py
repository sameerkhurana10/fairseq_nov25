import sys
import kaldiio
import torch

src_labse_scp = sys.argv[1]
tgt_labse_scp = sys.argv[2]
out_translation_file = sys.argv[3]

src_uttids = []
src_labse = []
with open(src_labse_scp) as f:
    for line in f:
        splits = line.rstrip().split()
        src_uttids.append(splits[0])
        labse = torch.tensor(kaldiio.load_mat(splits[1])).float()
        assert len(labse.shape) == 2
        # take the labse embedding of the source text
        labse = labse[0]
        src_labse.append(labse)

tgt_uttids = []
tgt_labse = []
with open(tgt_labse_scp) as f:
    for line in f:
        splits = line.rstrip().split()
        tgt_uttids.append(splits[0])
        labse = torch.tensor(kaldiio.load_mat(splits[1])).float()
        assert len(labse.shape) == 2
        # take the labse embedding of the target text
        labse = labse[1]
        tgt_labse.append(labse)

src_labse = torch.stack(src_labse, dim=0)
tgt_labse = torch.stack(tgt_labse, dim=0)

# the embedding vectors are already normalized and hence,
# the below is the cosine similarity
print("Compute Similarity matrix")
S = torch.mm(src_labse, tgt_labse.t())
print("Similarity Matrix: ", S.shape)
tgt_ids = torch.argmax(S, dim=1).tolist()
out_translation_file = open(out_translation_file, "w")
for i, tgt_id in enumerate(tgt_ids):
    tgt_uttid = tgt_uttids[tgt_id]
    src_uttid = src_uttids[i]
    print(src_uttid, tgt_uttid, file=out_translation_file)



