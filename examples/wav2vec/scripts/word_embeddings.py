import io
import numpy as np
import sys

src_path = sys.argv[1]
src_word = sys.argv[2]
tgt_word = sys.argv[3]

def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            if word in word2id:
                continue
            vect = np.fromstring(vect, sep=' ')
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

nmax = 100000
src_embeddings, src_id2word, src_word2id = load_vec(src_path, nmax)
print(len(src_embeddings))

def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=5):
    print("Nearest neighbors of \"%s\":" % word)
    word2id = {v: k for k, v in src_id2word.items()}
    word_emb = src_emb[word2id[word]]
    scores = (tgt_emb.dot(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    print(len(scores), len(tgt_id2word))
    for i, idx in enumerate(k_best):
        print('%.4f - %s' % (scores[idx], tgt_id2word[idx]))


def get_sim(word, tgt_word, src_emb, src_id2word, tgt_emb):
    print("Nearest neighbors of \"%s\":" % word)
    word2id = {v: k for k, v in src_id2word.items()}
    word_emb = src_emb[word2id[word]]
    tgt_word_emb = tgt_emb[word2id[tgt_word]]
    scores = (tgt_word_emb.dot(word_emb))
    print(scores)


get_nn(src_word, src_embeddings, src_id2word, src_embeddings, src_id2word, K=10)
get_sim(src_word, tgt_word, src_embeddings, src_id2word, src_embeddings)

