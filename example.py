import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import json


# load distance map and sequence (preprocessed from a PDB file)
pdb = np.load('./pdb_examples/1S3P-A.npz')
dist = pdb['C_alpha']
# contact map with 10A dist threshold
cmap = np.asarray(dist < 10.0, dtype=np.float32)
sequence = str(pdb['seqres'])

# sanity checks (display sequence and contact map)
print (sequence)

plt.figure()
plt.matshow(cmap)
plt.xlabel("residues")
plt.ylabel("residues")
plt.show()


# Using DeepFRI model pre-trained on Molecular Function (MF) GO terms

# load metadata (GO-terms & GO-names)
with open("./models/DeepFRI-GraphConv_MF_model_metadata.json") as json_file:
    params = json.load(json_file)
    goterms = params['goterms']
    gonames = params['gonames']

# load model
model = tf.keras.models.load_model('./models/DeepFRI-GraphConv_MF_model')
model.summary()

# alternatively (using tensorflow functions)
# model = tf.saved_model.load('./models/DeepFRI-GraphConv_MF_model')

# #################### MAKING PREDICTIONS  ###################################

# INPUT:
# A (shape=(1, L, L)) - contact map (10A cutoff, L-protein length)
# S (shape=(1, L, 26)) - one-hot encoding of sequence (26 - number of residues)


# OUTPUT:
# Preds (shape=(#goterms, )) - prediction probabilities for each GO term


sequence = np.asarray([list(sequence)])
S = model.encode(sequence)  # (build-in function for computing one-hot encoding)
A = cmap.reshape(1, *cmap.shape)

Preds = model([A, S])
sorted_idx = np.argsort(Preds)[::-1]

print ('\n\n')
print ("Score | GO term")
print ("-----------------------------")
for i in sorted_idx[:10]:
    print ("%0.3f | %s" % (Preds[i].numpy(), gonames[i]))

# ################# COMPUTING SALIENCY  #####################################

# INPUT:
# A (shape=(1, L, L)) - contact map (10A cutoff, L-protein length)
# S (shape=(1, L, 26)) - one-hot encoding of sequence (26 - number of residues)
# goidx (shape=None) - index


# OUTPUT:
# residue_scores (shape=(L, )) - prediction score for each residue


# extract index of the target GO term (using GO ID) from a list of GO terms
# goidx = goterms.index("GO:0005509")

# alternatively (using GO name)
goidx = gonames.index("calcium ion binding")

residue_scores = model.gradCAM([A, S], goidx)
plt.figure()
plt.plot(residue_scores.numpy())
plt.xlabel("residues")
plt.ylabel("gradCAM score")
plt.title(gonames[goidx])
plt.grid()
plt.show()
