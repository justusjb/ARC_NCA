import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHANNELS = 23
BATCH_SIZE = 16
MASKING = 0
GENESIZE = 0 # set to 0 for non-Engram or weird issues may occur
GENE_HIDDEN_N = 386
GENE_PROP_HIDDEN_N = 132
NUM_PROBLEMS = 120