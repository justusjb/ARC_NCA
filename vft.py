import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHANNELS = 50
BATCH_SIZE = 8
MASKING = 0
GENESIZE = 25
GENE_HIDDEN_N = 132
GENE_PROP_HIDDEN_N = 132
NUM_PROBLEMS = 120