import sys, os, psutil, pickle
mode = sys.argv[1]
if mode == "edge":
    import numpy as np, zlib      # numpy only: what the compressed model needs
    d = pickle.load(open("out/edge_model.pkl", "rb"))
else:
    import numpy as np, sklearn, scipy
    from sklearn.neural_network import MLPClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    d = pickle.load(open("out/full_model.pkl", "rb"))
print(mode, round(psutil.Process(os.getpid()).memory_info().rss / 1e6, 1), "MB RSS")
