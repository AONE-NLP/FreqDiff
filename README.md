# 👻 Denoising the Future: Context-Aware Spectral Diffusion for Temporal Knowledge Graph Extrapolation

<p align="center">
  <a href="https://arxiv.org/abs/2608.20804"><img src="https://img.shields.io/badge/Paper-PDF-red.svg" alt="Paper"></a>
  <a href="https://github.com/your-username/FreqDiff"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white" alt="PyTorch"></a>
  <img src="https://img.shields.io/badge/EMNLP-2026-blue" alt="EMNLP 2026">
</p>

> **Official PyTorch Implementation of the EMNLP 2026 (Main) paper:** *Denoising the Future: Context-Aware Spectral Diffusion for Temporal Knowledge Graph Extrapolation*

---






## 📜 Directory Structure

```text
FreqDiff_clean_*/
├── src/
│   ├── main.py          # training and evaluation entry point
│   ├── model.py         # FreqDiff model
│   ├── diffusion.py     # diffusion decoder
│   ├── utils_entity.py  # data utilities and ranking metrics
│   ├── knowledge_graph.py
│   ├── regcn.py
│   └── step_sample.py
├── data/
│   ├── ICEWS14/
│   ├── ICEWS18/
│   ├── ICEWS05-15/
│   └── GDELT/
├── checkpoints/         # generated during training; empty in this package
├── logs/                # generated during training/evaluation; empty in this package
└── results/             # generated prediction/ranking logs; empty in this package
```

Each dataset folder contains the raw temporal quadruple splits (`train.txt`, `valid.txt`, `test.txt`), dictionaries (`entity2id.txt`, `relation2id.txt`, `stat.txt`), and precomputed `history_seq/` files used by the model.


## 💾 Environment

A CUDA-enabled PyTorch environment is recommended.

Required Python packages include:

```text
python >= 3.8
torch
numpy
scipy
pandas
tqdm
fitlog
dgl
torch-scatter
```

Install the PyTorch, DGL, and torch-scatter builds that match your CUDA version.


## 📌 Citation

If you find our work or code useful for your research, please consider citing our KDD 2026 paper:

```

```

------

## 🙏 Acknowledgements

We sincerely appreciate the authors of the following open-source repositories for their valuable codebases and contributions:

- [DiffuTKG](https://github.com/AONE-NLP/DiffuTKG)
- [NADEx](https://github.com/AONE-NLP/TKG-NADEx)
