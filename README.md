# AutoAWQ Model Quantization CLI 🚀

A command-line tool for quantizing Hugging Face models using AutoAWQ and uploading them to the Hub.

## Quick Start 👾

### Google Colab ☁️
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gaoafIrAj60pvByFbN5iqTBySyhW6zEN?usp=sharing)

1. Click the Colab badge above
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run all cells in order

### Hugging Face Space 🤗
[![Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/mwdut/awq-my-repo)

Try the web interface version directly in your browser - no setup required!

### Local Installation 💻
```bash
git clone https://github.com/mwdut/awq-my-repo.git
cd awq-my-repo
python -m venv awq-env
source awq-env/bin/activate  # Linux/Mac
# awq-env\Scripts\activate   # Windows
pip install -r requirements.txt
# run
python app.py
```

## Requirements 📋

- Python 3.11+
- CUDA-compatible GPU (recommended)
- Hugging Face account
- 8GB+ RAM (16GB+ recommended)

## Acknowledgments

- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) - Core quantization library
- [AWQ Paper](https://arxiv.org/abs/2306.00978) - Original research
