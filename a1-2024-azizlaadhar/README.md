[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/A17Y3Zfo)
<div style="padding:15px 20px 20px 20px;border-left:3px solid green;background-color:#e4fae4;border-radius: 20px;color:#424242;">

## Welcome to the **1st assignment** for the **CS-552: Modern NLP course**!

In the first two parts of this assignment, you need to construct a dataset and use it to train language models (LSTM and Transformer);

In the third part, you will finetune language models (RNN-based and Transformer-based Encoder-Decoder) on a text simplification task.


### **Tasks**
- **[PART 1: Data Preprocessing](#1)**
    - [1.1 Data Cleaning](#11)
    - [1.2 Build Vocabulary](#12)
    - [1.2 Get PyTorch Dataset](#13)
- **[PART 2: Training Language Models](#2)**
    - [2.1 Vanilla LSTM](#21)
    - [2.2 Transformer (DistilGPT2)](#22)
- **[PART 3: Finetuning Language Models](#3)**
    - [3.1 Encoder-Decoder Model](#31)
    - [3.2 Transformer (T5)](#32)


### **Deliverables**
- ✅ This Jupyter notebook
- ✅ `data.py`, `modeling.py` file
- ✅ Checkpoints for two LSTM-variant and DistilGPT2 language models (Part 2)
- ✅ Checkpoints for finetuned encoder-decoder and T5 language models (Part 3)
- ✅ `./tensorboard` directory with logs for all trained/finetuned models

Large files such as model checkpoints and logs should be pushed to the repository with Git LFS. You may also find that training the models on a GPU can speed up the process, we recommend using Colab's free GPU service for this. A tutorial on how to use Git LFS and Colab can be found [here](https://github.com/epfl-nlp/cs-552-modern-nlp/blob/main/Exercises/tutorials.md).

</div>