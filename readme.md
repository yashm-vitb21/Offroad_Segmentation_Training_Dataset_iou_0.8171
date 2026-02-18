# 🌲 Off-Road Terrain Semantic Segmentation
**Autonomous Path Planning using Deep Learning (U-Net)**

!

## 📌 Project Overview
This project addresses the challenge of autonomous navigation in unstructured environments. Developed for a hackathon, it uses a **U-Net Convolutional Neural Network** to segment off-road terrains (paths, vegetation, and obstacles) from camera feeds. This allows for intelligent real-time path planning in environments where traditional road markers are absent.

## 🚀 Technical Configuration
The model is optimized for high-performance training on a mobile workstation (NVIDIA RTX 4050 GPU), balancing model depth with memory efficiency.

* **Model Architecture:** U-Net (Encoder-Decoder with Skip Connections)
* **Input Resolution:** 256 x 256 pixels
* **Batch Size:** 4
* **Training Duration:** 50 Epochs
* **Optimization:** Adam Optimizer ($LR=1e-4$)
* **Loss Function:** Cross-Entropy Loss

---

## 📊 Dataset & Label Mapping
The dataset consists of **689 image-mask pairs**. A significant technical hurdle involved handling non-standard pixel values in the raw masks ($200 \dots 10000$). I implemented a custom remapping layer to convert these into sequential IDs for the GPU.


