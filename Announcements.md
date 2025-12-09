# Announcements

## 🚀 Release: Multimodal & Diffusion Support

### 🧵 Twitter / X Thread

1/5
Big update for SanskritAI! 🚀 We've just landed Multimodal Inference and Diffusion Model support for Nalanda & Govardhan. Now you can generate images and process video with the same Sanskrit-native efficiency. 🕉️
#AI #Sanskrit #NLP #GenerativeAI

2/5
🖼️ **Diffusion Integration**: Generate culturally rich imagery with our wrapper for Stable Diffusion, optimized for our pipeline. Text-to-Image just got a whole lot more meaningful.

3/5
📹 **Multimodal**: It's not just text anymore. Our engines now support Image and Video inputs, closing the gap between ancient wisdom and modern media.

4/5
⚡ **CPU Optimized**: We've slashed the computational cost. New CPU-optimized weights mean you don't need an H100 to run these models. Accessible AI for everyone.

5/5
Check out the repo and star it! 🌟
https://github.com/ParamTatva-org/sanskrit

---

### 💼 LinkedIn Post

**Subject: Bridging Ancient Wisdom and Modern Generative AI**

I'm excited to announce a major update to the **SanskritAI** ecosystem (Nalanda & Govardhan models). We have significantly expanded our capabilities beyond text to truly multimodal applications.

✨ **Key Features Released:**
*   **Multimodal Inference**: Native support for Image and Video inputs, allowing for complex reasoning across media types.
*   **Diffusion Models**: Seamless integration for image generation, enabling text-to-image workflows within our Sanskrit-native architecture.
*   **CPU Optimization**: Massively reduced inference costs. We've optimized weights to ensure these models are performant on standard CPU hardware, democratizing access to advanced AI.

This release represents a huge step forward in our mission to build linguistically efficient AI that solves the $O(n^2)$ computational bottleneck.

Check out the code and contribute: https://github.com/ParamTatva-org/sanskrit

#AI #GenerativeAI #Sanskrit #MachineLearning #OpenSource #Nalanda #Govardhan

---

### 📸 Instagram / Facebook

**(Image specific)**
*Visual Idea: A split screen showing ancient Sanskrit manuscripts on one side and a futuristic digital neural network on the other, merging in the center.*

**Caption:**
History meets the Future. 🕉️🤖

We've just updated our **SanskritAI** models with massive new capabilities!
You can now use Nalanda and Govardhan for:
✅ **Text-to-Image Generation**
✅ **Video Processing**
✅ **CPU-Optimized Inference**

We are proving that linguistic efficiency = computational efficiency. 📉💰

Experience the power of efficient, Sanskrit-native AI.
Link in bio! 🔗

#Sanskrit #AI #Tech #Innovation #Coding #OpenSource #DeepLearning

---

## 🚀 Release: Native CPU Support

![Sanskrit AI on CPU](resources/cpu_optimization_announcement_1765312730935.png)

**Headline:** Native CPU Support for Sanskrit Inference Models! 🧠💻

**Body:**

Namaste Developers! 🙏

We are thrilled to announce that **SanskritAI** now supports fully optimized CPU inference!

Previously, running our large `Nalanda-62M` and Diffusion models required dedicated GPUs. Today, we've broken that barrier.

**What's New?**
*   **Dynamic Quantization:** We've implemented `torch.quantization.quantize_dynamic` for our Nalanda models, significantly reducing memory usage and speeding up inference on standard CPUs.
*   **Device Selection:** You can now explicitly choose between `--device cpu`, `--device cuda`, or `--device mps` (Apple Silicon) for all our inference scripts.
*   **Accessible to All:** Run advanced multimodal Sanskrit AI on your laptop, no heavy hardware required.

**How to use:**

```bash
# Run Nalanda on CPU with quantization
python src/nalanda_inference.py --model_path weights/nalanda_v1 --prompt "Your prompt" --device cpu --cpu_quantize

# Run Image Generation on CPU
python diffusion_inference.py image --prompt "Futuristic Temple" --device cpu
```

Get started today and bring the wisdom of the past to the machines of the future!

👉 **[Link to Repository]**

#SanskritAI #OpenSource #MachineLearning #CPUOptimization #AI #Python #Devanagari
