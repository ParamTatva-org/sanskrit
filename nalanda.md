# Nalanda LLM Information Card

| **Attribute**          | **Details** |
|------------------------|-------------|
| **Name**              | Nalanda-62M-Multi |
| **Description**       | A 62-million-parameter, transformer-based, multimodal large language model (LLM) designed using native Sanskrit linguistic principles for enhanced efficiency and context handling. It leverages Sanskrit's deterministic structure for superior performance in Indic languages. |
| **Key Features**      | - Extreme efficiency with a Token Compression Ratio (TCR) of ≈1.8:1 compared to English.<br>- Effective context capacity of ≈15,000 words within 8192 tokens.<br>- Open-source weights and inference engine.<br>- Multilingual support with strong cross-lingual generalization across Indic languages.<br>- Multimodal capabilities (text and image processing). |
| **Model Size**        | 62 million parameters |
| **Performance Metrics**| - Inference speed: >3× faster than comparable English models due to reduced sequence lengths.<br>- Training cost: ≈1,900× lower than 120B-parameter English models.<br>- Potential for a 1M-parameter model (Govardhan) to outperform larger English LLMs on specific tasks. |
| **Supported Languages**| Primary: Sanskrit (Devanāgarī script)<br>Additional: Hindi, English, and related Indic languages via generalization. |
| **Organization**      | ParamTatva-org |
| **Contributors**      | Prabhat Kumar Singh (research paper author); supported by AI4Bharat (IndicTrans) and Google AI Startup Grant. |
| **License**           | Not explicitly specified. |
| **Usage Instructions**| 1. Clone the repository: `git clone https://github.com/ParamTatva-org/sanskrit`<br>2. Install dependencies: `pip install -r requirements.txt`<br>3. Download weights: `python weights/nalanda-62m-multi/download_weights.py`<br>4. Run inference: `python src/nalanda_inference.py --model_path ./weights/nalanda-62m-multi --prompt "कः भवान्?"`<br>Online demo available at [https://mantr.net](https://mantr.net). |
| **Release Details**   | Flagship multimodal model with weights in `./weights/nalanda-62m-multi` and inference engine in `./engine/`. Research paper: "The Computational Superiority of Sanskrit-Native LLMs" (2025 pre-print). |
| **GitHub Repository** | [https://github.com/ParamTatva-org/sanskrit](https://github.com/ParamTatva-org/sanskrit) |
