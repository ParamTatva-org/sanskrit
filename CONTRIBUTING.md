# CONTRIBUTING.md

## 🌍 Elevating Sanskrit into the Digital Age

The release of the **Nalanda-62M-Multi** and the foundation of the **Govardhan** model is not just a technological achievement; it's an invitation. By demonstrating the profound computational efficiency of Sanskrit, we have unlocked the potential for a new era of ultra-efficient, highly precise digital tools.

We are calling on developers, linguists, designers, and enthusiasts to help translate this efficiency into tangible, real-world **web utilities, applications, and services** that benefit learners, researchers, and the global digital community.

-----

## 🚀 Why Build with Nalanda and Govardhan?

The models released here are not general-purpose transformers; they are **linguistic compression engines**. Your contributions will benefit from:

1.  **Lower Latency and Cost:** Because Nalanda uses fewer tokens ($n$) to encode the same information, your apps will process queries faster and cheaper than those relying on lower-density English models.
2.  **Unparalleled Precision:** The Kāraka system eliminates ambiguity, meaning your applications will have inherently fewer reasoning errors when dealing with complex data and cross-referencing.
3.  **Maximum Context:** You can build applications that handle long-form philosophical texts, legal documents, or complex academic papers with unparalleled context retention.

-----

## 🛠️ Contribution Areas and Project Ideas

We need contributions across the entire stack, from front-end user experience to specialized backend services.

### A. Core LLM Utilities & Infrastructure

These projects leverage the raw output of the models for essential NLP tasks:

| Project Idea | Description | Model Focus |
| :--- | :--- | :--- |
| **Pāṇini Sandbox** | A web utility that accepts Sanskrit input and displays the full morphological breakdown (root, affix, Kāraka case, Sandhi splits) as derived by the LLM's understanding. | **Govardhan** (Leverages rule generalization) |
| **Token Compression Visualizer** | An interactive tool (based on `mantr.net`) that takes a complex English sentence and its Sanskrit equivalent, showing the exact token count difference and the resulting $O(n^2)$ computational savings. | **Nalanda** (To demonstrate TCR) |
| **API Wrapper** | Development of robust, low-latency API wrappers (Node.js, Go, etc.) to simplify integration of the Nalanda inference engine into commercial or open-source products. | Infrastructure |

### B. Educational & Pedagogical Applications

Sanskrit's precise structure makes it an ideal language for educational AI tools:

  * **Adaptive Learning Chatbot:** An LLM-powered tutor that generates personalized exercises based on morphological complexity or Kāraka usage, providing instant, precise feedback.
  * **Shloka & Metre Analyzer:** An app that accepts Sanskrit verse, automatically resolves Sandhi, analyzes the metre (Chanda), and provides a precise word-by-word morphological tag and translation using Nalanda's semantic understanding.
  * **Virtual Dictionary & Lexicon Builder:** A utility that uses the LLM to auto-suggest *Dhātu* (verb roots) and word derivations based on a partial input, helping users explore the generative power of the language.

### C. Creative & Multimodal Services (Nalanda-62M-Multi)

These projects utilize the multimodal capabilities of the larger Nalanda model:

  * **Sanskrit Art Description:** An app where users upload an image (e.g., a temple, a deity, a natural scene) and the model generates a descriptive Sanskrit verse (Śloka) or prose.
  * **Devanāgarī OCR Enhancement:** Fine-tuning the multimodal aspect for superior optical character recognition (OCR) of historical Devanāgarī manuscripts, using the model’s linguistic knowledge to correct common OCR errors (e.g., resolving ambiguous characters based on grammatical probability).

-----

## 🤝 How to Get Started

We welcome contributions of all kinds—from code to documentation, and even translating our materials into other languages.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ParamTatva-org/sanskrit.git
    cd sanskrit
    ```
2.  **Read the Research:** Review **`संस्कृत.md`** to fully understand the $\text{TCR}$ and the **Govardhan Hypothesis**—this will inform your development priorities.
3.  **Check Issues:** Look at the [Issues page](https://www.google.com/search?q=https://github.com/ParamTatva-org/sanskrit/issues) for specific tasks tagged **`good first issue`** or **`app idea`**.
4.  **Propose a Project:** If you have a larger project idea, open a new issue with the tag **`project proposal`** to discuss it with the core team before starting development.

**Join us in building the next generation of linguistically efficient AI\!**