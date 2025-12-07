# 🤖 DocuCode: QLoRA Fine-Tuned Code Comment Generator

## ✨ Overview

DocuCode is a specialized Large Language Model (LLM) tailored for code documentation tasks. It leverages the strength of a Code LLM (like StarCoder or CodeLLaMA) and fine-tunes it efficiently using the **QLoRA (Quantized Low-Rank Adapters)** technique on a curated dataset of high-quality code and docstring pairs. This model is designed to adhere to internal code standards, significantly reducing documentation overhead and improving codebase maintainability.

## 🚀 Key Features

  * **Efficient Fine-Tuning (QLoRA):** Achieved production-level quality by fine-tuning a massive base model (e.g., CodeLLaMA-7B) using only commodity or Colab GPUs, minimizing training time and VRAM usage.
  * **Context-Aware Generation:** Model is trained to include context (imports, class definitions) when generating comments for a specific function, leading to semantically accurate documentation.
  * **High-Quality Output:** Optimized for generating **structured docstrings** (e.g., Google or NumPy style) and clear, concise **inline comments** following internal style guides.
  * **CodeBLEU Evaluation:** Model performance is validated using industry-standard CodeBLEU scores, which measure syntactic and semantic accuracy, not just token overlap.

-----

## ⚙️ Technology Stack

| Component | Purpose | Details & Implementation |
| :--- | :--- | :--- |
| **Base LLM** | Foundation model for code understanding. | **StarCoder / CodeLLaMA-7B** (Chosen for strong code pre-training.) |
| **Fine-Tuning** | Efficient training of task-specific parameters. | **PEFT (QLoRA)** with `bitsandbytes` (4-bit quantization). |
| **Framework** | Orchestration of training, datasets, and models. | **Python, Hugging Face Transformers, PEFT, Datasets, TRL.** |
| **Training Hardware** | Resource environment for the computationally intensive training phase. | **Colab GPU** (or internal cluster GPU, details specified in configuration). |
| **Data Processing** | Parsing and preparation of source code text. | **Python `ast` module** (for function splitting), **`pandas`** (for filtering/structuring). |
| **Evaluation** | Measuring the generated comment quality. | **CodeBLEU**, Human Evaluation Samples. |

-----

## 🧠 Implementation Details

### 1\. Custom Dataset Creation

The core value of DocuCode is the custom dataset, tailored to the firm's specific language and style.

  * **Source Data:** Combines public code datasets (e.g., filtered CodeSearchNet) with **proprietary/internal code snippets** and their corresponding high-quality docstrings.
  * **Parsing:** A custom script uses the language-specific **Abstract Syntax Tree (AST)** parser (e.g., Python's `ast`) to isolate code blocks.
      * **Snippet Definition:** Each sample is defined as a `function/method` body.
      * **Context Preservation:** The parser extracts surrounding imports and class names, prepending them to the code snippet for the LLM to understand the namespace.
  * **Data Formatting:** All data is converted into a structured, instruction-following format required by the TRL (Transformer Reinforcement Learning) library:
    ```json
    {
      "prompt": "Generate a concise NumPy-style docstring for this Python function:",
      "code_snippet": "def calculate_net_income(revenue, costs):\n    return revenue - costs",
      "completion": "Returns the net income by subtracting costs from revenue."
    }
    ```

### 2\. QLoRA Fine-Tuning Process

This ensures the 7B model is specialized without excessive hardware.

  * **Quantization:** The base model is loaded using 4-bit quantization (`load_in_4bit=True`), drastically reducing memory footprint.
  * **LoRA Adapter Config:** The PEFT configuration specifies the Low-Rank Adapter parameters:
      * `r` (rank): e.g., $r=16$.
      * `lora\_alpha`: e.g., $\alpha=32$.
      * `target\_modules`: Set to target critical layers like `q_proj` and `v_proj` for efficient attention adaptation.
  * **Training Arguments:** Defined using the `TrainingArguments` class from Hugging Face:
      * **Optimizer:** Utilized **8-bit AdamW** for further memory savings.
      * **Batch Size:** Adjusted `per_device_train_batch_size` and `gradient_accumulation_steps` to maximize the effective batch size while staying within VRAM limits.
      * **Data Type:** Used `torch.float16` for computations where possible (mixed-precision training).

### 3\. Evaluation & Validation

Model quality is verified through specialized metrics.

  * **Automated Metrics:** CodeBLEU is the primary metric, tracking syntactic (AST overlap) and semantic (Dataflow overlap) accuracy between the generated and gold-standard comments.
  * **Human-in-the-Loop (HITL) Check:** A small test set is subject to human review to ensure compliance with specific internal style guides (e.g., clarity, appropriate verb tense, absence of technical jargon).
  * **Inference Check:** Post-training, the model is tested with functions requiring complex logic (loops, recursion) to ensure it correctly identifies the *intent* of the code.

-----

## 🚀 Usage and Deployment

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/docucode.git
cd docucode

# Create and activate environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
# Note: Requires specific versions of transformers/peft/bitsandbytes
pip install -r requirements.txt
