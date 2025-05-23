# Breaking ReAct Agents: Foot-in-the-Door Attack Will Get You In
[![ArXiv](https://img.shields.io/badge/arXiv-2410.16950-b31b1b)](https://arxiv.org/abs/2410.16950)


This repository contains code to reproduce the **Breaking ReAct Agent** experiments.  
This code is based on the code published in the paper ["InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents"](https://github.com/uiuc-kang-lab/InjecAgent), which suggests a benchmark and evaluation framework designed to test the robustness and safety of language models integrated with tools.  
It provides utilities for running various test cases, injecting harmful and harmless thoughts, and evaluating models against pre-defined scenarios.

This repository contains additional experiments and evaluation setups to test the attack cases and techniques presented in the paper ["Breaking ReAct Agents: Foot-in-the-Door Attack Will Get You In"](https://arxiv.org/abs/2410.16950).


---

## Features

- **Tool-based Evaluation**: Simulates agent interactions with tools.
- **Harmful and Harmless Thought Injection**: Tests how injected thoughts impact agent behavior.
- **Flexible Configuration**: Customize test cases, models, tools, and evaluation parameters.
- **Support for Multiple Models**: Compatible with BAM API and popular LLMs (e.g., Mixtral, Llama, GPT).

---

## Setup

### Prerequisites

- Python Python 3.9.19+
- Install required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

To evaluate a model on a specific test case:

```bash
python src/evaluate_prompted_agent.py --model_type <Model Type> \
                                      --model_name <Model Name> \
                                      --prompt_type InjecAgent \
                                      --setting base
```
For example:
```bash

python src/evaluate_prompted_agent.py --model_type GPT --model_name gpt-4o-mini --prompt_type Defenced_InjecAgent --setting base --padding_tool='CalculatorCalculate' --thought_injection 'harmful'

```
Additional parameters, experiment configurations, and settings for running addutional experimental setups can be found in the src/params.py file.


### Citation

```bibtex
@inproceedings{nakash-etal-2025-breaking,
    title = "Breaking {R}e{A}ct Agents: Foot-in-the-Door Attack Will Get You In",
    author = "Nakash, Itay  and
      Kour, George  and
      Uziel, Guy  and
      Anaby Tavor, Ateret",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2025",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-naacl.363/",
    pages = "6484--6509",
    ISBN = "979-8-89176-195-7",
}

