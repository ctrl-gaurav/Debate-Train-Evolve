# 🚀 Debate, Train, Evolve: Self-Evolution of Language Model Reasoning

<div align="center">

[![EMNLP 2025](https://img.shields.io/badge/📄_EMNLP_2025-Main_Conference-brightgreen?style=for-the-badge)](https://2025.emnlp.org/)
[![Website](https://img.shields.io/badge/🌐_Website-Live-blue?style=for-the-badge)](https://ctrl-gaurav.github.io/debate-train-evolve.github.io/)
[![GitHub](https://img.shields.io/badge/💻_Code-Repository-orange?style=for-the-badge&logo=github)](https://github.com/ctrl-gaurav/Debate-Train-Evolve)
[![License](https://img.shields.io/badge/📜_License-MIT-green?style=for-the-badge)](https://opensource.org/licenses/MIT)

*DTE Framework: Revolutionizing Language Model Reasoning*

**🏆 EMNLP 2025 Main Conference • 🧠 Multi-Agent Debate • 🎯 GRPO Training • 📈 Up to 13.92% Improvement**

[**🌟 Explore Project**](https://ctrl-gaurav.github.io/debate-train-evolve.github.io/) | [**📖 Read Paper**](https://2025.emnlp.org/) | [**💻 View Code**](https://github.com/ctrl-gaurav/Debate-Train-Evolve)

</div>

---

## 📢 Latest News & Updates

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">

### 🎉 **EMNLP 2025**
- **🆕 ACCEPTED**: Paper accepted at EMNLP 2025 Main Conference!
- **📦 RELEASED**: Complete end-to-end implementation with all paper results

### 🔮 **Key Features**
- **🗣️ Multi-Agent Debate**: Structured debate prompting without external supervision
- **⚡ GRPO Training**: Group Relative Policy Optimization for self-evolution
- **📊 Real Metrics**: Comprehensive evaluation with sycophancy detection

</div>

---

## 💡 What is DTE Framework?

**DTE (Debate, Train, Evolve)** introduces a **revolutionary approach** to evolving language model reasoning capabilities through multi-agent debate traces. Our framework combines the benefits of multi-agent collaboration with the efficiency of single-model inference, achieving **ground truth-free training** that improves reasoning without external supervision.

<div align="center">
<a href="https://ctrl-gaurav.github.io/debate-train-evolve.github.io/">
<img src="https://img.shields.io/badge/🎯_Visit_Project_Website-Live_Demo-brightgreen?style=for-the-badge&logo=github-pages" alt="Visit Project Website">
</a>
</div>

## 🎯 Overview

We introduce **DTE (Debate, Train, Evolve)**, a novel ground truth-free training framework that uses multi-agent debate traces to evolve a single language model's reasoning capabilities. Our approach combines the benefits of multi-agent debate with the efficiency of single-model inference.

### 🌟 Key Highlights

<table>
<tr>
<td width="33%">

#### 🔄 **Structured Debate Prompting**
- Reduces sycophancy bias by 50%
- Eliminates verbosity problems
- Ground truth-free training

</td>
<td width="33%">

#### ⚡ **GRPO Self-Evolution**
- No external supervision needed
- Group Relative Policy Optimization
- Iterative improvement process

</td>
<td width="33%">

#### 📈 **Strong Performance**
- Up to **13.92% accuracy gain**
- Cross-domain generalization
- **5.8% average improvement**

</td>
</tr>
<tr>
<td width="33%">

#### 🗣️ **Multi-Agent Debate**
- Structured reasoning traces
- Multiple agent perspectives
- High-quality data generation

</td>
<td width="33%">

#### 🎯 **Production Ready**
- Complete end-to-end pipeline
- Comprehensive error handling
- Industry-standard code quality

</td>
<td width="33%">

#### ⚙️ **Highly Configurable**
- YAML-based configuration
- Rich CLI interface
- Standalone components

</td>
</tr>
</table>

### 🏗️ Framework Components

1. **🗣️ Multi-Agent Debate**: Multiple agents engage in structured debates to generate high-quality reasoning traces
2. **🎯 GRPO Training**: Models are trained using **Group Relative Policy Optimization** that eliminates the need for separate value functions
3. **📈 Iterative Evolution**: The process iterates across multiple rounds, with each round improving upon the previous

### ✨ Key Features

- **🔄 Complete End-to-End Pipeline**: Fully automated from debate generation to model evolution
- **🖥️ Standalone Multi-Agent Debate**: Run debates on single queries or entire datasets independently
- **📊 Comprehensive Evaluation**: Real metrics calculation with detailed analysis including sycophancy tracking
- **⚙️ Highly Configurable**: YAML-based configuration for all parameters and settings
- **🏭 Production Ready**: Industry-standard code with comprehensive error handling, logging, and resource management
- **📈 Rich CLI Interface**: Easy-to-use command-line interface for all operations

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 16GB+ RAM recommended for full pipeline
- 8GB+ RAM for standalone debate functionality

### Installation

```bash
# Clone the repository
git clone https://github.com/ctrl-gaurav/Debate-Train-Evolve.git
cd Debate-Train-Evolve

# Create a virtual environment (recommended)
python -m venv dte_env
source dte_env/bin/activate  # On Windows: dte_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Verify Installation

```bash
# Check if everything is working
python main.py info

# Initialize default configuration
python main.py init
```

## 🚀 Quick Start

### 1. Initialize Configuration

```bash
# Create a default configuration file
python main.py init

# Or create with custom name and force overwrite
python main.py init --output my_config.yaml --force
```

### 2. Run Standalone Multi-Agent Debate

```bash
# Single query debate
python main.py debate --query "What is 15 * 24?" --agents 3 --rounds 3

# Dataset evaluation with verbose output
python main.py debate --dataset gsm8k --samples 20 --verbose

# Use specific models for agents
python main.py debate --query "Solve: 3x + 5 = 14" \
  --models "Qwen/Qwen2.5-1.5B-Instruct,meta-llama/Llama-3.2-3B-Instruct,microsoft/Phi-3.5-mini-instruct"

# Save results to file
python main.py debate --dataset arc_challenge --samples 10 --output results.json
```

### 3. Run Complete DTE Pipeline

```bash
# Run the full DTE pipeline
python main.py run --config config.yaml

# Resume from checkpoint
python main.py run --resume checkpoint.json --save-checkpoint new_checkpoint.json
```

### 4. Run Individual Components

```bash
# Generate debate data
python main.py generate --samples 100 --output debate_data.jsonl --round 1

# Train model with GRPO
python main.py train --data debate_data.jsonl --epochs 3 --batch-size 4

# Validate configuration
python main.py validate config.yaml
```

## 📊 Research Results & Performance

### 🏆 Main Performance Results (One Evolution Round)

<table>
<thead>
<tr>
<th align="center">🤖 Model</th>
<th align="center">📊 GSM8K</th>
<th align="center">🎯 GSM-Plus</th>
<th align="center">📐 MATH</th>
<th align="center">🧪 ARC-Challenge</th>
<th align="center">🏅 Best Improvement</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Qwen-2.5-1.5B</strong></td>
<td>62.77 → <strong>73.09</strong></td>
<td>42.00 → <strong>55.92</strong></td>
<td>45.08 → <strong>52.20</strong></td>
<td>69.21 → 68.36</td>
<td><strong>+13.92%</strong> (GSM-Plus)</td>
</tr>
<tr>
<td><strong>Qwen-2.5-3B</strong></td>
<td>84.08 → <strong>86.05</strong></td>
<td>61.75 → <strong>69.50</strong></td>
<td>61.36 → <strong>67.10</strong></td>
<td>83.53 → <strong>83.95</strong></td>
<td><strong>+7.75%</strong> (GSM-Plus)</td>
</tr>
<tr>
<td><strong>Qwen-2.5-7B</strong></td>
<td>90.67 → 88.32</td>
<td>68.62 → <strong>74.71</strong></td>
<td>73.08 → <strong>77.20</strong></td>
<td>87.22 → <strong>90.89</strong></td>
<td><strong>+6.09%</strong> (GSM-Plus)</td>
</tr>
<tr>
<td><strong>Qwen-2.5-14B</strong></td>
<td>92.80 → <strong>93.74</strong></td>
<td>71.79 → <strong>78.88</strong></td>
<td>76.18 → <strong>80.10</strong></td>
<td>90.27 → <strong>93.13</strong></td>
<td><strong>+7.09%</strong> (GSM-Plus)</td>
</tr>
<tr>
<td><strong>Llama-3.2-3B</strong></td>
<td>72.55 → <strong>75.06</strong></td>
<td>45.67 → <strong>53.79</strong></td>
<td>39.76 → <strong>43.80</strong></td>
<td>73.12 → <strong>77.23</strong></td>
<td><strong>+8.12%</strong> (GSM-Plus)</td>
</tr>
<tr>
<td><strong>Llama-3.1-8B</strong></td>
<td>81.73 → <strong>86.81</strong></td>
<td>55.62 → <strong>66.17</strong></td>
<td>46.66 → <strong>49.40</strong></td>
<td>77.65 → <strong>86.53</strong></td>
<td><strong>+10.55%</strong> (GSM-Plus)</td>
</tr>
</tbody>
</table>

<sub>*Values show Original → Evolved performance. Bold indicates improvement.*</sub>

### 🔍 Key Research Insights

#### 📈 **Cross-Domain Generalization**
Our evolved models show strong generalization across different reasoning tasks:
- **📊 Mathematical Reasoning**: Average **+8.92%** improvement on GSM-Plus
- **🧪 Science Reasoning**: Average **+3.67%** improvement on ARC-Challenge
- **📐 Complex Math**: Average **+4.48%** improvement on MATH dataset
- **🔄 Cross-task Transfer**: Models trained on one dataset improve on others

#### 🎯 **Performance Patterns**
- **🚀 Scaling Effects**: Consistent improvements across different model sizes
- **🎪 Dataset Flexibility**: Framework works across mathematical and scientific reasoning
- **⚡ Efficiency**: Single model inference after training maintains speed

## 📖 Detailed Usage

### Configuration

The DTE framework uses YAML configuration files. Here's a minimal example:

```yaml
# Basic Configuration
model:
  base_model_name: "Qwen/Qwen2.5-1.5B-Instruct"
  temperature: 0.7
  max_length: 2048

debate:
  num_agents: 3
  max_rounds: 3
  rcr_prompting:
    enabled: true
    require_novel_reasoning: true

training:
  learning_rate: 2e-5
  max_epochs: 3
  batch_size: 4
  lora:
    enabled: true
    rank: 128

evolution:
  max_rounds: 3
  samples_per_round: 500
```

### CLI Commands

#### Main Pipeline

```bash
# Run complete pipeline
python main.py run --config config.yaml

# Resume from checkpoint
python main.py run --resume checkpoint.json

# Save checkpoints automatically
python main.py run --save-checkpoint checkpoint.json
```

#### Data Generation

```bash
# Generate training data from debates
python main.py generate \
  --samples 500 \
  --output training_data.jsonl \
  --round 1

# Specify configuration
python main.py generate \
  --config custom_config.yaml \
  --samples 1000 \
  --output data/round_2.jsonl
```

#### Training

```bash
# Train with GRPO
python main.py train \
  --data training_data.jsonl \
  --epochs 5 \
  --batch-size 8 \
  --learning-rate 1e-5

# Specify output directory
python main.py train \
  --data data.jsonl \
  --output-dir ./models/round_1
```

#### Debate

```bash
# Run single debate
python main.py debate \
  --query "Solve: 3x + 5 = 14" \
  --agents 3 \
  --rounds 2 \
  --task-type math

# Save debate results
python main.py debate \
  --query "What causes seasons?" \
  --output debate_result.json
```

#### Utilities

```bash
# Validate configuration
python main.py validate config.yaml

# Show system information
python main.py info

# Show help
python main.py --help
```

### Python API

```python
from dte import DTEConfig, DTEPipeline

# Load configuration
config = DTEConfig.from_yaml("config.yaml")

# Create and run pipeline
pipeline = DTEPipeline(config)
results = pipeline.run_complete_pipeline()

print(f"Best performance: {results['best_performance']}")
```

## 🏗️ Architecture

### Project Structure

```
dte-framework/
├── dte/                          # Main package
│   ├── core/                     # Core components
│   │   ├── config.py            # Configuration management
│   │   ├── logger.py            # Logging system
│   │   └── pipeline.py          # Main pipeline orchestrator
│   ├── debate/                   # Multi-agent debate
│   │   ├── agent.py             # Individual debate agents
│   │   ├── manager.py           # Debate orchestration
│   │   └── prompts.py           # RCR prompting system
│   ├── training/                 # GRPO training
│   │   ├── grpo_trainer.py      # GRPO implementation
│   │   └── reward_model.py      # Reward calculation
│   ├── data/                     # Data processing
│   │   ├── generator.py         # Debate data generation
│   │   └── processor.py         # Data preprocessing
│   └── utils/                    # Utilities
├── config.yaml                  # Default configuration
├── main.py                      # CLI interface
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

### Key Components

1. **🧠 Debate Manager**: Orchestrates multi-agent debates using RCR prompting
2. **🎯 GRPO Trainer**: Implements Group Relative Policy Optimization
3. **📊 Data Generator**: Converts debate results into training data
4. **⚙️ Pipeline Controller**: Manages the complete evolution process
5. **📝 Logger**: Comprehensive logging and metrics tracking

## 📊 Performance

Our method shows consistent improvements across multiple models and datasets:

| Model | GSM8K | GSM-Plus | MATH | ARC-Challenge |
|-------|-------|----------|------|---------------|
| Qwen2.5-1.5B | +10.32 | +13.92 | +7.12 | -0.85 |
| Qwen2.5-3B | +1.97 | +7.75 | +5.74 | +0.42 |
| Llama3.1-8B | +5.08 | +10.55 | +2.74 | +8.88 |

*Values show absolute improvement over base models.*

## 🔧 Advanced Configuration

### Model Configuration

```yaml
model:
  base_model_name: "meta-llama/Llama-3.1-8B-Instruct"
  base_model_path: null  # Optional local path
  max_length: 4096
  temperature: 0.7
  top_p: 0.9
  top_k: 50
```

### Debate Settings

```yaml
debate:
  num_agents: 5
  max_rounds: 4
  consensus_threshold: 0.8

  rcr_prompting:
    enabled: true
    require_novel_reasoning: true
    critique_pairs: 2

  temperature_annealing:
    enabled: true
    start_temp: 0.7
    end_temp: 0.3
    min_model_size: "3B"
```

### Training Configuration

```yaml
training:
  learning_rate: 1e-5
  weight_decay: 0.01
  warmup_steps: 100
  max_epochs: 5
  batch_size: 8
  gradient_accumulation_steps: 2

  grpo:
    group_size: 6
    advantage_normalization: true
    clip_ratio: 0.2
    kl_penalty: 0.01

  lora:
    enabled: true
    rank: 256
    alpha: 512
    dropout: 0.1
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

### Dataset Configuration

```yaml
datasets:
  train_datasets:
    - name: "gsm8k"
      path: "openai/gsm8k"
      split: "train"
      max_samples: 2000

    - name: "math"
      path: "hendrycks/competition_math"
      split: "train"
      max_samples: 1000

  eval_datasets:
    - name: "gsm8k_test"
      path: "openai/gsm8k"
      split: "test"
      max_samples: 500
```

## 🐛 Troubleshooting

### Common Issues

#### GPU Memory Issues
```bash
# Reduce batch size
python main.py run --config config.yaml
# Edit config.yaml: training.batch_size: 2

# Enable gradient checkpointing
# Edit config.yaml: hardware.gradient_checkpointing: true
```

#### Model Loading Errors
```bash
# Check model name and availability
python main.py info

# Use local model path
# Edit config.yaml: model.base_model_path: "/path/to/model"
```

#### Configuration Validation
```bash
# Validate your configuration
python main.py validate config.yaml

# Check for common errors in logs
tail -f logs/dte_experiment.jsonl
```

### Performance Optimization

1. **GPU Optimization**:
   ```yaml
   hardware:
     mixed_precision: true
     gradient_checkpointing: true
     max_memory_per_gpu: "20GB"
   ```

2. **Training Speed**:
   ```yaml
   training:
     batch_size: 8  # Increase if GPU memory allows
     gradient_accumulation_steps: 4
     num_workers: 4
   ```

3. **Memory Management**:
   ```yaml
   model:
     max_length: 2048  # Reduce if needed
   training:
     lora:
       enabled: true  # Use LoRA for memory efficiency
   ```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/dte-framework.git
cd dte-framework

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black dte/ main.py

# Type checking
mypy dte/
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_debate.py

# Run with coverage
pytest --cov=dte tests/
```

## 🤝 Contributing & Community

We welcome contributions to improve DTE Framework and expand its capabilities:

### 🛠️ Ways to Contribute
- **🐛 Bug Reports**: Found an issue? [Report it here](https://github.com/ctrl-gaurav/Debate-Train-Evolve/issues)
- **✨ Feature Requests**: Have ideas? [Share them here](https://github.com/ctrl-gaurav/Debate-Train-Evolve/issues)
- **🔧 Code Contributions**: Submit PRs for improvements
- **📚 Documentation**: Help improve our docs
- **🤖 Model Evaluations**: Test DTE with new models

### 🔄 Contribution Process
1. **🍴 Fork** the repository
2. **🌿 Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **💾 Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **📤 Push** to the branch (`git push origin feature/amazing-feature`)
5. **🔃 Open** a Pull Request

### 👥 Development Setup

```bash
# Clone repository
git clone https://github.com/ctrl-gaurav/Debate-Train-Evolve.git
cd Debate-Train-Evolve

# Install development dependencies
pip install -e .

# Run basic validation
python main.py validate config.yaml
```

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact & Support

### 🔗 Connect With Us
- **📧 Email**: [gks@vt.edu](mailto:gks@vt.edu), [xuanw@vt.edu](mailto:xuanw@vt.edu)
- **🏠 Project Website**: [https://ctrl-gaurav.github.io/debate-train-evolve.github.io/](https://ctrl-gaurav.github.io/debate-train-evolve.github.io/)
- **🐛 Issues**: [GitHub Issues](https://github.com/ctrl-gaurav/Debate-Train-Evolve/issues)
- **💻 Repository**: [GitHub Repository](https://github.com/ctrl-gaurav/Debate-Train-Evolve)

### 🆘 Getting Help
1. Check our documentation and examples above
2. Search [existing issues](https://github.com/ctrl-gaurav/Debate-Train-Evolve/issues)
3. Create a [new issue](https://github.com/ctrl-gaurav/Debate-Train-Evolve/issues/new) with details
4. Join our community discussions

### 🙏 Acknowledgments

This work was supported by NSF NAIRR Pilot with PSC Neocortex, NCSA Delta; Amazon, Cisco Research, Commonwealth Cyber Initiative, Amazon–Virginia Tech Center for Efficient and Robust Machine Learning, and Sanghani Center for AI and Data Analytics at Virginia Tech.

---

<div align="center">

## 🚀 Ready to Evolve Your Language Models?

<a href="https://ctrl-gaurav.github.io/debate-train-evolve.github.io/">
<img src="https://img.shields.io/badge/🎯_Explore_Project-Visit_Now-brightgreen?style=for-the-badge&logo=rocket" alt="Explore Project">
</a>

**Made with ❤️ by the DTE Framework Team**

[![Virginia Tech](https://img.shields.io/badge/Virginia_Tech-CS_Department-maroon?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDE1Ljc0TDEyIDIyTDEwLjkxIDE1Ljc0TDQgOUwxMC45MSA4LjI2TDEyIDJaIiBmaWxsPSJjdXJyZW50Q29sb3IiLz4KPC9zdmc+)](https://cs.vt.edu/)
[![EMNLP 2025](https://img.shields.io/badge/EMNLP-2025-blue?style=flat&logo=academia)](https://2025.emnlp.org/)

*Advancing the frontier of language model reasoning, one debate at a time* 🌟

</div>

---

## 🔗 Quick Navigation

<div align="center">

| 🏠 [**Project**](https://ctrl-gaurav.github.io/debate-train-evolve.github.io/) | 📊 [**Performance**](#-research-results--performance) | 📖 [**Paper**](https://2025.emnlp.org/) | 💻 [**Code**](https://github.com/ctrl-gaurav/Debate-Train-Evolve) |
|:---:|:---:|:---:|:---:|
| Main website | Results & insights | Research paper | Source code |

</div>

> **🎯 Transform your language models' reasoning abilities.** DTE Framework shows what multi-agent debate can achieve for model evolution, beyond traditional fine-tuning. [**Start evolving now →**](https://ctrl-gaurav.github.io/debate-train-evolve.github.io/)

---

### 📝 Citation
If you find DTE Framework useful in your research, please cite our paper:

```bibtex
@article{srivastava2025debate,
  title={Debate, Train, Evolve: Self-Evolution of Language Model Reasoning},
  author={Srivastava, Gaurav and Bi, Zhenyu and Lu, Meng and Wang, Xuan},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year={2025},
  note={To appear}
}
```