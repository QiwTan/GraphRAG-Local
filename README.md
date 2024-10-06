# GraphRAG-Local


This project allows you to use a Large Language Model (LLM) to extract a knowledge graph from PDF documents and perform Q&A on the generated knowledge graph.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [1. Preprocess PDF Files](#1-preprocess-pdf-files)
  - [2. Generate Knowledge Graph](#2-generate-knowledge-graph)
  - [3. Perform Q&A on Knowledge Graph](#3-perform-qa-on-knowledge-graph)
  - [4. Run the Entire Workflow](#4-run-the-entire-workflow)
- [Notes](#notes)
- [License](#license)

## Prerequisites

- Python 3.7 or higher
- Required Python packages (see [Installation](#installation))
- PDF documents to process

## Installation

1. **Clone this repository or copy the code to your local machine.**

2. **Install the required Python packages:**

   ```bash
   pip install langchain networkx tqdm PyPDF2 langchain_experimental langchain_community
   ```

3. **Set up the LLM (Large Language Model):**

   - Ensure you have access to the `Ollama` model specified in `config.py`.
   - Modify the `llm_model_name` in `config.py` as needed.

## Configuration

All configurable parameters are located in the `config.py` file.

Open `config.py` and adjust the parameters as needed:

```python
# config.py

# Path Configuration
input_folder = "/Users/tanqiwen/Documents/GraphRAG-Local"  # Folder containing input PDF files
output_folder = "/Users/tanqiwen/Documents/GraphRAG-Local"  # Output folder
json_folder_path = output_folder  # Folder to store chunked JSON files
combined_graphml_file = "global_knowledge_graph.graphml"  # Name of the combined graph file

# Text Splitting Parameters
chunk_size = 1200  # Size of text chunks
chunk_overlap = 400  # Overlap size between text chunks

# LLM Model Parameters
llm_model_name = "llama3.1:latest"  # Name of the LLM model to use, can be modified as needed
```

## Usage

### 1. Preprocess PDF Files

This step processes the PDF files, cleans the text, splits it into chunks, and saves the chunks as JSON files.

Run the `preprocess_pdfs.py` script:

```bash
python preprocess_pdfs.py
```

- Ensure your input PDF files are placed in the folder specified by `input_folder` in `config.py`.
- The processed JSON files will be saved in the `output_folder`.

### 2. Generate Knowledge Graph

This step uses the LLM to extract entities and relationships from the text chunks and constructs a knowledge graph.

Run the `process_chunks.py` script:

```bash
python process_chunks.py
```

- The script reads the chunked JSON files from `json_folder_path`.
- Individual knowledge graphs for each document and a global knowledge graph will be saved in `.graphml` format in the `output_folder`.

### 3. Perform Q&A on Knowledge Graph

This step allows you to perform interactive Q&A on the global knowledge graph using the LLM.

Run the `graph_qa.py` script:

```bash
python graph_qa.py
```

- The script loads the combined knowledge graph specified by `combined_graphml_file` in `config.py`.
- You can input questions in the console and receive answers based on the knowledge graph.
- Type `exit` to quit the interactive session.

### 4. Run the Entire Workflow

To execute all the above steps sequentially using a single command, use the `main.py` script.

Run the `main.py` script:

```bash
python main.py
```

- This script will:
  1. **Preprocess PDFs:** Clean and split PDF files into chunks.
  2. **Build Knowledge Graphs:** Extract entities and relationships to build the knowledge graph.
  3. **Start Q&A:** Launch the interactive Q&A session on the generated knowledge graph.

**Note:** Ensure that each step completes successfully before moving to the next. If any step fails, the workflow will stop, and an error message will be displayed.

### Example Output

```bash
(rag1) tanqiwen@tanqiwendebijibendiannao-2 GraphRAG-Local % /Users/tanqiwen/Documents/rag1/bin/python /Users/tanqiwen/Documents/GraphRAG-Local/graph_qa.py
Please enter your question (or type 'exit' to quit): PaLM
/Users/tanqiwen/Documents/GraphRAG-Local/graph_qa.py:44: LangChainDeprecationWarning: The method `Chain.run` was deprecated in langchain 0.1.0 and will be removed in 1.0. Use :meth:`~invoke` instead.
  response = graph_qa_chain.run(user_input)


> Entering new GraphQAChain chain...
Entities Extracted:
PaLM
Full Context:
PaLM RELATED_TO Language Modeling
PaLM PERFORMS_FEW_SHOT_LEARNING hundreds of language understanding and generation benchmarks
PaLM ACHIEVES_BREAKTHROUGH_PERFORMANCE BIG-bench benchmark
PaLM OUTPERFORMS_FINETUNED_STATE_OF_THE_ART multi-step reasoning tasks
PaLM HAS_STRONG_CAPABILITIES_IN multilingual tasks and source code generation
PaLM PERFORMS_COMPETITIVELY_ON wide array of benchmarks
PaLM STUDIES_TRAINING_DATA_MEMORIZATION model scale
PaLM PROVIDES_COMPREHENSIVE_ANALYSIS_ON bias and toxicity
PaLM RELATED_TO researcher at Google
PaLM TRAINED_ON training dataset
PaLM RELATED_TO Training Dataset
PaLM USED_ON Training Infrastructure
PaLM USED_ON Training Setup
PaLM EVALUATED_ON Evaluation
PaLM PERFORMED_ON BIG-bench
PaLM PERFORMED_ON Code Tasks
PaLM PERFORMED_ON Multilingual Natural Language Generation
PaLM PERFORMED_ON Analysis
PaLM RELATED_WORK Multilingual Question Answering
PaLM MEMORIZED Memorization
PaLM FOUND_TO_CONTAMINATE Dataset Contamination
PaLM FOUND_TO_EXHIBIT_REPRESENTATIONAL_BIAS Representational Bias Analysis
PaLM FOUND_TO_PRODUCE_TOXIC_OUTPUT Toxicity in open-ended generation
PaLM FOUND_TO_HAVE_LIMITATIONS Limitations
PaLM REQUIRED_ETHICAL_CONSIDERATION Ethical Considerations
PaLM RELATED_WORK Related Work
PaLM OPEN_QUESTION_IN_SCALING Open Questions in Scaling
PaLM CONCLUDED Conclusion
PaLM RELATED_TO BERT and T5
PaLM USED_IN English NLP tasks on smaller models
PaLM RELATED_TO BIG-bench results
PaLM RELATED_TO BERT
PaLM RELATED_TO T5
PaLM IS_A Transformer architecture
PaLM SUCCEEDED_PREVIOUS_MODEL GPT-3
PaLM USED_TOGETHER_WITH Pathways
PaLM TRAINED_ON 780 billion tokens of high-quality text
PaLM HAS_PARAMETER 540 billion
PaLM TRAINED_ON 780 billion tokens of text
PaLM ACHIEVES_RESULT state-of-the-art few-shot results
PaLM PERFORMS_WELL_ON hundreds of natural language tasks
PaLM ACHIEVES_RESULT breakthrough performance
PaLM TRAINED_ON TPU v4 chips
PaLM EVALUATED_ON hundreds of natural language, code, and mathematical reasoning tasks
PaLM USED_FOR TPU v4 Pods
PaLM ACHIEVED_HIGH_EFFICIENCY 46.2%
PaLM ACHIEVED_STATE_OF_THE_ART_RESULTS Section 6
PaLM GENERATED_EXPLANATIONS explanations using chain-of-thought prompting (Wei et al., 2022b)
PaLM IMPROVED_SCALING large LMs
PaLM OUTPERFORMED_OTHER_MODELS GLaM, GPT-3, Megatron–Turing NLG, Gopher, Chinchilla, LaMDA
PaLM PERFORMS_BETTER_THAN GLaM
PaLM PERFORMS_BETTER_THAN Megatron–Turing NLG
PaLM PERFORMS_BETTER_THAN Gopher
PaLM PERFORMS_BETTER_THAN Chinchilla
PaLM PERFORMS_BETTER_THAN LaMDA
PaLM SCALING_BEHAVIOR 8B
PaLM SCALING_BEHAVIOR 62B
PaLM SCALING_BEHAVIOR 540B
PaLM PERFORMS_WELL_ON few-shot evaluation
PaLM MATCHES_STATE_OF_THE_ART state of the art
PaLM DEMONSTRATES_EXPLORATORY_CAPABILITIES exploratory capabilities
PaLM EXHIBITS_DISCONTINUOUS_IMPROVEMENTS discontinuous improvements
PaLM PERFORMS_BETTER_THAN prior state of the art in non-English summarization tasks
PaLM OUTPERFORMS prior state of the art in translation tasks
PaLM USES_ARCHITECTURE Transformer
PaLM SET_A_NEW_STATE_OF_THE_ART Winogender coreference task
PaLM FALSLEY_AFFIRMS_STEREOTYPES Muslims
PaLM FALSLEY_AFFIRMS_STEREOTYPES_ASSOCIATING_WITH terrorism, extremism, and violence
PaLM USES_ACTIVATION_FUNCTION SwiGLU Activation
PaLM USES_LAYER_FORMULATION Parallel Layers
PaLM USES_TRANSFORMER_MODEL_ARCHITECTURE Transformer model architecture
PaLM IMPROVES training speed
PaLM EVALUATED_WITH Ablation experiments
PaLM EVALUATED_AT 8B scale
PaLM EVALUATED_AT 62B scale
PaLM PREDICTS quality neutral effect of parallel layers
PaLM USES RoPE Embeddings
PaLM SHARES Input-Output Embeddings
PaLM DOES_NOT_USE Biases
PaLM HAS_LAYER 32
PaLM HAS_PARAMETER 16
PaLM HAS_BATCH_SIZE 8
PaLM USES_DATA_SOURCE Social media conversations
PaLM USES_DATA_SOURCE Filtered webpages
PaLM USES_DATA_SOURCE Books
PaLM USES_DATA_SOURCE GitHub (code)
PaLM BASED_ON JAX and T5X
PaLM USED TPU conﬁguration
PaLM SUPPORTED researchers
PaLM SCALES_TRAINING_ACROSS two TPU v4 pods
PaLM EXECUTES_FORWARD_AND_BACKWARD_COMPUTATION standard within-pod data and model parallelism
PaLM RELATED_TO Scaling Language Modeling
PaLM IMPLEMENTED_ON DCN links
PaLM TRAINING_ON Language Models
PaLM ACHIEVES_THROUGHPUT_OF 1.95x
PaLM HAS_OPTIMIZATION Optimizations
PaLM HAS_Hardware_FLOPs_utilization 57.8%
PaLM RELATED_TO Megatron LM
PaLM RELATED_TO Narayanan et al. (2021b)
PaLM USES Adam
PaLM USES Adafactor
PaLM USES global norm gradient clipping
PaLM HAS_FEATURE bitwise determinism
PaLM HAS_FEATURE large batch size schedule
PaLM INCREASES_EFFICIENCY TPU
PaLM IS_REPRODUCIBLE Bitwise determinism
PaLM USES_METHOD JAX+XLA+T5X
PaLM HAS_PIPELINE Deterministic dataset pipeline
PaLM USES_METHOD Dropout
PaLM IS_STABLE Training Instability
PaLM IS_MORE_INSTABLE_THAN Smaller models
PaLM EVALUATED_ON Du et al.
PaLM EVALUATED_ON Brown et al.
PaLM IS_STATE_OF_THE_ART Prior state-of-the-art results from other large language models
PaLM IMPROVES_ON Natural Questions
PaLM IMPROVES_ON TriviaQA
PaLM PERFORMED_ON SOTAPaLM 540B
PaLM PERFORMED_ON Prior SOTAPaLM 540B
PaLM PERFORMED_ON TriviaQA (EM)
PaLM PERFORMED_ON Natural Questions (EM)
PaLM PERFORMED_ON Web Questions (EM)
PaLM PERFORMED_ON Lambada (EM)
PaLM PERFORMED_ON HellaSwag
PaLM PERFORMED_ON StoryCloze
PaLM PERFORMED_ON Winograd
PaLM PERFORMED_ON Winogrande
PaLM ACHIEVED 57.3a69.4
PaLM ACHIEVED 57.8a70.8
PaLM ACHIEVED 58.6a 70.8
PaLM ACHIEVED 81.5b77.6
PaLM ACHIEVED 84.0b79.9
PaLM ACHIEVED 85.0b 81.5
PaLM ACHIEVED 71.1a80.8
PaLM ACHIEVED 71.8a82.9
PaLM ACHIEVED 71.8a 83.3
PaLM ACHIEVED 64.7a75.5
PaLM ACHIEVED 66.5a78.7
PaLM ACHIEVED 67.0a 79.6
PaLM ACHIEVED_SCORE 41 .5
PaLM WORKED_ON Natural Language Generation and Natural Language Understanding results across 29 benchmarks using 1-shot evaluation.
PaLM ACHIEVED_SCORE 57 .7
PaLM WORKED_ON Table 5: Average (Avg) Natural Language Generation and Natural Language Understanding results across 29 benchmarks using 1-shot evaluation.
PaLM IMPROVES MMLU benchmark
PaLM FINETUNED_ON SuperGLUE benchmark
PaLM CONVERGED_IN less than 15K steps
PaLM PERFORMED_BETTER_THAN T5-11B
PaLM PERFORMED_BETTER_THAN ST-MoE-32B
PaLM OBTAINED competitive close-to-SOTA performance
PaLM OUTPERFORMS_HUMANS humans
PaLM ACHIEVES higher score than the average score of humans
PaLM DESCRIPTION_OF Scaling Language Modeling with Pathways
PaLM ACHIEVES_ACCURACY wikihow
PaLM ACHIEVES_ACCURACY logical args
PaLM IMPROVES_UPON PaLM 62B
PaLM IMPROVES_UPON PaLM 8b
PaLM HAS_DIFFICULTY human asked to solve the task
PaLM PERFORMS_BEYOND_AVERAGE_HUMAN Human
PaLM EXHIBITS_LANGUAGE_ABILITY Persian Idioms
PaLM EXHIBITS_LANGUAGE_ABILITY Swedish to German Proverbs
PaLM LEVERAGES_MEMORIZATION_CAPABILITY Periodic Elements
PaLM EXHIBITS_NLP_CAPABILITY Common Morpheme
PaLM EXHIBITS_NLP_CAPABILITY Sufficient Information
PaLM EXHIBITS_NLP_CAPABILITY Logical Args
PaLM DETERMINES_CAUSE_EFFECT Cause and Effect Task
PaLM RELATED_TO BIG-bench Lite
PaLM USES 8B model
PaLM USES 540B model
PaLM PERFORMS_BEST_ON 3 tasks
PaLM USED_IN training data
PaLM EVALUATED_ON reasoning tasks
PaLM IMPROVED_BY Chain-of-thought prompting
PaLM RELATED_TO Nye et al. (2021)
PaLM RELATED_TO Cobbe et al. (2021)
PaLM RELATED_TO Wei et al. (2022b)
PaLM ACHIEVED_SOTA SOTA accuracy across a variety of arithmetic and commonsense reasoning tasks
PaLM USED_CHAIN_OF_THOUGHT_PROMPTING chain-of-thought prompting
PaLM EVALUATED_ON arithmetic datasets GSM8K
PaLM USED_METHOD common sense reasoning datasets
PaLM EVALUATED_ON CommonsenseQA
PaLM EVALUATED_ON StrategyQA
PaLM ACHIEVED_PERFORMANCE 58%
PaLM OUTPERFORMS Cobbe et al.
PaLM ACHIEVES_SOTA SVAMP
PaLM ACHIEVES_SOTA ASDiv
PaLM ACHIEVES_SOTA AQuA
PaLM USES_TECHNIQUE Chain-of-Thought Prompting
PaLM USES_TECHNIQUE Model Scaling
PaLM ACHIEVES_SOTA GSM8K
PaLM ACHIEVES_SOTA MAWPS
PaLM USED_FOR few-shot prompts
PaLM USED_TO_CREATE test set
PaLM EVALUATED_ON DeepFix (Gupta et al., 2017)
PaLM USED_WITH compiler errors
PaLM USED_TO_TEST 1260 programs
PaLM COMPARABLE_TO Codex
PaLM MEASURED_BY Researchers (Thoppilan et al., 2022)
PaLM INCLUDED_IN_TRAINING_DATA GitHub code
PaLM ACHIEVES_STATE_OF_THE_ART_PERFORMANCE_ON Code and natural language tasks
PaLM ACHIEVES_EXCELLENT_PERFORMANCE_ON code and natural language tasks
PaLM ACHIEVES_BEST_PUBLISHED_PERFORMANCE_IN both
PaLM WAS_TRAINED_ON Python code
PaLM WAS_TRAINED_ON Codex models
PaLM ACHIEVES_COMPARABLE_PERFORMANCE_IN few-shot evaluations and previously-published results
PaLM IS_MORE_SAMPLE_EFFICIENT_THAN smaller models
PaLM FINE_TUNED_ON code
PaLM FINE_TUNED_TO 540B token count
PaLM IMPROVES HumanEval
PaLM OBTAINED_SCORE 58 .1
PaLM PRODUCES_EDITS small edits
PaLM COMPILED DeepFix
PaLM ASSOCIATED_WITH Schuster et al.
PaLM ASSOCIATED_WITH Chen et al.
PaLM USED_FOR machine translation
PaLM USED_FOR evaluation
PaLM USED_FOR machine translation settings
PaLM TRANSLATES_TO English
PaLM USED_FOR_EVALUATION WMT
PaLM RELATED_TO WMT'14 English-French
PaLM RELATED_TO WMT'16 English-German
PaLM RELATED_TO WMT'16 English-Romanian
PaLM RELATED_TO WMT'19 French-German
PaLM RELATED_TO Kazakh
PaLM HAS_TRANSLATION_CAPABILITY translation capabilities
PaLM OUTPERFORMS FLAN
PaLM SCORED_HIGHEST_IN_TABLE_14 Table 14
PaLM RELATED_TO Generalist models
PaLM COMPARABLE_TO Specialized models
PaLM COMPARE_AGAINST Megatron-Turing NLG
PaLM EVALUATED_ON Generation Evaluation and Metrics benchmark (GEM)
PaLM USED_FOR_FINETUNING mT5
PaLM USED_FOR_FINETUNING BART
PaLM EVALUATED_ON Gehrmann et al. (2021)
PaLM RELATED_TO known weaknesses
PaLM EVALUATED_ON GEM benchmark
PaLM EVALUATED_ON MLSum
PaLM EVALUATED_ON WikiLingua
PaLM EVALUATED_ON XSum
PaLM EVALUATED_ON Clean E2E NLG
PaLM EVALUATED_ON Czech Restaurant response generation
PaLM RELATED_TO NLG tasks
PaLM USED_FOR_FEW_SHOT_INFERENCE few-shot exemplars
PaLM USED_FOR_FINETUNING decoder-only architecture
PaLM HAS_PARAMETER_SETTING constant learning rate of 5 ×10−5
PaLM SELECTED_MODEL_CHECKPOINT best model checkpoint for each dataset
PaLM USED_FOR_INFERENCE top-k sampling with k=10
PaLM FINE_TUNED_TOGETHER_WITH T5 XXL baselines
PaLM DECODED_USING beam-search with a beam size of 4
PaLM PARAMETERIZED_BY parameters
PaLM DECODED_USING beam-search

> Finished chain.
Knowledge graph's response: I'd be happy to help you understand the capabilities and characteristics of PaLM (a transformer-based language model). Here's what I found:

**Key Features and Capabilities:**

1. **Transformer-based Architecture**: PaLM uses a transformer-based architecture, similar to other popular models like BERT and RoBERTa.
2. **Large Model Size**: PaLM comes in two variants: 8B (8 billion parameters) and 540B (540 billion parameters).
3. **Excellent Performance on Arithmetic and Commonsense Reasoning Tasks**: PaLM has achieved state-of-the-art (SOTA) performance on various arithmetic and commonsense reasoning tasks.
4. **Common Sense Reasoning**: PaLM is trained on common sense reasoning datasets, such as GM8K, which evaluates its ability to reason about the world.
5. **Few-shot Inference**: PaLM can be fine-tuned for few-shot inference with small exemplars, making it suitable for applications where limited training data is available.

**Comparison and Related Work:**

1. **Related to Generalist Models**: PaLM is compared to generalist models like Codex and Megatron-Turing NLG.
2. **Comparable to Specialized Models**: PaLM is also comparable to specialized models, such as the XSum model.
3. **Chain-of-Thought Prompting**: PaLM was trained using chain-of-thought prompting, which is a technique used to generate human-like text.

**Applications and Use Cases:**

1. **Natural Language Processing (NLP)**: PaLM can be applied to various NLP tasks, including question answering, text classification, and machine translation.
2. **Code and Natural Language Tasks**: PaLM achieves excellent performance on code and natural language tasks, such as code completion and natural language generation.
3. **Evaluation and Testing**: PaLM has been used for evaluation and testing in various applications, including few-shot inference and human evaluation.

**Limitations:**

1. **Known Weaknesses**: PaLM is associated with known weaknesses, such as the need for large amounts of training data to achieve state-of-the-art performance.
2. **Parameter Setting**: The model uses a constant learning rate of 5 ×10−5, which might not be suitable for all applications.

Overall, PaLM appears to be a powerful and versatile language model that can excel in various NLP tasks, particularly those requiring common sense reasoning and few-shot inference capabilities.
Source: global_knowledge_graph.graphml
```

## Notes

- **Model Availability:** Ensure that the specified LLM model (`llm_model_name`) is available and correctly set up in your environment.