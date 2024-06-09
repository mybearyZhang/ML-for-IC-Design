# ML-for-IC-Design
Project of Machine Learning (CS3308 SJTU)



## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Task 1](#task-1-logic-synthesis-evaluation-prediction)
- [Task 2](#task-2-logic-synthesis-decision)
- [Task 3](#task-3-high-level-synthesis-with-large-language-model)


## Installation

Clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/mybearyZhang/ML-for-IC-Design.git

# Navigate to the project directory
cd ML-for-IC-Design

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation
Before running the tasks, ensure your datasets are properly prepared and located in the designated directories.

You should have the following directory structure:

```plaintext
ML-for-IC-Design/
├── project_data/
│   ├── adder_0.pkl
│   ├── adder_1.pkl
│   └── ...
├── project_data2/
│   ├── adder_0.pkl
│   ├── adder_1.pkl
│   └── ...
├── InitialAIG/
│   ├── test/
│   │   └── ...
│   └── train/
│       └── ...
├── lib/
│   └── 7nm/
│       └── 7nm.lib
└──src
    └── ...
```

## Task 1: Logic Synthesis Evaluation Prediction

This task involves predicting the evaluation metrics of a logic synthesis process using machine learning techniques. 

### Steps to Run Task 1

1. **Run read.py to Read and Cache Data:**

   Execute the `read.py` script to read data from the `project_data` directory and cache it for further processing.

   ```bash
   python -m src.task1.read
   ```

2. **Run split.py to Prepare Train-Test Splits:** 

   Execute the `split.py` script to convert the cached data into AIG format and split it into train and test sets.

   ```bash
   python -m src.task1.split
   ```

3. **Run train.py to Train the Model:**

   Execute the `train.py` script to train the model using the train set.

   ```bash
   python -m src.task1.train
   ```

## Task 2: Logic synthesis decision


## Task 3: High level synthesis with large language model
In this task, we introduce the prompting process utilized for generating Verilog code using Large Language Models (LLMs). Two prompting techniques are employed: naive prompting and in-context learning.

### Naive Prompting
Naive prompting involves providing task descriptions directly to LLMs to generate corresponding Verilog code. Prompts outline the task scope, granting LLMs creative freedom. Sample prompts include:

> + Row Multiplication: Generate Verilog code for row multiplication.
> + Finite State Machine: Create Verilog code for a finite state machine with more than 2 stages.
> + Memory Unit Design: Develop Verilog code for a memory-like storage structure.

### In Context Learning
In Context Learning entails dynamic model adaptation based on task-specific context. Initial solutions, generated without parameter fine-tuning, are iteratively refined using original samples as prompts. The iterative refinement process involves:

> + First Prompting: Initial Verilog code generation for row multiplication.
> + Second Prompting: Refinement by altering matrix dimensions.
> + Third Prompting: Further refinement by incorporating additional matrices.
> + Fourth Prompting: Enhancement with error handling.
