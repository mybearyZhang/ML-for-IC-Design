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

Moreover, you need to install the `abc`, `abc_py`, `yosys` packages for logic synthesis. You can install it refer to the following repositories:

- [abc](https://github.com/berkeley-abc/abc)
- [abc_py](https://github.com/krzhu/abc_py)
- [yosys](https://github.com/YosysHQ/yosys)

## Data Preparation
Before running the tasks, ensure your datasets are properly prepared and located in the designated directories. Datas can be found at [https://jbox.sjtu.edu.cn/l/01O2vH](https://jbox.sjtu.edu.cn/l/01O2vH).

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
   python -m src.task1.train [arguments]
   ```

    Additional parameters can be specified to customize the training process:

    + --**data_dir**: Directory containing the data files.
    + --**max_train_samples**: Maximum number of training samples to load.
    + --**batch_size**: Batch size for training and evaluation.
    + --**num_workers**: Number of workers for data loading.
    + --**hidden_channels**: Number of hidden channels in the GCN.
    + --**learning_rate**: Learning rate for the optimizer.
    + --**weight_decay**: Weight decay for the optimizer.
    + --**num_epochs**: Number of epochs to train.
    + --**best_model_path**: Path to save the best model.
    + --**project_name**: Project name for wandb logging.



## Task 2: Logic synthesis decision
In this task, we aim to predict the logic synthesis decision based on the input AIGs. 

### Predict future rewards
First we train a GNN network to predict the future rewards of the logic synthesis process. The rewards are calculated based on the evaluation metrics of the AIGs.

Similar to Task 1, we first read and cache the data, then split the data into train and test sets. Finally, we train the model to predict the future rewards. The execution commands are as follows:

```bash
python -m src.task2.read
python -m src.task2.split
python -m src.task2.train_reward [arguments]
```

The arguments are similar to Task 1, please refer to the previous section for more details.

### Search for the best decision
[TODO]


## Task 3: High level synthesis with large language model
In this task, we introduce the prompting process utilized for generating Verilog code using Large Language Models (LLMs). Two prompting techniques are employed: naive prompting and in-context learning.

### Naive Prompting
Naive prompting involves providing task descriptions directly to LLMs to generate corresponding Verilog code. Prompts outline the task scope, granting LLMs creative freedom. Sample prompts include:

> + **Row Multiplication**: Write a Verilog code that generates the results of row multiplication.
> + **Finite State Machine**: Write a Verilog code of a finite state machine that encompasses more
than 2 stages.
> + **Memory Unit Design**: Write a Verilog code to generate a memory-like storage structure

### In Context Learning
In Context Learning entails dynamic model adaptation based on task-specific context. Initial solutions, generated without parameter fine-tuning, are iteratively refined using original samples as prompts. The iterative refinement process involves:

> + **First Prompting**: Write a Verilog code that generates the results of row multiplication.
> + **Second Prompting**: I have a generated row multiplication code as follows *{last-version-of-code}*, refine it by changing the dimension of matrix A and B into 5*5 
> + **Third Prompting**: I have a generated row multiplication code as follows *{last-version-of-code}*, refine it by adding another matrix.
> + **Fourth Prompting**: I have a generated row multiplication code as follows *{last-version-of-code}*, refine it by adding error handling to the original code.
> + **Fifth Prompting**: I have a generated row multiplication code as follows *{last-version-of-code}*, refine it by adding test benches.