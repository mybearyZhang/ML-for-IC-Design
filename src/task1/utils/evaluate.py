import os
import re

# Set up the variables (replace these with your actual values)
AIG = "alu2_0130622.aig"
libFile = "./lib/7nm/7nm.lib"
logFile = "./eval.log"

# Construct the command to run yosys with abc
abcRunCmd = f"yosys-abc -c \"read {AIG}; read_lib {libFile}; map; topo; stime\" > {logFile}"

# Execute the command
os.system(abcRunCmd)

# Open the log file and extract area information
with open(logFile) as f:
    lines = f.readlines()
    areaInformation = re.findall(r'[a-zA-Z0-9.]+', lines[-1])

# Calculate the evaluation value
eval = float(areaInformation[-9]) * float(areaInformation[-4])

# Print the evaluation value
print("Evaluation value:", eval)

RESYN2_CMD = ("balance; rewrite; refactor; balance; rewrite; rewrite -z; "
              "balance; refactor -z; rewrite -z; balance;")
state = 'alu2_0130622'
circuitName, actions = state.split('_')
circuitPath = './InitialAIG/train/' + circuitName + '.aig'
libFile = './lib/7nm/7nm.lib'
logFile = 'resyn2.log'
nextState = state + '.aig'  # current AIG file
nextBench = state + '.bench'  # current bench file

# Construct the command to run yosys with abc
abcRunCmd = (f"yosys-abc -c \"read {circuitPath}; {RESYN2_CMD} "
             f"read_lib {libFile}; write {nextState}; write_bench -l {nextBench}; map; topo; stime\" > {logFile}")

# Execute the command
os.system(abcRunCmd)

# Open the log file and extract area information
with open(logFile) as f:
    lines = f.readlines()
    areaInformation = re.findall(r'[a-zA-Z0-9.]+', lines[-1])

# Calculate the baseline and evaluation value
baseline = float(areaInformation[-9]) * float(areaInformation[-4])
eval = 1 - (eval / baseline)

# Print the evaluation value
print("Evaluation value:", eval)
