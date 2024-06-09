import os
import re

synthesisOpToPosDic = {
    0: "refactor",
    1: "refactor -z",
    2: "rewrite",
    3: "rewrite -z",
    4: "resub",
    5: "resub -z",
    6: "balance"
}
libFile = '/root/ML-for-IC-Design/data/lib/7nm/7nm.lib'

def generate_next_aig(aig_file, action, next_aig_file, logFile='aig.log'):
    abcRunCmd = "yosys-abc -c \" read " + aig_file + "; " + synthesisOpToPosDic[action] + "; read_lib " + libFile + "; write " + next_aig_file + "; print_stats \" > " + logFile
    os.system(abcRunCmd)

def score_aig_baseline(aig_file, next_aig_file='resyn2.aig', logFile='aig.log'):
    RESYN2_CMD = ("balance; rewrite; refactor; balance; rewrite; rewrite -z; "
              "balance; refactor -z; rewrite -z; balance;")
    abcRunCmd = (f"yosys-abc -c \"read {aig_file}; {RESYN2_CMD} "
                f"read_lib {libFile}; write {next_aig_file}; map; topo; stime\" > {logFile}")
    os.system(abcRunCmd)

    with open(logFile) as f:
        areaInformation = re.findall(r'[a-zA-Z0-9.]+', f.readlines()[-1])
        baseline = float(areaInformation[-9]) * float(areaInformation[-4])
    return baseline

def score_aig_adp(aig_file, logFile='aig.log'):
    abcRunCmd = "yosys-abc -c \" read " + aig_file + "; read_lib " + libFile + "; map ; topo; stime \" > " + logFile
    os.system(abcRunCmd)
    with open(logFile) as f:
        arealnformation = re.findall(r'[a-zA-Z0-9.]+', f.readlines()[-1])
        adpVal = float(arealnformation[-9]) * float(arealnformation[-4])
    return adpVal

def score_aig_regularized(aig_file, baseline, logFile='aig.log'):
    adpVal = score_aig_adp(aig_file, logFile)
    return (baseline - adpVal) / baseline
