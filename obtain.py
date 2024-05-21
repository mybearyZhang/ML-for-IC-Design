import os

state = 'alu2_0130622'
circuitName, actions = state.split('_')
circuitPath = './InitialAIG/train/' + circuitName + '.aig'
libFile = './lib/7nm/7nm.lib'
logFile = 'alu2.log'
nextState = state + '.aig'  # current AIG file
synthesisOpToPosDic = {
    0: "refactor",
    1: "refactor -z",
    2: "rewrite",
    3: "rewrite -z",
    4: "resub",
    5: "resub -z",
    6: "balance"
}
actionCmd = ''
for action in actions:
    actionCmd += synthesisOpToPosDic[int(action)] + ' ; '
    
abcRunCmd = (
    './yosys/yosys-abc -c "read ' + circuitPath + ' ; ' +
    actionCmd +
    ' read_lib ' + libFile + ' ; write ' + nextState +
    ' ; print_stats" > ' + logFile
)
os.system(abcRunCmd)
