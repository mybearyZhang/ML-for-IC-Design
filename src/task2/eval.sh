if [ "$1" = "greedy" ] || [ "$1" = "astar" ]; then
    model_path="/root/ML-for-IC-Design/models/best_model.pth"
else
    model_path="/root/ML-for-IC-Design/src/task2/bcq/models/BCQ_IC-Design_0_200000"
fi
aig_args=("alu4" "apex1" "apex2" "apex4" "b9" "bar" "c880" "c7552" "cavlc" "div" "i9" "m4" "max1024" "mem_ctrl" "pair" "prom1" "router" "sqrt" "square" "voter")

for arg in "${aig_args[@]}"
do
    CUDA_VISIBLE_DEVICES=1 python main.py --aig "$arg" --method "$1" --model_path "$model_path"
done