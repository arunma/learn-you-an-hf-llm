# runs/h100_tinystories

Training pipeline for the model in [`BLOG.md`](BLOG.md). 91M params, ~20 min on a rented H100, ~$1.50.

On a RunPod H100:

```bash
git clone https://github.com/arunma/learn-you-an-hf-llm.git
cd learn-you-an-hf-llm
pip install -e .
cd runs/h100_tinystories

python prepare_data.py
tmux new -s train
python train_d12.py | tee train.log
```

Detach with `Ctrl-b d`. When training finishes, from your laptop:

```bash
./runs/h100_tinystories/pull_artifacts.sh <pod-ip> <pod-port>
python runs/h100_tinystories/infer.py
```

Full setup (SSH, cleanup, etc.): [`setup_runpod.md`](setup_runpod.md).
