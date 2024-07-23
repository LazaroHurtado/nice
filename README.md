
Implementation of the [NICE](https://arxiv.org/pdf/1410.8516) normalization flow paper in modern PyTorch.

## Usage

```zsh
>> python3 -m venv venv
>> pip3 install -r requirements.txt
>> torchrun --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:12345 main.py
```

If you want to run on multiple GPUs, you can increase the `--nproc_per_node` argument.

Log-likelihood loss of about -1.8k should be reached around 200 epochs and divergence should occur for validation loss afterwords.