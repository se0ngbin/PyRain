# Dev Notes
## TODOs
### Implement Denormalizer
Uncommenting the following line in the collate_fn in utils.py returns an error:
File "/home/seongbin/PyRain/src/rain_forecast/utils/metrics.py", line 111, in lat_weighted_rmse
    pred = transform(pred)
TypeError: 'dict' object is not callable

We need transform=self.denormalizer and implement the denormalizer

### Keep Getting Out of Memory Error
Things I tried:
- Reduce batch size
- Set precision to 16
- Set num_workers to 0
Helpful link: [here](https://towardsdatascience.com/i-am-so-done-with-cuda-out-of-memory-c62f42947dca)

#### Mitigations
- reduce timestamps in batch
- keep number of timestamps but use multiple GPUs
- in pytorch lightning, there are ways to reduce memory usage ([link](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html)) - deep speed

## How to run
`python3 src/rain_forecast/run_benchmark.py --sources simsat --inc_time --config_file config.yml --use_amp --gpus 2`


## 8/22
- need to fix some `partition_conf` issues
- I don't think denormalizer is implemented correctly
- everything else seems to be working


## 8/23
- figure out what multiple GPUs do in collect_outputs
- figure out how to use the average of multiple GPUs instead of the last one


## 8/28
- had to run `export NCCL_P2P_DISABLE=1` for multi gpu training to work
- I MUST use GPU 0 for some reason

python3 src/rain_forecast/run_benchmark.py --sources simsat_era --config_file config3.yml --sample_freq 6 --batch_size 2 --use_amp --version test7 --gpus 7; python3 src/rain_forecast/run_benchmark.py --sources simsat_era --config_file config.yml --sample_freq 6 --batch_size 2 --use_amp --version test8 --gpus 7;"


## 8/31
- learning rate + batch size
- ordering of the input data


## 9/7
screen -dmSL scr bash -c "python3 src/rain_forecast/run_benchmark.py --inc_time --config_file config.yml --use_amp --gpus 8 --acc_grad 1 --version ntest7; python3 src/rain_forecast/run_benchmark.py --inc_time --config_file config2.yml --use_amp --gpus 8 --acc_grad 1 --version ntest10; python3 src/rain_forecast/run_benchmark.py --inc_time --config_file config.yml --use_amp --gpus 8 --acc_grad 2 --version ntest8; python3 src/rain_forecast/run_benchmark.py --inc_time --config_file config2.yml --use_amp --gpus 8 --acc_grad 2 --version ntest11; python3 src/rain_forecast/run_benchmark.py --inc_time --config_file config.yml --use_amp --gpus 8 --acc_grad 4 --version ntest9; python3 src/rain_forecast/run_benchmark.py --inc_time --config_file config2.yml --use_amp --gpus 8 --acc_grad 4 --version ntest12;"
- fixed metric to lat weighted rmse


screen -dmSL scr bash -c "python3 src/rain_forecast/run_benchmark.py --inc_time --config_file config.yml --use_amp --gpus 8 --acc_grad 2 --version test1; python3 src/rain_forecast/run_benchmark.py --inc_time --config_file config2.yml --use_amp --gpus 8 --acc_grad 2 --version test2;"


Removing key net.head.0.weight from pretrained checkpoint.
Removing key net.head.0.bias from pretrained checkpoint.
Removing key net.head.2.weight from pretrained checkpoint.
Removing key net.head.2.bias from pretrained checkpoint.
Removing key net.head.4.weight from pretrained checkpoint.
Removing key net.head.4.bias from pretrained checkpoint.
Removing key net.token_embeds.27.proj.weight from pretrained checkpoint
Removing key net.token_embeds.27.proj.bias from pretrained checkpoint
Removing key net.token_embeds.28.proj.weight from pretrained checkpoint
Removing key net.token_embeds.28.proj.bias from pretrained checkpoint
Removing key net.token_embeds.29.proj.weight from pretrained checkpoint
Removing key net.token_embeds.29.proj.bias from pretrained checkpoint
Removing key net.token_embeds.30.proj.weight from pretrained checkpoint
Removing key net.token_embeds.30.proj.bias from pretrained checkpoint
Removing key net.token_embeds.31.proj.weight from pretrained checkpoint
Removing key net.token_embeds.31.proj.bias from pretrained checkpoint
Removing key net.token_embeds.32.proj.weight from pretrained checkpoint
Removing key net.token_embeds.32.proj.bias from pretrained checkpoint
Removing key net.token_embeds.33.proj.weight from pretrained checkpoint
Removing key net.token_embeds.33.proj.bias from pretrained checkpoint
Removing key net.token_embeds.34.proj.weight from pretrained checkpoint
Removing key net.token_embeds.34.proj.bias from pretrained checkpoint
Removing key net.token_embeds.35.proj.weight from pretrained checkpoint
Removing key net.token_embeds.35.proj.bias from pretrained checkpoint
Removing key net.token_embeds.36.proj.weight from pretrained checkpoint
Removing key net.token_embeds.36.proj.bias from pretrained checkpoint
Removing key net.token_embeds.37.proj.weight from pretrained checkpoint
Removing key net.token_embeds.37.proj.bias from pretrained checkpoint
Removing key net.token_embeds.38.proj.weight from pretrained checkpoint
Removing key net.token_embeds.38.proj.bias from pretrained checkpoint
Removing key net.token_embeds.39.proj.weight from pretrained checkpoint
Removing key net.token_embeds.39.proj.bias from pretrained checkpoint
Removing key net.token_embeds.40.proj.weight from pretrained checkpoint
Removing key net.token_embeds.40.proj.bias from pretrained checkpoint
Removing key net.token_embeds.41.proj.weight from pretrained checkpoint
Removing key net.token_embeds.41.proj.bias from pretrained checkpoint
Removing key net.token_embeds.42.proj.weight from pretrained checkpoint
Removing key net.token_embeds.42.proj.bias from pretrained checkpoint
Removing key net.token_embeds.43.proj.weight from pretrained checkpoint
Removing key net.token_embeds.43.proj.bias from pretrained checkpoint
Removing key net.token_embeds.44.proj.weight from pretrained checkpoint
Removing key net.token_embeds.44.proj.bias from pretrained checkpoint
Removing key net.token_embeds.45.proj.weight from pretrained checkpoint
Removing key net.token_embeds.45.proj.bias from pretrained checkpoint
Removing key net.token_embeds.46.proj.weight from pretrained checkpoint
Removing key net.token_embeds.46.proj.bias from pretrained checkpoint
Removing key net.token_embeds.47.proj.weight from pretrained checkpoint
Removing key net.token_embeds.47.proj.bias from pretrained checkpoint
Removing key net.var_embed from pretrained checkpoint
_IncompatibleKeys(missing_keys=['net.var_embed', 'net.time_pos_embed', 'net.time_query', 'net.head.0.weight', 'net.head.0.bias', 'net.head.2.weight', 'net.head.2.bias', 'net.head.4.weight', 'net.head.4.bias', 'net.time_agg.in_proj_weight', 'net.time_agg.in_proj_bias', 'net.time_agg.out_proj.weight', 'net.time_agg.out_proj.bias'], unexpected_keys=[])