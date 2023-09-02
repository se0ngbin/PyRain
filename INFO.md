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
`python3 src/rain_forecast/run_benchmark.py --sources simsat --inc_time --config_file config.yml --gpus 2`


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

screen -dmSL scr bash -c "python3 src/rain_forecast/run_benchmark.py --config_file config.yml --use_amp --version itest3 --gpus 7 --imerg; python3 src/rain_forecast/run_benchmark.py --config_file config2.yml --use_amp --version itest4 --gpus 7 --imerg; python3 src/rain_forecast/run_benchmark.py --config_file config3.yml --use_amp --version itest5 --gpus 7 --imerg; python3 src/rain_forecast/run_benchmark.py --config_file config4.yml --use_amp --version itest6 --gpus 7 --imerg;"

python3 src/rain_forecast/run_benchmark.py --sources simsat_era --config_file config3.yml --sample_freq 6 --batch_size 2 --use_amp --version test7 --gpus 7; python3 src/rain_forecast/run_benchmark.py --sources simsat_era --config_file config4.yml --sample_freq 6 --batch_size 2 --use_amp --version test8 --gpus 7;"


## 8/31
- learning rate + batch size
- ordering of the input data


_IncompatibleKeys(missing_keys=['net.var_embed', 'net.time_pos_embed', 'net.time_query', 'net.token_embeds.0.proj.weight', 'net.token_embeds.0.proj.bias', 'net.token_embeds.1.proj.weight', 'net.token_embeds.1.proj.bias', 'net.token_embeds.2.proj.weight', 'net.token_embeds.2.proj.bias', 'net.token_embeds.3.proj.weight', 'net.token_embeds.3.proj.bias', 'net.token_embeds.4.proj.weight', 'net.token_embeds.4.proj.bias', 'net.token_embeds.5.proj.weight', 'net.token_embeds.5.proj.bias', 'net.token_embeds.6.proj.weight', 'net.token_embeds.6.proj.bias', 'net.token_embeds.7.proj.weight', 'net.token_embeds.7.proj.bias', 'net.token_embeds.8.proj.weight', 'net.token_embeds.8.proj.bias', 'net.token_embeds.9.proj.weight', 'net.token_embeds.9.proj.bias', 'net.token_embeds.10.proj.weight', 'net.token_embeds.10.proj.bias', 'net.token_embeds.11.proj.weight', 'net.token_embeds.11.proj.bias', 'net.token_embeds.12.proj.weight', 'net.token_embeds.12.proj.bias', 'net.token_embeds.13.proj.weight', 'net.token_embeds.13.proj.bias', 'net.token_embeds.14.proj.weight', 'net.token_embeds.14.proj.bias', 'net.token_embeds.15.proj.weight', 'net.token_embeds.15.proj.bias', 'net.token_embeds.16.proj.weight', 'net.token_embeds.16.proj.bias', 'net.token_embeds.17.proj.weight', 'net.token_embeds.17.proj.bias', 'net.token_embeds.18.proj.weight', 'net.token_embeds.18.proj.bias', 'net.token_embeds.19.proj.weight', 'net.token_embeds.19.proj.bias', 'net.token_embeds.20.proj.weight', 'net.token_embeds.20.proj.bias', 'net.token_embeds.21.proj.weight', 'net.token_embeds.21.proj.bias', 'net.token_embeds.22.proj.weight', 'net.token_embeds.22.proj.bias', 'net.token_embeds.23.proj.weight', 'net.token_embeds.23.proj.bias', 'net.token_embeds.24.proj.weight', 'net.token_embeds.24.proj.bias', 'net.token_embeds.25.proj.weight', 'net.token_embeds.25.proj.bias', 'net.token_embeds.26.proj.weight', 'net.token_embeds.26.proj.bias', 'net.head.0.weight', 'net.head.0.bias', 'net.head.2.weight', 'net.head.2.bias', 'net.head.4.weight', 'net.head.4.bias', 'net.time_agg.in_proj_weight', 'net.time_agg.in_proj_bias', 'net.time_agg.out_proj.weight', 'net.time_agg.out_proj.bias'], unexpected_keys=[])

_IncompatibleKeys(missing_keys=['net.var_embed', 'net.time_pos_embed', 'net.time_query', 'net.token_embeds.0.proj.weight', 'net.token_embeds.0.proj.bias', 'net.token_embeds.1.proj.weight', 'net.token_embeds.1.proj.bias', 'net.token_embeds.2.proj.weight', 'net.token_embeds.2.proj.bias', 'net.token_embeds.3.proj.weight', 'net.token_embeds.3.proj.bias', 'net.token_embeds.4.proj.weight', 'net.token_embeds.4.proj.bias', 'net.token_embeds.5.proj.weight', 'net.token_embeds.5.proj.bias', 'net.token_embeds.6.proj.weight', 'net.token_embeds.6.proj.bias', 'net.token_embeds.7.proj.weight', 'net.token_embeds.7.proj.bias', 'net.token_embeds.8.proj.weight', 'net.token_embeds.8.proj.bias', 'net.token_embeds.9.proj.weight', 'net.token_embeds.9.proj.bias', 'net.token_embeds.10.proj.weight', 'net.token_embeds.10.proj.bias', 'net.token_embeds.11.proj.weight', 'net.token_embeds.11.proj.bias', 'net.token_embeds.12.proj.weight', 'net.token_embeds.12.proj.bias', 'net.token_embeds.13.proj.weight', 'net.token_embeds.13.proj.bias', 'net.token_embeds.14.proj.weight', 'net.token_embeds.14.proj.bias', 'net.token_embeds.15.proj.weight', 'net.token_embeds.15.proj.bias', 'net.token_embeds.16.proj.weight', 'net.token_embeds.16.proj.bias', 'net.token_embeds.17.proj.weight', 'net.token_embeds.17.proj.bias', 'net.token_embeds.18.proj.weight', 'net.token_embeds.18.proj.bias', 'net.token_embeds.19.proj.weight', 'net.token_embeds.19.proj.bias', 'net.token_embeds.20.proj.weight', 'net.token_embeds.20.proj.bias', 'net.token_embeds.21.proj.weight', 'net.token_embeds.21.proj.bias', 'net.token_embeds.22.proj.weight', 'net.token_embeds.22.proj.bias', 'net.token_embeds.23.proj.weight', 'net.token_embeds.23.proj.bias', 'net.token_embeds.24.proj.weight', 'net.token_embeds.24.proj.bias', 'net.token_embeds.25.proj.weight', 'net.token_embeds.25.proj.bias', 'net.token_embeds.26.proj.weight', 'net.token_embeds.26.proj.bias', 'net.head.0.weight', 'net.head.0.bias', 'net.head.2.weight', 'net.head.2.bias', 'net.head.4.weight', 'net.head.4.bias', 'net.time_agg.in_proj_weight', 'net.time_agg.in_proj_bias', 'net.time_agg.out_proj.weight', 'net.time_agg.out_proj.bias'], unexpected_keys=[])

Keys in checkpoint:
net.pos_embed
net.blocks.0.norm1.weight
net.blocks.0.norm1.bias
net.blocks.0.attn.qkv.weight
net.blocks.0.attn.qkv.bias
net.blocks.0.attn.proj.weight
net.blocks.0.attn.proj.bias
net.blocks.0.norm2.weight
net.blocks.0.norm2.bias
net.blocks.0.mlp.fc1.weight
net.blocks.0.mlp.fc1.bias
net.blocks.0.mlp.fc2.weight
net.blocks.0.mlp.fc2.bias
net.blocks.1.norm1.weight
net.blocks.1.norm1.bias
net.blocks.1.attn.qkv.weight
net.blocks.1.attn.qkv.bias
net.blocks.1.attn.proj.weight
net.blocks.1.attn.proj.bias
net.blocks.1.norm2.weight
net.blocks.1.norm2.bias
net.blocks.1.mlp.fc1.weight
net.blocks.1.mlp.fc1.bias
net.blocks.1.mlp.fc2.weight
net.blocks.1.mlp.fc2.bias
net.blocks.2.norm1.weight
net.blocks.2.norm1.bias
net.blocks.2.attn.qkv.weight
net.blocks.2.attn.qkv.bias
net.blocks.2.attn.proj.weight
net.blocks.2.attn.proj.bias
net.blocks.2.norm2.weight
net.blocks.2.norm2.bias
net.blocks.2.mlp.fc1.weight
net.blocks.2.mlp.fc1.bias
net.blocks.2.mlp.fc2.weight
net.blocks.2.mlp.fc2.bias
net.blocks.3.norm1.weight
net.blocks.3.norm1.bias
net.blocks.3.attn.qkv.weight
net.blocks.3.attn.qkv.bias
net.blocks.3.attn.proj.weight
net.blocks.3.attn.proj.bias
net.blocks.3.norm2.weight
net.blocks.3.norm2.bias
net.blocks.3.mlp.fc1.weight
net.blocks.3.mlp.fc1.bias
net.blocks.3.mlp.fc2.weight
net.blocks.3.mlp.fc2.bias
net.blocks.4.norm1.weight
net.blocks.4.norm1.bias
net.blocks.4.attn.qkv.weight
net.blocks.4.attn.qkv.bias
net.blocks.4.attn.proj.weight
net.blocks.4.attn.proj.bias
net.blocks.4.norm2.weight
net.blocks.4.norm2.bias
net.blocks.4.mlp.fc1.weight
net.blocks.4.mlp.fc1.bias
net.blocks.4.mlp.fc2.weight
net.blocks.4.mlp.fc2.bias
net.blocks.5.norm1.weight
net.blocks.5.norm1.bias
net.blocks.5.attn.qkv.weight
net.blocks.5.attn.qkv.bias
net.blocks.5.attn.proj.weight
net.blocks.5.attn.proj.bias
net.blocks.5.norm2.weight
net.blocks.5.norm2.bias
net.blocks.5.mlp.fc1.weight
net.blocks.5.mlp.fc1.bias
net.blocks.5.mlp.fc2.weight
net.blocks.5.mlp.fc2.bias
net.blocks.6.norm1.weight
net.blocks.6.norm1.bias
net.blocks.6.attn.qkv.weight
net.blocks.6.attn.qkv.bias
net.blocks.6.attn.proj.weight
net.blocks.6.attn.proj.bias
net.blocks.6.norm2.weight
net.blocks.6.norm2.bias
net.blocks.6.mlp.fc1.weight
net.blocks.6.mlp.fc1.bias
net.blocks.6.mlp.fc2.weight
net.blocks.6.mlp.fc2.bias
net.blocks.7.norm1.weight
net.blocks.7.norm1.bias
net.blocks.7.attn.qkv.weight
net.blocks.7.attn.qkv.bias
net.blocks.7.attn.proj.weight
net.blocks.7.attn.proj.bias
net.blocks.7.norm2.weight
net.blocks.7.norm2.bias
net.blocks.7.mlp.fc1.weight
net.blocks.7.mlp.fc1.bias
net.blocks.7.mlp.fc2.weight
net.blocks.7.mlp.fc2.bias
net.norm.weight
net.norm.bias
net.lead_time_embed.weight
net.lead_time_embed.bias
net.var_query
net.var_agg.in_proj_weight
net.var_agg.in_proj_bias
net.var_agg.out_proj.weight
net.var_agg.out_proj.bias

Keys in model:
net.var_embed
net.var_query
net.pos_embed
net.time_pos_embed
net.time_query
net.token_embeds.0.proj.weight
net.token_embeds.0.proj.bias
net.token_embeds.1.proj.weight
net.token_embeds.1.proj.bias
net.token_embeds.2.proj.weight
net.token_embeds.2.proj.bias
net.token_embeds.3.proj.weight
net.token_embeds.3.proj.bias
net.token_embeds.4.proj.weight
net.token_embeds.4.proj.bias
net.token_embeds.5.proj.weight
net.token_embeds.5.proj.bias
net.token_embeds.6.proj.weight
net.token_embeds.6.proj.bias
net.token_embeds.7.proj.weight
net.token_embeds.7.proj.bias
net.token_embeds.8.proj.weight
net.token_embeds.8.proj.bias
net.token_embeds.9.proj.weight
net.token_embeds.9.proj.bias
net.token_embeds.10.proj.weight
net.token_embeds.10.proj.bias
net.token_embeds.11.proj.weight
net.token_embeds.11.proj.bias
net.token_embeds.12.proj.weight
net.token_embeds.12.proj.bias
net.token_embeds.13.proj.weight
net.token_embeds.13.proj.bias
net.token_embeds.14.proj.weight
net.token_embeds.14.proj.bias
net.token_embeds.15.proj.weight
net.token_embeds.15.proj.bias
net.token_embeds.16.proj.weight
net.token_embeds.16.proj.bias
net.token_embeds.17.proj.weight
net.token_embeds.17.proj.bias
net.token_embeds.18.proj.weight
net.token_embeds.18.proj.bias
net.token_embeds.19.proj.weight
net.token_embeds.19.proj.bias
net.token_embeds.20.proj.weight
net.token_embeds.20.proj.bias
net.token_embeds.21.proj.weight
net.token_embeds.21.proj.bias
net.token_embeds.22.proj.weight
net.token_embeds.22.proj.bias
net.token_embeds.23.proj.weight
net.token_embeds.23.proj.bias
net.token_embeds.24.proj.weight
net.token_embeds.24.proj.bias
net.token_embeds.25.proj.weight
net.token_embeds.25.proj.bias
net.token_embeds.26.proj.weight
net.token_embeds.26.proj.bias
net.var_agg.in_proj_weight
net.var_agg.in_proj_bias
net.var_agg.out_proj.weight
net.var_agg.out_proj.bias
net.lead_time_embed.weight
net.lead_time_embed.bias
net.blocks.0.norm1.weight
net.blocks.0.norm1.bias
net.blocks.0.attn.qkv.weight
net.blocks.0.attn.qkv.bias
net.blocks.0.attn.proj.weight
net.blocks.0.attn.proj.bias
net.blocks.0.norm2.weight
net.blocks.0.norm2.bias
net.blocks.0.mlp.fc1.weight
net.blocks.0.mlp.fc1.bias
net.blocks.0.mlp.fc2.weight
net.blocks.0.mlp.fc2.bias
net.blocks.1.norm1.weight
net.blocks.1.norm1.bias
net.blocks.1.attn.qkv.weight
net.blocks.1.attn.qkv.bias
net.blocks.1.attn.proj.weight
net.blocks.1.attn.proj.bias
net.blocks.1.norm2.weight
net.blocks.1.norm2.bias
net.blocks.1.mlp.fc1.weight
net.blocks.1.mlp.fc1.bias
net.blocks.1.mlp.fc2.weight
net.blocks.1.mlp.fc2.bias
net.blocks.2.norm1.weight
net.blocks.2.norm1.bias
net.blocks.2.attn.qkv.weight
net.blocks.2.attn.qkv.bias
net.blocks.2.attn.proj.weight
net.blocks.2.attn.proj.bias
net.blocks.2.norm2.weight
net.blocks.2.norm2.bias
net.blocks.2.mlp.fc1.weight
net.blocks.2.mlp.fc1.bias
net.blocks.2.mlp.fc2.weight
net.blocks.2.mlp.fc2.bias
net.blocks.3.norm1.weight
net.blocks.3.norm1.bias
net.blocks.3.attn.qkv.weight
net.blocks.3.attn.qkv.bias
net.blocks.3.attn.proj.weight
net.blocks.3.attn.proj.bias
net.blocks.3.norm2.weight
net.blocks.3.norm2.bias
net.blocks.3.mlp.fc1.weight
net.blocks.3.mlp.fc1.bias
net.blocks.3.mlp.fc2.weight
net.blocks.3.mlp.fc2.bias
net.blocks.4.norm1.weight
net.blocks.4.norm1.bias
net.blocks.4.attn.qkv.weight
net.blocks.4.attn.qkv.bias
net.blocks.4.attn.proj.weight
net.blocks.4.attn.proj.bias
net.blocks.4.norm2.weight
net.blocks.4.norm2.bias
net.blocks.4.mlp.fc1.weight
net.blocks.4.mlp.fc1.bias
net.blocks.4.mlp.fc2.weight
net.blocks.4.mlp.fc2.bias
net.blocks.5.norm1.weight
net.blocks.5.norm1.bias
net.blocks.5.attn.qkv.weight
net.blocks.5.attn.qkv.bias
net.blocks.5.attn.proj.weight
net.blocks.5.attn.proj.bias
net.blocks.5.norm2.weight
net.blocks.5.norm2.bias
net.blocks.5.mlp.fc1.weight
net.blocks.5.mlp.fc1.bias
net.blocks.5.mlp.fc2.weight
net.blocks.5.mlp.fc2.bias
net.blocks.6.norm1.weight
net.blocks.6.norm1.bias
net.blocks.6.attn.qkv.weight
net.blocks.6.attn.qkv.bias
net.blocks.6.attn.proj.weight
net.blocks.6.attn.proj.bias
net.blocks.6.norm2.weight
net.blocks.6.norm2.bias
net.blocks.6.mlp.fc1.weight
net.blocks.6.mlp.fc1.bias
net.blocks.6.mlp.fc2.weight
net.blocks.6.mlp.fc2.bias
net.blocks.7.norm1.weight
net.blocks.7.norm1.bias
net.blocks.7.attn.qkv.weight
net.blocks.7.attn.qkv.bias
net.blocks.7.attn.proj.weight
net.blocks.7.attn.proj.bias
net.blocks.7.norm2.weight
net.blocks.7.norm2.bias
net.blocks.7.mlp.fc1.weight
net.blocks.7.mlp.fc1.bias
net.blocks.7.mlp.fc2.weight
net.blocks.7.mlp.fc2.bias
net.norm.weight
net.norm.bias
net.head.0.weight
net.head.0.bias
net.head.2.weight
net.head.2.bias
net.head.4.weight
net.head.4.bias
net.time_agg.in_proj_weight
net.time_agg.in_proj_bias
net.time_agg.out_proj.weight
net.time_agg.out_proj.bias