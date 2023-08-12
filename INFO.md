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

