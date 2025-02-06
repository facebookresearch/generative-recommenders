# MLCommons (MLPerf) DLRMv3 Inference Benchmarks

## Build loadgen
```
cd generative_recommenders/dlrm_v3/inference/thirdparty/loadgen/
CFLAGS="-std=c++14" python setup.py develop --user
```

## Download dataset
```
mkdir -p data/ && python preprocess_public_data.py --dataset kuairand-1k
mv data ~/data
```

## Inference benchmark
```
WORLD_SIZE=2 python inference/main.py -- --dataset kuairand-1k
```
The config file is listed in `dlrm_v3/inference/gin/kuairand_1k.gin`. `WORLD_SIZE` is the number of GPUs used in the inference benchmark.

To load checkpoint from training, modify `run.model_path` inside the inference gin config file.

## Run unit tests

```
python tests/inference_test.py
```
