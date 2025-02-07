# Demo

## Loadgen Over the Network

### Overview


This folder provides a demo implementation for LoadGen over the network.\
Two sides are implemented:

1. The SUT side which is implemented in [sut_over_network_demo.py](sut_over_network_demo.py). Each Node should run it for multiple Nodes operation.
2. The LoadGen node running the LoadGen, QSL and QDL instances, implemented in [py_demo_server_lon.py](py_demo_server_lon.py)

The demo SUT is implemented with a Flask server. the LON node implements a Flask client for network operation.

The test runs in MLPerf Server mode. the SUT is not implementing a benchmark but contains dummy interface to preprocessing, postprocessing and  model calling functions.

### Setup

Install python packages:

```sh
pip install absl-py numpy wheel flask requests
```

Clone:

```sh
git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
```

Build:

```sh
cd mlperf_inference/loadgen
CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel
cd ..; pip install --force-reinstall loadgen/dist/`ls -r loadgen/dist/ | head -n1` ; cd -
```

### Run the demo (single machine)

Start the demo SUT server (run this at a separate terminal):

```sh
python demos/lon/sut_over_network_demo.py --port 8000
```

Start the test:

```sh
python demos/lon/py_demo_server_lon.py --sut_server http://localhost:8000
```

### Run the demo (over the network)

To run over a network - simply run the demo SUT over on a different machine. For multiple Nodes run the demo SUT on each machine specifying the node number.\

```sh
python demos/lon/sut_over_network_demo.py --port 8000 --node N1
```

Then, when running the client, replace `localhost` with the correct IP.


```sh
python demos/lon/py_demo_server_lon.py --sut_server IP1:8000,IP2:8000,IP3:8000
```
