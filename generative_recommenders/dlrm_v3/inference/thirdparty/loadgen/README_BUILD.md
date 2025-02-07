# Building the LoadGen {#ReadmeBuild}

## Prerequisites

    sudo apt-get install libglib2.0-dev python-pip python3-pip
    pip2 install absl-py numpy
    pip3 install absl-py numpy

## Quick Start
### Installation - Python

    pip install absl-py numpy
    git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
    cd mlperf_inference/loadgen
    CFLAGS="-std=c++14 -O3" python -m pip install .

This will fetch the loadgen source, build and install the loadgen as a python module, and run a simple end-to-end demo.

Alternatively, we provide wheels for several python versions and operating system that can be installed using pip directly.

    pip install mlperf-loadgen

**NOTE:** Take into account that we only update the published wheels after an official release, they may not include the latest changes.

### Testing your Installation
The following command will run a simple end-to-end demo:

    python mlperf_inference/loadgen/demos/py_demo_single_stream.py

A summary of the test results can be found in the *"mlperf_log_summary.txt"* logfile.

For a timeline visualization of what happened during the test, open the *"mlperf_log_trace.json"* file in Chrome:
* Type “chrome://tracing” in the address bar, then drag-n-drop the json.
* This may be useful for SUT performance tuning and understanding + debugging the loadgen.

### Installation - C++
To build the loadgen as a C++ library, rather than a python module:

    git clone https://github.com/mlcommons/inference.git mlperf_inference
    cd mlperf_inference
    mkdir loadgen/build/ && cd loadgen/build/
    cmake .. && cmake --build .
    cp libmlperf_loadgen.a ..

## Quick start: Loadgen Over the Network

Refer to [LON demo](demos/lon/README.md) for a basic example.
