# LearningCudaAttention

## Description

Transformers/Foundation models are state-of-the-art models that achieve high accuracy on numerous tasks ranging from Object Detection (DDETR, ViT), NLP (BERT), Segmentation (Segformer), etc…, Originally introduced in “Attention is all you need” by Google Brain

Transformers have the self-attention module which lets the model to compute attention scores to weigh/provide importance to different parts of the input tokens to itself. With more variants of transformers coming out over the years, performance optimization for these new modules/variants is an interesting domain

For the CUDA Coursera course, I will be implementing a simple attention module using CUDA

The attention module comprises of:
- three linear/FC layers are used to compute query, key and value 
- The dot product of query and key is computed to get the similarity which is then scaled and passed into softmax to compute similarity scores in range of 0-1 
- The scores are multiplied with the values to get the output of the attention module which in some networks (ViT) is followed by a projection FC layer

## Key Concepts 

Performance Strategies, Image Processing, NPP Library 

## Supported SM Architectures 

[SM 3.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 3.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes 

Linux 

## Supported CPU Architecture 

x86_64

## CUDA APIs involved 

 ./bin/attention data/input.bin data/query.bin data/key.bin data/value.bin data/out_weight.bin data/out_bias.bin data/output.bin 1024 8 64 1024 1

## Dependencies needed to build/run

## Prerequisites

Download and install the [CUDA Toolkit 11.4](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## Build and Run 

The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:
```
$ cd <sample_dir>
$ make
```
The samples makefiles can take advantage of certain options:
*   **dbg=1** - build with debug symbols
    ```
    $ make dbg=1
    ```
*   **SMS="A B ..."** - override the SM architectures for which the sample will be built, where `"A B ..."` is a space-delimited list of SM architectures. For example, to generate SASS for SM 50 and SM 60, use `SMS="50 60"`.
    ```
    $ make SMS="50 60"
    ```

*  **HOST_COMPILER=<host_compiler>** - override the default g++ host compiler. See the [Linux Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements) for a list of supported host compilers.
```
    $ make HOST_COMPILER=g++
```

## Build and Run - Docker based

Docker offers a way to build an environment to get all your dependencies without having to install everything on baremetal, this way you can experiment with several versions of CUDA for instance. See [NVIDIA container installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

This repo contains a dockerfile which you can use to build a docker image and use to run the program and generate outputs 

```
# Change to cloned repo location
cd $PATH_TO_CLONED_REPO
# Build the container image
docker build -t cuda_attention:latest -f docker/Dockerfile .
# Launch the container (the -v allows us to mount a volume from host to the container allow us to copy files/outputs to/from the docker environment)
docker run --gpus all --rm -it -v /tmp:/tmp cuda_attention:latest

# Once inside container you can run the program as it comes pre-compiled as part of the image
./bin/attention data/input.bin data/query.bin data/key.bin data/value.bin data/out_weight.bin data/out_bias.bin data/output.bin 1024 8 64 1024 1

```

## Usage 

Once you have either the baremetal or the docker based CLI setup you can use the CLI in this way 
```
 Usage ./attention inputs.npy query.bin key.bin value.bin 
        out_w.bin out_b.bin output.bin <dim> <heads> <dim_head> <seq_len> 
        <kernel_type>
```

## Personal Notes 

### Rubric Check

Code Repository 
    - URL: https://github.com/AdityanJo/CourseraCUDAatScaleProject
    - README : Describes how to run and multiple options to run as well as CLI usage details
    - Google Style C++ : Used clang-format-9 to comply with the style format

Proof of execution artifacts:
    - Added log outputs for the outputs that are part of the repo for the example commands shown in this readme

Code Project Description:
    - Added description for the underlying algorithm and underlying NPP algorithm used

## References 

- Template codes taken from https://github.com/PascaleCourseraCourses/CUDAatScaleForTheEnterpriseCourseProjectTemplate
- https://docs.nvidia.com/cuda/cublas/#cublas-t-gemm