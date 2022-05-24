# Cudnn_EX


## Environment
- Windows 10
- CUDA 11.1
- TensorRT 8.0.1.6
- Cudnn 8.2.1
***
## Cudnn Convolution APIs
- Cudnn Convolution Algorithm execution time Comparison (1000 iteration)   
- input [1,3,224,224] weight [32,3,3,3] bias [32]   

    - [IMPLICIT_GEMM]   
    avg_dur_time= 1.612[msec]
    
    - [IMPLICIT_PRECOMP_GEMM]   
    avg_dur_time= 0.110 [msec]
    
    - [GEMM]   
    avg_dur_time= 0.135 [msec]
    
    - [FFT]   
    avg_dur_time= 1.655 [msec]
    
    - [FFT_TILING]   
    avg_dur_time= 1.305 [msec]
    
    - [WINOGRAD]   
    avg_dur_time= 0.105 [msec]
    
    - [WINOGRAD_NONFUSED]   
    avg_dur_time= 2.868 [msec]

***