#include <cuda.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <iostream>
using namespace std;
using namespace chrono;

void initData(vector<float>& container)
{
	int count = 0;
	for (vector<int>::size_type i = 0; i < container.size(); i++) {
		container[i] = count;
		count++;
	}
}

void initDataOne(vector<float>& container)
{
	int count = 1;
	for (vector<int>::size_type i = 0; i < container.size(); i++) {
		container[i] = count;
	}
}

void checkCUDNN(cudnnStatus_t status)
{
	if (status != CUDNN_STATUS_SUCCESS)
		std::cout << "[ERROR] CUDNN " << status << std::endl;
}

void checkCUDA(cudaError_t error)
{
	if (error != CUDA_SUCCESS)
		std::cout << "[ERROR] CUDA " << error << std::endl;
}

void print_algorithm_name(int algo) {

	switch (algo) {
	case 0: 
		std::cout << "[IMPLICIT_GEMM]" << std::endl;
		break; 
	case 1: 
		std::cout << "[IMPLICIT_PRECOMP_GEMM]" << std::endl;
		break; 
	case 2: 
		std::cout << "[GEMM]" << std::endl;
		break; 
	case 3: 
		std::cout << "[DIRECT]" << std::endl;
		break; 
	case 4: 
		std::cout << "[FFT]" << std::endl;
		break; 
	case 5:
		std::cout << "[FFT_TILING]" << std::endl;
		break;
	case 6:
		std::cout << "[WINOGRAD]" << std::endl;
		break;
	case 7:
		std::cout << "[WINOGRAD_NONFUSED]" << std::endl;
		break;
	default: 
		std::cout << "[Unknown]"; 
		break; 
	}
}

int main() {

	int iteration = 1000;
	int N = 1;
	int C = 3; 
	int H = 224; 
	int W = 224;
	int K = 32;
	int KH = 3;
	int KW = 3; 
	int SH = 1;
	int SW = 1;
	int left = 0;
	int right = 0;
	int top = 0;
	int bottom = 0;
	int P = (H - KH + top + bottom) / SH + 1;
	int Q = (W - KW + left + right) / SW + 1;

	vector<float> data(N * C * H * W);		// input	[N,C,H,W]
	vector<float> weight(K * C * KH* KW);	// weight	[K,C,KH,KW]
	vector<float> bias(K);					// bias		[K]
	vector<float> output_cudnn(N*K*P*Q);	// output	[N,K,P,Q]

	initData(data);		//  1씩 증가하는 등차수열 
	initDataOne(weight);
	initDataOne(bias);

	cudaSetDevice(0); // 특정 GPU Device를 위한 handle을 생성
	cudaStream_t stream;
	cudnnHandle_t cudnn;
	checkCUDA(cudaStreamCreate(&stream));
	checkCUDNN(cudnnCreate(&cudnn));
	checkCUDNN(cudnnSetStream(cudnn, stream));

	float* d_data; // device input data
	checkCUDA(cudaMalloc(&d_data, data.size() * sizeof(float)));
	checkCUDA(cudaMemcpy(d_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice));

	float* d_bias; // device bias
	checkCUDA(cudaMalloc(&d_bias, bias.size() * sizeof(float)));
	checkCUDA(cudaMemcpy(d_bias, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice));

	float* d_weight; // device weight
	checkCUDA(cudaMalloc(&d_weight, weight.size() * sizeof(float)));
	checkCUDA(cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice));

	float* d_output_cudnn; // device output data
	checkCUDA(cudaMalloc(&d_output_cudnn, output_cudnn.size() * sizeof(float)));
	
	// input Tensor desc
	cudnnTensorDescriptor_t xdesc;
	checkCUDNN(cudnnCreateTensorDescriptor(&xdesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(xdesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

	// output Tensor desc
	cudnnTensorDescriptor_t ydesc;
	checkCUDNN(cudnnCreateTensorDescriptor(&ydesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(ydesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, K, P, Q));

	// bias Tensor desc 
	cudnnTensorDescriptor_t bias_desc;
	checkCUDNN(cudnnCreateTensorDescriptor(&bias_desc));
	checkCUDNN(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, K, 1, 1));

	// filter Tensor desc
	cudnnFilterDescriptor_t wdesc; // CUDNN_TENSOR_NHWC, CUDNN_TENSOR_NCHW
	checkCUDNN(cudnnCreateFilterDescriptor(&wdesc));
	checkCUDNN(cudnnSetFilter4dDescriptor(wdesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, KH, KW));

	// convolution desc
	cudnnConvolutionDescriptor_t conv_desc;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
	checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc, left, top, SH, SW, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	// activation desc
	cudnnActivationDescriptor_t act_desc;
	checkCUDNN(cudnnCreateActivationDescriptor(&act_desc));
	checkCUDNN(cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0));

	float scale_factor_prod = 1.f, scale_factor_sum = 0.f; //  scaling factors
	for (int al = 0; al < CUDNN_CONVOLUTION_FWD_ALGO_COUNT; al++) 
	{
		print_algorithm_name(al);
		checkCUDA(cudaMemset(d_output_cudnn, 0, output_cudnn.size() * sizeof(float))); // initialize ZERO

		// 해당 연산을 수행하기 위한 최소 workspace 크기
		size_t workspace_size;
		checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, xdesc, wdesc, conv_desc, ydesc, (cudnnConvolutionFwdAlgo_t)al, &workspace_size));
			
		float* d_workspace = nullptr;
		checkCUDA(cudaMalloc(&d_workspace, workspace_size));
			
		uint64_t total_time = 0;
		for (int idx = 0; idx < iteration; idx++) {
			uint64_t start_time = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
			//cudnnConvolutionBiasActivationForward(
			//	cudnn,
			//	&scale_factor_prod, xdesc, d_data,
			//	wdesc, d_weight,
			//	conv_desc, (cudnnConvolutionFwdAlgo_t)al,
			//	d_workspace, workspace_size,
			//	&scale_factor_sum, ydesc, d_output_cudnn,
			//	bias_desc, d_bias,
			//	act_desc,
			//	ydesc, d_output_cudnn);
				
			cudnnConvolutionForward(
				/*cudnnHandle_t handle,*/cudnn,
				/*const void *alpha,*/&scale_factor_prod,
				/*const cudnnTensorDescriptor_t xDesc,*/xdesc,
				/*const void *x,*/ d_data,
				/*const cudnnFilterDescriptor_t wDesc,*/wdesc, 
				/*const void *w,*/d_weight,
				/*const cudnnConvolutionDescriptor_t convDesc,*/conv_desc, 
				/*cudnnConvolutionFwdAlgo_t algo,*/(cudnnConvolutionFwdAlgo_t)al,
				/*void *workSpace,*/d_workspace,
				/*size_t workSpaceSizeInBytes,*/ workspace_size,
				/*const void *beta,*/&scale_factor_sum,
				/*const cudnnTensorDescriptor_t yDesc,*/ ydesc,
				/*void *y*/ d_output_cudnn
				);
				
			checkCUDA(cudaStreamSynchronize(stream));
			total_time += duration_cast<microseconds>(system_clock::now().time_since_epoch()).count() - start_time;
		}
		checkCUDA(cudaMemcpy(output_cudnn.data(), d_output_cudnn, output_cudnn.size() * sizeof(float), cudaMemcpyDeviceToHost));
		double checksum = 0;
		for (auto d : output_cudnn) checksum += abs((double)d);
		printf("cuDNN(%d/%d) avg_dur_time=%6.3f[msec] checksum=%.6f\n\n", al, (int)CUDNN_CONVOLUTION_FWD_ALGO_COUNT,  total_time / 1000.f / iteration, checksum);			
		checkCUDA(cudaFree(d_workspace));
	}

	checkCUDA(cudaFree(d_output_cudnn));
	checkCUDA(cudaFree(d_weight));	
	checkCUDA(cudaFree(d_data));
	checkCUDA(cudaFree(d_bias));
	checkCUDNN(cudnnDestroy(cudnn));
	checkCUDA(cudaStreamDestroy(stream));
	return 0;
}