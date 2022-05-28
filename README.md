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

<table border="0"  width="100%">
	<tbody align="center">
		<tr>
			<td></td>
			<td><strong>IMPLICIT_GEMM</strong></td>
            <td><strong>IMPLICIT_PRECOMP_GEMM</strong></td>
            <td><strong>GEMM</strong></td>
            <td><strong>FFT</strong></td>
            <td><strong>FFT_TILING</strong></td>
            <td><strong>WINOGRAD</strong></td>
            <td><strong>WINOGRAD_NONFUSED</strong></td>
		</tr>
		<tr>
			<td>Avg Duration time [ms]</td>
			<td>1.612 ms</td>
			<td>0.110 ms </td>
			<td>0.135 ms</td>
			<td>1.655 ms</td>
			<td>1.305 ms</td>
			<td>0.105 ms</td>
			<td>2.868 ms</td>
		</tr>
	</tbody>
</table>

***