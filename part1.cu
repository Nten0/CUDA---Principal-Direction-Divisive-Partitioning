#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "file_io.c"

#define BLOCK_SIZE 32
#define TILE_WIDTH 2 

#define cudaCheckError() {                               \
  cudaError_t e = cudaGetLastError();                    \
  if (e != cudaSuccess) {                                \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e));                      \
    exit(1);                                             \
  }                                                      \
}

// Initialize a vector of size m to 1
__global__ void ones(double* vec, int m)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if(row >= m) return;    
	vec[row] = 1.0;
}

// Calculate the value of w = average value of each row
__global__ void w_calc(double* objects, int m, int n, double *w)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row >= n) return;
    
   	double w_value = 0;
   	for (int i = 0; i < m ; i++)
   		w_value += objects[row * m + i];   
   	w[row] = w_value/m;
}

// Calculate tmp = A * xk = ( M - w*e' )*xk
__global__ void pddp1(double* objects, int m, int n, double *w,double *x,double *tmp)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row >= n ) return;

	double Cvalue = 0;
	
	for(int i = 0; i < m; i++)
		Cvalue += (objects[row * m + i] - w[row]) * x [i];
	tmp[row] = Cvalue;

	__syncthreads();
}

// Calculate output = A' * tmp = (M - w*e')' * ( M - w*e' )*xk
__global__ void pddp2(double* objects, double *output, int m, int n, double *w, double *tmp)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row >= m ) return;

	double final_value = 0;

	for(int j=0; j < n; j++)
		final_value += (objects[j*m+row] - w[j]) * tmp[j];
	output[row] = final_value; 
	
	__syncthreads();
}

// Calculate the power of each vector's vaule 
__global__ void power(double *input, int m , double *output)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;	
	if(row >= m ) return;

	output[row] = (double)powf(input[row] , 2);

	__syncthreads();
}

// Divide each element of a vector with a value
__global__ void division(double *input, int m, double norm, double *output)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if(row >= m ) return;

	output[row] = input[row] / norm;

	__syncthreads();
}

// Calculate the difference in order to diverge
__global__ void diff_pow(double *x, int m, double *y)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if(row >= m ) return;

	x[row] = y[row] - x[row];
	x[row] = (double)powf(x[row] , 2);
}

// Copy contents of vector c to vector x
__global__ void swap(double* x, int m, double *c)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if(row >= m) return;   
	x[row] = c[row];
}

void StartKernelTiming (cudaEvent_t& tic, cudaEvent_t& toc, cudaStream_t iStream)
{
	cudaEventCreate(&tic);
	cudaEventCreate(&toc);
	cudaEventRecord(tic , iStream);
}

void StopKernelTiming (cudaEvent_t& tic, cudaEvent_t& toc, cudaStream_t iStream, float* ptimer)
{
	float kt = 0;
	cudaEventRecord(toc , iStream);
	cudaEventSynchronize(toc);
	cudaEventElapsedTime(&kt , tic , toc);
	cudaEventDestroy(tic); cudaEventDestroy(toc);
	(*ptimer) += kt;
}


int main(int argc, char **argv) {

	cudaSetDevice(0);

	int n,m;
	char *input_file = argv[1];
	double *objects; 
	objects = file_read(input_file, &n, &m);
	printf("::Objects loaded::\n");
	printf("Objects: %d\n", m);
	printf("Attributes: %d\n", n);
 
	double fnorm, f_sum, final[m], eps;
	eps = pow(10,-6);
	double *w, *x, *tmp, *den, *y;
	w = (double*) malloc(n*sizeof(double));
	x = (double*) malloc(m*sizeof(double));
	tmp = (double*) malloc(n*sizeof(double));	
	den = (double*) malloc(m*sizeof(double));
	y = (double*)malloc(m*sizeof(double));	

	double *objects_d,*final_d,*w_d, *x_d,*tmp_d, *in_d, *den_d, *y_d;

  	dim3 dimBlock(1,BLOCK_SIZE);
    dim3 dimGrid(1,	(m + dimBlock.y - 1) / dimBlock.y);

    printf("Grid Size: (%d,%d) \n",dimGrid.x,dimGrid.y);
    printf("Block Size: (%d,%d) \n",dimBlock.x,dimBlock.y);
    
	cudaMalloc((void**) &in_d , m*sizeof(double));
	cudaMemcpy(in_d , x , m*sizeof(double) , cudaMemcpyHostToDevice);
  	
  	cudaMalloc((void**) &x_d , m*sizeof(double));
	cudaMemcpy(x_d , x , m*sizeof(double) , cudaMemcpyHostToDevice);
		
	cudaMalloc((void **) &objects_d , m*n*sizeof(double));
	cudaMemcpy (objects_d , objects , m*n*sizeof(double) , cudaMemcpyHostToDevice );

	cudaMalloc((void **) &final_d,m*sizeof(double));
	cudaMalloc((void **) &tmp_d,n*sizeof(double));
	cudaMalloc((void **) &w_d,n*sizeof(double));
	cudaMalloc((void **) &den_d,m*sizeof(double));
	cudaMalloc((void **) &y_d,m*sizeof(double));
	cudaCheckError();

	cudaEvent_t tic, toc;
	float Elapsed_Time;

	StartKernelTiming(tic, toc, 0);
	ones<<<dimGrid,dimBlock>>>(x_d,m);
	StopKernelTiming(tic, toc, 0, &Elapsed_Time);
	cudaMemcpy (x, x_d, m*sizeof(double), cudaMemcpyDeviceToHost );
	cudaCheckError();

	StartKernelTiming(tic, toc, 0);
	w_calc<<<dimGrid,dimBlock>>>(objects_d,m,n,w_d);
	StopKernelTiming(tic, toc, 0, &Elapsed_Time);
	cudaMemcpy (w , w_d , n*sizeof(double) , cudaMemcpyDeviceToHost );

    do{
		StartKernelTiming(tic, toc, 0);
		pddp1<<<dimGrid,dimBlock>>>(objects_d,m,n,w_d,x_d,tmp_d);
		StopKernelTiming(tic, toc, 0, &Elapsed_Time);
		cudaMemcpy (tmp , tmp_d , n*sizeof(double) , cudaMemcpyDeviceToHost );

		StartKernelTiming(tic, toc, 0);
		pddp2<<<dimGrid,dimBlock>>>(objects_d,final_d,m,n,w_d,tmp_d);
		StopKernelTiming(tic, toc, 0, &Elapsed_Time);
		cudaMemcpy (final , final_d , m*sizeof(double) , cudaMemcpyDeviceToHost );
		
		//calculate each elements square
		StartKernelTiming(tic, toc, 0);
		power<<<dimGrid,dimBlock>>>(final_d,m,den_d);
		StopKernelTiming(tic, toc, 0, &Elapsed_Time);
		cudaMemcpy (den , den_d , m*sizeof(double) , cudaMemcpyDeviceToHost);
		cudaCheckError();

		//sum all elements
		StartKernelTiming(tic, toc, 0);
		f_sum = 0.0;
		for (int i = 0 ; i<m; i++)
			f_sum+=den[i];
		f_sum = sqrt(f_sum);
		StopKernelTiming(tic, toc, 0, &Elapsed_Time);

		//divide each element of y with norm
		StartKernelTiming(tic, toc, 0);
		division<<<dimGrid,dimBlock>>>(final_d,m,f_sum,y_d);
		StopKernelTiming(tic, toc, 0, &Elapsed_Time);
		cudaMemcpy (y, y_d,m* sizeof(double), cudaMemcpyDeviceToHost );
		cudaCheckError();

		//calculate difference xk+1 - xk and find each element's square
		StartKernelTiming(tic, toc, 0);
		diff_pow<<<dimGrid,dimBlock>>>(x_d,m,y_d);
		StopKernelTiming(tic, toc, 0, &Elapsed_Time);
		cudaMemcpy (x, x_d, m*sizeof(double), cudaMemcpyDeviceToHost );
		cudaCheckError();

		//calculate final norm 
		StartKernelTiming(tic, toc, 0); 
		fnorm = 0.0;
		for (int i = 0 ; i<m; i++)
			fnorm+=x[i];
		fnorm = sqrt(fnorm);
		StopKernelTiming(tic, toc, 0, &Elapsed_Time);

		StartKernelTiming(tic, toc, 0);
		swap<<<dimGrid,dimBlock>>>(in_d,m,y_d);
		StopKernelTiming(tic, toc, 0, &Elapsed_Time);
		cudaMemcpy (x, in_d, m*sizeof(double), cudaMemcpyDeviceToHost );
		cudaMemcpy(x_d,x,m*sizeof(double),cudaMemcpyHostToDevice);
		cudaCheckError();

	}while(fnorm > eps);
	
	printf("-----------------\n");
	printf("Elapsed_Time=%f ms\n", Elapsed_Time);

	//Print the last 5 values of y to show correctness
	printf("-----------------\n");
	for(int i=m-5; i<m; i++)
		printf("y[%d] = %.7f \n",i,y[i]);

	cudaFree(objects_d);
	cudaFree(final_d);
	cudaFree(w_d);
	cudaFree(x_d);
	cudaFree(tmp_d);
	cudaFree(in_d);
	cudaFree(den_d);
	cudaFree(y_d);
	free(w);
	free(x);
	free(tmp);
	free(den);
	free(y);
	free(objects);

	return (0);
}

