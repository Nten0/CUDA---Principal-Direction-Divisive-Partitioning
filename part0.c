#include <stdio.h>
#include <stdlib.h>
#include "file_io.h"
#include <math.h>
#include <time.h>

double *objects; /* [i*numObjs + j] data objects */

int main(int argc, char **argv) {

	int numCoords, numObjs;
	
	char *input_file;

	input_file = argv[1];

	/* read data points from file ------------------------------------------*/
	objects = file_read(input_file, &numCoords, &numObjs);

	printf("::Objects loaded::\n");
	printf("Objects: %d\n", numObjs);
	printf("Attributes: %d\n", numCoords);

	int n = numCoords;
	int m = numObjs;
	int i, j, k;
	int *e = malloc(m * sizeof(int));
	int rounds =0;
	double fnorm;
	double eps = pow(10,-6);
	double *w = malloc(n * sizeof(double));
	double *x = malloc(m*sizeof(double));
	double *y = malloc(m*sizeof(double));
	double *tmp = malloc(n*sizeof(double));
	double *row = malloc(m*sizeof(double));
	clock_t start, end;

	start = clock();


	for (i = 0; i < n; i++)
	{
		w[i]=0;
		for (j = 0; j < m; j++)
			w[i] += objects[i*m +j];
		w[i] /= m;
	}
	do
	{
		if(rounds == 0)
		{
			for(i=0;i<m;i++)
				x[i]=1.0;
		}
		else
		{
			for(i=0;i<m;i++)
					x[i]=y[i];
		}
		rounds++;
		
		double final[m];
		

		for(i=0;i<n;i++)
		{
			tmp[i]=0.0;
			for(j=0;j<m;j++)
				tmp[i]+= ( objects[i*m+j] - w[i] ) * x[j];
		}
		
	
		//upologismos M'-e*w'
		for(i=0;i<m;i++)
		{
			final[i]=0.0;
			for(j=0;j<n;j++)
				final[i] += ( objects[j*m+i] - w[j] ) * tmp[j];
		}

		//upologismos normas
		double sum=0.0;
		for(i=0;i<m;i++)
		{
			sum+=pow(final[i],2);
		}
		double norm=sqrt(sum);
		//upologismos x(k+1)???
		for(i=0;i<m;i++)
			y[i]=final[i]/norm;

		sum = 0.0;
		for(i=0;i<m;i++)
		{
			x[i] = y[i] - x[i];
			sum+=pow(x[i],2);
		}
		fnorm=sqrt(sum);

	}while(fnorm > eps);

	end = clock();

	for(i=m-5;i<m;i++)
		printf("y[%d] = %.7f \n",i,y[i]);
	printf("Rounds = %d\n",rounds);

	printf("Time = %f msec\n",(double) (end - start)/CLOCKS_PER_SEC*1000);
	free(objects);
	return (0);
}

