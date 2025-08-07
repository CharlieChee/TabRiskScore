#include <stdio.h>
#include <stdlib.h>

#include "qbs.h"


// Test function (temporary).
int main(int argc, char **argv){
	int n_attr = 3;
	int n_data = 10;
	unsigned int seed = 420;
	// Create the dataset.
	Dataset *data = makeDataset(n_attr, n_data);
	for(int i=0; i<n_data*n_attr; i++){
		data->data[i] = 0;
	}
	fprintf(stderr, "Dataset (%d, %d)\n", data->n_data, data->n_attrs);
	//QBSInstance *d = makeSimpleQBS(data, 1, 5.0, seed);
	// QBSInstance *d = makeDiffix(data, seed);
	QBSInstance *d = makeTableBuilder(data, 2, 4, seed);
	// The queries to perform.
	int values[6] = {1,2,3,1,2,3};
	int conditions[6] = {-1,0,-2,-1,0,-2};
	int output[2];
	performQueries(d, 2, values, conditions, output);
	fprintf(stderr, "==> Output %d.\n", output[0]);
	fprintf(stderr, "==> Output %d.\n", output[1]);
	freeDataset(data);
	freeQBSInstance(d);
	return 0;
}