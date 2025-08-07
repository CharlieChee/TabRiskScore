#ifndef QBS_H
#define QBS_H 1

// A class that represents and stores a dataset.
typedef struct {
	int *data;
	int n_attrs;
	int n_data;
} Dataset;

// Query-based systems are represented by a flag.
typedef enum qbs_types {
	SIMPLE=0,
	DIFFIX=1,
	TABLEBUILDER=2,
	DPLAPLACE=3,
} QBS_TYPE;

// QBS: simple.
typedef struct {
	int bucket_threshold;
	double noise_scale;
	// Trick to have different noises: this is a seed to
	//  increase over time to generate independent samples over time.
	unsigned int prev_seed;
} SimpleQBSInstance;

// QBS: Diffix.
typedef struct {
	unsigned int seed;
} DiffixInstance;

// QBS: TableBuilder.
typedef struct {
	int noise_scale;
	int threshold;
	unsigned int seed;
} TableBuilderInstance;

// Generic "class" for the QBSes.
typedef struct{
	Dataset *data;
	void *instance;
	QBS_TYPE type;
} QBSInstance;

// QBS: Differentially Private Laplace.
// Defined after QBSInstance, as it is required in it.
typedef struct {
	unsigned int seed;
	double epsilon;
	// To simplify the interface, the DP QBS maintains a SimpleQBSInstance
	//  (mapping to the same data) with no noise or threshold, to get the
	//  correct results to queries.
	// Importantly, this is *not* a pointer, mostly for ease of code writing.
	// (also, the object is quite light, so there is little overhead here).
	QBSInstance true_qbs;
} DPLaplaceInstance;



// Creator instances: datasets.

Dataset* makeDataset(int n_attrs, int n_data);

void setDataset(Dataset *d, int *data);

void freeDataset(Dataset* d);


// Creator instances: simple QBS and Diffix.

QBSInstance* makeSimpleQBS(Dataset *d, int bucket_threshold, double noise_scale, unsigned int seed);

QBSInstance* makeDiffix(Dataset *d, unsigned int seed);

QBSInstance* makeTableBuilder(Dataset *d, int noise_scale, int threshold, unsigned int seed);

QBSInstance* makeDPLaplace(Dataset *d, double epsilon, unsigned int seed);

void freeQBSInstance(QBSInstance* d);


// Perform queries on a QBS instance.

void performQueries(QBSInstance* diffix, int n_queries, int *values, int *conditions, int *output);

void performQueriesWithBudget(DPLaplaceInstance* qbs, int n_queries, int *values, int *conditions,
	double *budget_fractions, int *output);

#endif