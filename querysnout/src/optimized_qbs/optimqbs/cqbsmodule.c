#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "qbs.h"



// Create a new QBS object with a data array.
static PyObject *cqbs_new_instance(PyObject *self, PyObject *args){
	// The list of tuples (data) to be extracted.
	PyObject *DataList;
	// Dictionary containing the Query-Based System's arguments.
	PyObject *QBSArguments;
	// Perform and parse the query.
	if (!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &DataList, &PyDict_Type, &QBSArguments)) {
		PyErr_SetString(PyExc_TypeError, "Data must be a list of tuples. Second argument must be the seed.");
		return NULL;
	}
	// It's a list! Get it's length = data size.
	int n_data = PyList_Size(DataList);
	// Get the number of records from the first element.
	PyObject *element = PyList_GetItem(DataList, 0);
	if(!PyTuple_Check(element)){
		PyErr_SetString(PyExc_TypeError, "Data elements must be tuples.");
		return NULL;
	}
	// Get the number of features from the tuple.
	int n_features = PyTuple_Size(element);
	// Create the QBS instance with (n_data, n_features);
	Dataset *dataset = makeDataset(n_features, n_data);
	// Populate the QBS instance with the data.
	PyObject *data_entry;
	for(int i=0; i<n_data; i++){
		element = PyList_GetItem(DataList, i);
		if(!PyTuple_Check(element)){
			PyErr_SetString(PyExc_TypeError, "Data elements must be tuples.");
			freeDataset(dataset);
			return NULL;
		}
		if(PyTuple_Size(element) != n_features){
			PyErr_SetString(PyExc_IndexError, "Data elements must all be of same size.");
			freeDataset(dataset);
			return NULL;
		}
		for(int j=0; j<n_features; j++){
			data_entry = PyTuple_GetItem(element, j);
			dataset->data[i*n_features + j] = (int) PyLong_AsLong(data_entry);
		}
	}
	// Read the parameters from the dictionary.
	// Parse the type of the Query-Based System.
	PyObject *py_QBS_type = PyDict_GetItemString(QBSArguments, "type");
	QBS_TYPE qbs_type;
	if(!py_QBS_type){
		PyErr_SetString(PyExc_ValueError, "Data elements must all be of same size.");
		freeDataset(dataset);
		return NULL;
	} else {
		qbs_type = (QBS_TYPE) PyLong_AsLong(py_QBS_type);
#ifdef DEBUG
		printf("Using QBS type: %d.\n", qbs_type);
#endif
	}
	// Depending on the type, parse additional parameters.
	QBSInstance *instance;
	if (qbs_type == SIMPLE){
		/* Threshold for bucket suppression. */
		int bucket_threshold = 0;
		PyObject *py_bucket_threshold = PyDict_GetItemString(QBSArguments, "bucket_threshold");
		if(py_bucket_threshold){
			bucket_threshold = (unsigned int) PyLong_AsLong(py_bucket_threshold);
#ifdef DEBUG
			printf("Setting: bucket_threshold=%d\n", bucket_threshold);
#endif
		}
		/* Scale for the Gaussian noise on counts. */
		double noise_scale = 0.;
		PyObject *py_noise_scale = PyDict_GetItemString(QBSArguments, "noise_scale");
		if(py_noise_scale){
			noise_scale = PyFloat_AsDouble(py_noise_scale);
#ifdef DEBUG
			printf("Setting: noise_scale=%.2f\n", noise_scale);
#endif
		}
		/* Seed to use for noises (incremented over time). */
		unsigned int seed = 0;
		PyObject *py_seed = PyDict_GetItemString(QBSArguments, "seed");
		if(py_seed){
			seed = (unsigned int) PyLong_AsLong(py_seed);
#ifdef DEBUG
			printf("Setting: seed=%d\n", seed);
#endif
		}
		// Create the Simple QBS instance.
		instance = makeSimpleQBS(dataset, bucket_threshold, noise_scale, seed);
	} else if (qbs_type == DIFFIX) {
		/* Diffix central seed (to compute other seeds). */
		unsigned int seed = 0;
		PyObject *py_seed = PyDict_GetItemString(QBSArguments, "seed");
		if(py_seed){
			seed = (unsigned int) PyLong_AsLong(py_seed);
#ifdef DEBUG
			printf("[Diffix] Setting: seed=%d\n", seed);
#endif
		}
		// Create the Diffix instance.
		instance = makeDiffix(dataset, seed);
	} else if (qbs_type == TABLEBUILDER) {
		/* TableBuilder attributes. */
		unsigned int seed = 0;
		PyObject *py_seed = PyDict_GetItemString(QBSArguments, "seed");
		if(py_seed){
			seed = (unsigned int) PyLong_AsLong(py_seed);
#ifdef DEBUG
			printf("[Tablebuilder] Setting: seed=%d\n", seed);
#endif
		}
		// Get the noise scale.
		int noise_scale = 2;
		PyObject *py_noise_scale = PyDict_GetItemString(QBSArguments, "noise_scale");
		if(py_noise_scale){
			noise_scale = (int) PyLong_AsLong(py_noise_scale);
#ifdef DEBUG
			printf("[Tablebuilder] Setting: noise_scale=%d\n", noise_scale);
#endif
		}
		// Get the threshold.
		int threshold = 4;
		PyObject *py_threshold = PyDict_GetItemString(QBSArguments, "threshold");
		if(py_threshold){
			threshold = (int) PyLong_AsLong(py_threshold);
#ifdef DEBUG
			printf("[Tablebuilder] Setting: threshold=%d\n", threshold);
#endif
		}
		// Create the TableBuiler instance.
		instance = makeTableBuilder(dataset, noise_scale, threshold, seed);
	} else if (qbs_type == DPLAPLACE) {
		/* DPLaplace attributes. */
		unsigned int seed = 0;
		PyObject *py_seed = PyDict_GetItemString(QBSArguments, "seed");
		if(py_seed){
			seed = (unsigned int) PyLong_AsLong(py_seed);
#ifdef DEBUG
			printf("[DPLaplace] Setting: seed=%d\n", seed);
#endif
		}
		double epsilon = 1;
		PyObject *py_epsilon = PyDict_GetItemString(QBSArguments, "epsilon");
		if(py_epsilon){
			epsilon = (double) PyFloat_AsDouble(py_epsilon);
#ifdef DEBUG
			printf("[DPLaplace] Setting: epsilon=%.3f\n", epsilon);
#endif
		}
		instance = makeDPLaplace(dataset, epsilon, seed);
	} else {
		// Unknown QBS type: abort.
		freeDataset(dataset);
		return NULL;
	}
	// Return the pointer to the QBS instance!
	return PyLong_FromLong((long) instance);
}



static PyObject *cqbs_query(PyObject *self, PyObject *args){
	// Parse the input to retrieve the QBS instance, and two lists.
	long QBSInstanceAddress;
	PyObject *PyValues;
	PyObject *PyConditions;
	PyObject *PyBudgetFractions = NULL;
	if (!PyArg_ParseTuple(args, "lO!O!|O!", &QBSInstanceAddress,
			&PyList_Type, &PyValues,
			&PyList_Type, &PyConditions,
			&PyList_Type, &PyBudgetFractions)){
		PyErr_SetString(PyExc_TypeError, "Invalid inputs: must be QBS-instance, list of values, list of queries.");
		return NULL;
	}
	// Parse the input in tables that can be processed by the QBS.
	int n_queries = PyList_Size(PyValues);
	if(n_queries != PyList_Size(PyConditions)){
		PyErr_SetString(PyExc_IndexError, "Values and conditions have incompatible lengths.");
		return NULL;
	}
	if(PyBudgetFractions){
		if(n_queries != PyList_Size(PyBudgetFractions)){
			PyErr_SetString(PyExc_IndexError, "Conditions and budgets have incompatible lengths.");
			return NULL;
		}
	}
	QBSInstance *instance = (QBSInstance*) QBSInstanceAddress;  // Open the QBS instance.
	Dataset *dataset = instance->data;
	int n_attrs = dataset->n_attrs;
	// Parse the values and conditions from the Python inputs.
	int *values = (int *) malloc(sizeof(int)*n_queries*n_attrs);
	int *conditions = (int *) malloc(sizeof(int)*n_queries*n_attrs);
	double *budget_fractions = NULL;
	PyObject *element_values, *element_conditions, *data_entry;
	for(int i=0; i<n_queries; i++){
		element_values = PyList_GetItem(PyValues, i);
		element_conditions = PyList_GetItem(PyConditions, i);
		for(int j=0; j<n_attrs; j++){
			data_entry =  PyTuple_GetItem(element_values, j);
			values[i*n_attrs+j] = (int) PyLong_AsLong(data_entry);
			data_entry = PyTuple_GetItem(element_conditions, j);
			conditions[i*n_attrs+j] = (int) PyLong_AsLong(data_entry);
		}
	}
	// Do the same for the budget fractions, if provided.
	if(PyBudgetFractions){
		PyObject *fraction_entry;
		budget_fractions = (double *) malloc(sizeof(double)*n_queries);
		for(int i=0; i<n_queries; i++){
			fraction_entry = PyList_GetItem(PyBudgetFractions, i);
			budget_fractions[i] = (double) PyFloat_AsDouble(fraction_entry);
			if((budget_fractions[i] <= 0) || (budget_fractions[i] > 1)){
				PyErr_SetString(PyExc_ValueError, "Budget fractions must be in (0, 1].");
				free(values);
				free(conditions);
				free(budget_fractions);
				return NULL;
			}
		}
	}
	// Perform the queries on the QBS.
	int *result = (int *) malloc(sizeof(int)*n_queries);
	if(instance->type == DPLAPLACE){
		if(!budget_fractions){
			PyErr_SetString(PyExc_ValueError, "Must have BudgetFractions as an argument for DP Laplace.");
			free(result);
			free(values);
			free(conditions);
			return NULL;
		}
		performQueriesWithBudget((DPLaplaceInstance*) instance->instance,
			n_queries, values, conditions, budget_fractions, result);
	} else {
		if(budget_fractions){
			fprintf(stderr, "[WARNING] BudgetFractions provided but not used by this QBS type.\n");
		}
		performQueries(instance, n_queries, values, conditions, result);
	}
	// Transform the result to a Python list.
	PyObject *output = PyList_New(n_queries);
	for(int i=0; i<n_queries; i++){
		PyList_SetItem(output, i, PyLong_FromLong((long)result[i]));
	}
	// Free the memory allocared for input arrays.
	free(values);
	free(conditions);
	free(result);
	if(budget_fractions){
		free(budget_fractions);
	}
	// Return the result (in PyList format).
	return output;
}


static PyObject *cqbs_structured_query(PyObject *self, PyObject *args){
	// Parse the inputs.
	long QBSInstanceAddress;
	PyObject *PyUsersList;
	PyObject *PyConditions;
	PyObject *PyBudgetFractions = NULL;
	if (!PyArg_ParseTuple(args, "lO!O!|O!", &QBSInstanceAddress,
			&PyList_Type, &PyUsersList,
			&PyList_Type, &PyConditions,
			&PyList_Type, &PyBudgetFractions)){
		PyErr_SetString(PyExc_TypeError,
			"Invalid inputs: must be QBS-instance, list of users, list of queries, [list of budget fractions].");
		return NULL;
	}
	// From the lists, get the number of conditions and the number of users (product = #queries).
	QBSInstance *instance = (QBSInstance*) QBSInstanceAddress;  // Open the QBS instance.
	Dataset *dataset = instance->data;
	int n_users = PyList_Size(PyUsersList);
	int n_conditions = PyList_Size(PyConditions);
	int n_queries = n_users * n_conditions;
	int n_attrs = dataset->n_attrs;
	// Fill in the arrays with the users data and the condition rules.
	// These will then be processed to have the values and conditions arrays.
	int *users_data = (int*) malloc(sizeof(int)*n_users*n_attrs);
	int *condition_rules = (int*) malloc(sizeof(int)*n_queries*n_attrs);
	for(int u=0; u<n_users; u++){
		PyObject *element = PyList_GetItem(PyUsersList, u);
		int userid = (int) PyLong_AsLong(element);
		// Copy the memory from the data.
		memcpy(&(users_data[n_attrs*u]), &(dataset->data[n_attrs*userid]), sizeof(int)*n_attrs);
		// Note that users_data currently contains the sensitive attribute. However, that attribute is
		//  not known to the attacker. Hence, we interpret the semantics of the condition rules differently.
		// We assume that +1 means "sensitive == 1", -1 means "sensitive == 0", 0 means "no condition".
		// This can simply be enforced by setting the value of the sensitive attribute to 1 in the user data.
		// This attribute is assumed to be _the last_.
		users_data[n_attrs*u+n_attrs-1] = 1;
	}
	for(int c=0; c<n_conditions; c++){
		PyObject *element = PyList_GetItem(PyConditions, c);
		if(!PyTuple_Check(element)){
			PyErr_SetString(PyExc_TypeError, "Condition elements must be tuples.");
			free(users_data);
			free(condition_rules);
			return NULL;
		}
		for(int i=0; i<n_attrs; i++){
			PyObject *entry = PyTuple_GetItem(element, i);
			condition_rules[c*n_attrs+i] = (int) PyLong_AsLong(entry);
		}
	}
	// Finally, *if* the budget_fractions is not NULL, we create an array that stores the budget
	//  of individual queries, after duplication (in the same order as below: each user's queries
	//  are consecutive, hence the array is PyBudgetFractions repeated n_users times).
	double *budget_fractions = NULL;
	if(PyBudgetFractions){
		if(PyList_Size(PyBudgetFractions) != n_conditions){
			PyErr_SetString(PyExc_ValueError, "length(budgets) must be equal to length(conditions).");
			free(users_data);
			free(condition_rules);
			return NULL;
		}
		// This array maintains, for each *query*, the budget allocated to it.
		// For this, we duplicate the budget fractions for each user (similarly to conditions).
		budget_fractions = (double*) malloc(sizeof(double)*n_queries);  // n_users * n_conditions
		for(int c=0; c<n_conditions; c++){
			PyObject *element = PyList_GetItem(PyBudgetFractions, c);
			double budget_for_this_query = PyFloat_AsDouble(element);
			if((budget_for_this_query <= 0) || (budget_for_this_query > 1)){
				PyErr_SetString(PyExc_ValueError, "Budget fractions must be in (0, 1].");
				free(users_data);
				free(condition_rules);
				free(budget_fractions);
				return NULL;
			}
			for(int u=0; u<n_users; u++){
				budget_fractions[u*n_conditions + c] = budget_for_this_query;
			}
		}
	}
#ifdef DEBUG
	fprintf(stderr, "RULES\n");
	for(int i=0;i<n_conditions;i++){
		for(int j=0;j<n_attrs;j++){
			fprintf(stderr, "%d ", condition_rules[i*n_attrs+j]);
		}
		fprintf(stderr, "\n");
	}
#endif
	// Now, we duplicate the memory from these users.
	// The values of the users are duplicated n_conditions times, for each user (all data from
	//  the same user is consecutive).
	int *values = (int*) malloc(sizeof(int)*n_queries*n_attrs);
	for(int i=0; i<n_users; i++){  // Each user's row.
		for(int j=0; j<n_conditions; j++){  // Repetition of the same row.
			memcpy(&(values[(i*n_conditions+j)*n_attrs]), &(users_data[n_attrs*i]), sizeof(int)*n_attrs);
		}
	}
#ifdef DEBUG
	fprintf(stderr, "VALUES\n");
	for(int i=0;i<n_queries;i++){
		for(int j=0;j<n_attrs;j++){
			fprintf(stderr, "%d ", values[i*n_attrs+j]);
		}
		fprintf(stderr, "\n");
	}
#endif
	// Conditions are duplicated, all as a block, for each user.
	int *conditions = (int*) malloc(sizeof(int)*n_queries*n_attrs);
	for(int i=0; i<n_users; i++){
		memcpy(&(conditions[i*n_conditions*n_attrs]), condition_rules, sizeof(int)*n_attrs*n_conditions);
	}
#ifdef DEBUG
	fprintf(stderr, "CONDITIONS\n");
	for(int i=0;i<n_queries;i++){
		for(int j=0;j<n_attrs;j++){
			fprintf(stderr, "%d ", conditions[i*n_attrs+j]);
		}
		fprintf(stderr, "\n");
	}
	if(budget_fractions){
		fprintf(stderr, "BUDGET FRACTIONS\n");
		for(int i=0;i<n_users;i++){
			for(int j=0;j<n_conditions;j++){
				fprintf(stderr, "%.2f ", budget_fractions[i*n_conditions+j]);
			}
			fprintf(stderr, "\n");
		}
	}
#endif
	// Perform the queries.
	int *result = (int*) malloc(sizeof(int)*n_queries);
	// If the QBS uses differential privacy, we additionally use the budget argument.
	if(instance->type == DPLAPLACE){
		// If budget fractions are not given, throw an error.
		if(!budget_fractions){
			PyErr_SetString(PyExc_ValueError, "Must have BudgetFractions as an argument for DP Laplace.");
			free(result);
			free(users_data);
			free(condition_rules);
			free(values);
			free(conditions);
			return NULL;
		}
		performQueriesWithBudget((DPLaplaceInstance*) instance->instance,
			n_queries, values, conditions, budget_fractions, result);
	} else {
		if(budget_fractions){
			fprintf(stderr, "[WARNING] BudgetFractions provided but not used by this QBS type.\n");
		}
		performQueries(instance, n_queries, values, conditions, result);
	}
	// Transform the result to a Python list.
	PyObject *output = PyList_New(n_queries);
	for(int i=0; i<n_queries; i++){
		PyList_SetItem(output, i, PyLong_FromLong((long)result[i]));
	}
	// Free the memory.
	free(result);
	free(users_data);
	free(condition_rules);
	free(values);
	free(conditions);
	if(budget_fractions){ free(budget_fractions); }
	// Return the result (as a list).
	return output;
}



static PyObject *cqbs_free(PyObject *self, PyObject *args){
	long QBSInstanceAddress;
	if (!PyArg_ParseTuple(args, "l", &QBSInstanceAddress)) {
		PyErr_SetString(PyExc_TypeError, "Wrong format (must be long).");
		return NULL;
	}
	QBSInstance *instance = (QBSInstance*) QBSInstanceAddress;
	freeQBSInstance(instance);
	return PyLong_FromLong(0);
}





/* All the Cython stuff */

static PyMethodDef QBSMethods[] = {
    {"create_qbs",  cqbs_new_instance, METH_VARARGS, "Create a new QBS instance."},
    {"free_qbs", cqbs_free, METH_VARARGS, "Free the QBS instance."},
    {"query_qbs", cqbs_query, METH_VARARGS, "Query the QBS instance."},
    {"structured_query_qbs", cqbs_structured_query, METH_VARARGS, "Structured queries on a QBS instance"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


char doc[] = "Super-fast QBS implementations";

static struct PyModuleDef cqbsmodule = {
    PyModuleDef_HEAD_INIT,
    "cqbs",   /* name of module */
    doc, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    QBSMethods
};


PyMODINIT_FUNC
PyInit_cqbs(void)
{
    return PyModule_Create(&cqbsmodule);
}