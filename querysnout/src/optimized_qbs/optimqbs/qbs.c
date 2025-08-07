#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "qbs.h"

#define SQRT2 1.41421356237
#define TRUE 1

// Setting this flag to 1 will make the code print (a lot) of debug prompts.
#define DEBUGPRINT 0

// Macro: accessing a Dataset struct.
#define GETDATA(dataset,row,attr) (dataset->data[row*(dataset->n_attrs)+attr])




// Utils for QBSes.
// From https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key .
unsigned int simple_hash(unsigned int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

/*
  Returns noise ~ N(0,1), useful for Diffix.
  Uses the Box-Muller transform: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform

  TODO: use http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html ?
*/
double normal_noise(unsigned int seed){
  srand(seed);
  double x = ((double)rand())/RAND_MAX;
  double y = ((double)rand())/RAND_MAX;
  return sqrt(-2 * log(x)) * cos(2 * M_PI * y);;
}

/*
  Returns noise ~ [-c, +c], useful for TableBuilder.
  
  We use the suggestions from the link below to ensure the noise is unbiased.
  The key idea is that taking the modulo creates bias for values of rand()
  below (high + 1 - low), yet is uniform for all other numbers.
  We thus exclude those numbers by rejection sampling.

  Source: https://stackoverflow.com/questions/5008804/generating-random-integer-from-a-range
*/
int uniform_noise(int scale, unsigned int seed){
  srand(seed);
  int low = -scale, high = scale;
  unsigned int r;
  do {
    r = rand();
  } while (r < ((unsigned int)(RAND_MAX) + 1) % (high + 1 - low));
return r % (high + 1 - low) + low;
}

/*
  Returns noise ~ Laplace(0, 1), useful for Differential Privacy.
   X = - sign(U) * ln(1 - 2|U|)   for U ~ U[-1/2, 1/2] ==> X ~ Lap(0,1).

  https://en.wikipedia.org/wiki/Laplace_distribution
*/
double laplace_noise(unsigned int seed){
  srand(seed);
  double U = (((double)rand())/RAND_MAX) - 0.5;  // ~ U[-0.5,0.5]
  if (U >= 0){
    // sign(U) = +1, abs(U) = U
    return - log(1 - 2 * U);
  } else {
    // sign(U) = -1, abs(U) = -U
    return log(1 + 2 * U);
  }
}


/*
  Create (allocate memory) for a Dataset.
*/
Dataset* makeDataset(int n_attrs, int n_data){
  Dataset *res = (Dataset*) malloc(sizeof(Dataset));
  res->n_data = n_data;
  res->n_attrs = n_attrs;
  res->data = (int*) malloc(sizeof(int)*n_attrs*n_data);
  return res;
}

/*
  Set the data for a Diffix instance.
  This copies the data in the object's memory.
*/
void setDataset(Dataset *d, int *data){
  for(int i=0; i<(d->n_data)*(d->n_attrs); i++){
    if(data[i] < 0){
      fprintf(stderr, "[WARN] Entry %d of data is negative (should be >= 0).\n", i);
    }
    d->data[i] = data[i];
  }
}

/*
  Free the memory for a Diffix instance.
*/
void freeDataset(Dataset *d){
  free(d->data);
  free(d);
}


// Internal: create (allocate memory) a QBS instance.
QBSInstance *makeQBSInstance(Dataset *d, QBS_TYPE type, void* instance){
  QBSInstance *qbs = (QBSInstance*) malloc(sizeof(QBSInstance));
  qbs->data = d;
  qbs->type = type;
  qbs->instance = instance;
  return qbs;
}

/*
  Create (allocate memory) a simple QBS instance.
*/
QBSInstance* makeSimpleQBS(Dataset *d, int bucket_threshold, double noise_scale, unsigned int seed){
  SimpleQBSInstance *instance = (SimpleQBSInstance*) malloc(sizeof(SimpleQBSInstance));
  instance->bucket_threshold = bucket_threshold;
  instance->noise_scale = noise_scale;
  instance->prev_seed = seed;
  return makeQBSInstance(d, SIMPLE, (void*) instance);
}

/*
  Create (allocate mmeory) a Diffix instance.
*/
QBSInstance* makeDiffix(Dataset *d, unsigned int seed){
  DiffixInstance *instance = (DiffixInstance*) malloc(sizeof(DiffixInstance));
  instance->seed = seed;
  return makeQBSInstance(d, DIFFIX, (void*) instance);
}

/*
  Create (allocate memory) for TableBuilder instance.
*/
QBSInstance* makeTableBuilder(Dataset *d, int noise_scale, int threshold, unsigned int seed){
  TableBuilderInstance *instance = (TableBuilderInstance*) malloc(sizeof(TableBuilderInstance));
  instance->noise_scale = noise_scale;
  instance->threshold = threshold;
  instance->seed = seed;
  return makeQBSInstance(d, TABLEBUILDER, (void*) instance);
}



// We define here a SimpleQBSInstance, useful for DP Laplace, 
// that has no noise, no thresholding, and seed = 0 (arbitrary).
SimpleQBSInstance TRUE_QBS = {0, 0.0, 0};

/*
  Create (allocate memory) for a DP Laplace instance.
*/
QBSInstance* makeDPLaplace(Dataset *d, double epsilon, unsigned int seed){
  DPLaplaceInstance *instance = (DPLaplaceInstance*) malloc(sizeof(DPLaplaceInstance));
  instance->epsilon = epsilon;
  instance->seed = seed;
  // Create a SimpleQBSInstance with the same data, for easier interface.
  // Note that no new memory is allocated: the "object" is stored in DPLaplaceInstance.
  QBSInstance true_qbs;
  true_qbs.data = d;
  true_qbs.instance = &TRUE_QBS;
  true_qbs.type = SIMPLE;
  instance->true_qbs = true_qbs;
  return makeQBSInstance(d, DPLAPLACE, (void*) instance);
}

/*
  Free the memory used by a QBS instance.
  This also frees the memory used by the instance.
*/
void freeQBSInstance(QBSInstance* d){
  free(d->instance);
  free(d);
}


/*
  Debugging function to display the bits of an unsigned integer.
  This is used to visualise the different steps of static_seed.
*/
void printfbinary(unsigned int x){
  for(int i=0; i<32; i++){
    fprintf(stderr, "%d", (x&(1 << 31))>>31);
    x <<= 1;
    // Display spaces every byte (for readability).
    if(!((i+1) % 8)){fprintf(stderr, " ");}
  }
}

/*
  Returns the static seed for a condition x[attribute] OP value, with OP = == if condition == 1
   and != if condition == -1.
  We hash a unsigned int A, such that A[32:29] is the condition, A[28:21] is the attribute number,
   and the rest (A[20:1]) is the value. This is a fast heuristic which produces unique hash
   values for each condition triplet (attribute, value, condition) under normal operation of
   the QBS, i.e., if attribute < 2^8 (less than 64 columns), condition < 2^4 (which is required
   by the syntax anyway), and value < 2^20.
*/
unsigned int static_seed(DiffixInstance* diffix, int attribute, int value, int condition){
  // First, encode the value of the condition as a unsigned int (bitfield of size 32).
  // Since values are in [0, num_values], this should be relatively small (<< 2^20), so
  //  the least significant bits will encode the value, while the MSBs should be zero.
  unsigned int bitfield = (unsigned int) value;
#if DEBUGPRINT
  fprintf(stderr, "bitfield = ");
  printfbinary(bitfield);
  fprintf(stderr, " (value=%d)\n", value);
#endif
  // We then encode the attribute = the number of the column being queried.
  // This shouldn't usually be more than 16-32, so a few bits suffice.
  unsigned int attribute_bitfield = (unsigned int) attribute;
  // Encode this attribute in the bitfield, in positions [20, 28]
  bitfield = bitfield ^ (attribute_bitfield << 20);
#if DEBUGPRINT
  fprintf(stderr, "bitfield = ");
  printfbinary(bitfield);
  fprintf(stderr, " (attr=%d)\n", attribute);
#endif
  // The condition can take values {-3, -2, -1, 1, 2, 3} (0 is ignored, since it means
  //  no condition is put on the attribute ==> no noise). 6 values can be encoded in
  //  three bits, which will be the LSBs of an unsigned int containing condition+3.
  unsigned int condition_bitfield = (unsigned int) (condition + 3);
  // We shift this field to keep only the 4 LSBs (we have some margin, just in case).
  bitfield = bitfield ^ (condition_bitfield << 28);
#if DEBUGPRINT
  fprintf(stderr, "bitfield = ");
  printfbinary(bitfield);
  fprintf(stderr, " (condition=%d)\n", condition);
#endif
  // Finally, hash the bitfield (which encodes this condition), and XOR it with
  //  Diffix's seed, as per the specifications.
  return simple_hash(bitfield) ^ (diffix->seed);
}


/*
  Perform queries on a QBS instance.

  The queries are given as a pair of arrays (values, flags) of size n_queries * d->n_attrs,
   with values[i*n_attrs:(i+1)*n_attrs[ corresponding to the fields of one query, in order
   and conditions[idem[ with values -1 (=/=), 0 (no condition) or +1 (==).

  For instance, for a dataset with attributes (A1, A2, A3), the query (A1=42 and A3!=7)
   is encoded as values=[42, x, 7] and conditions=[1,0,-1].

  The results are stored in the output array.
*/
void performQueries(QBSInstance* qbs, int n_queries, int *values, int *conditions, int *output){
  int *q_values, *q_conditions;  // Variables to represent the current query.
  Dataset *data = qbs->data;
  int n_attrs = data->n_attrs;
  // Process each query in order.
  for(int q=0; q<n_queries; q++){
    q_values = &(values[q*n_attrs]);
    q_conditions = &(conditions[q*n_attrs]);
    // First, perform a (clean) query on this dataset.
    unsigned int user_set_size = 0;
    unsigned int user_set_hash = 0;
    // Iterate over all records to count users matching queries.
    for(int rec=0; rec<data->n_data; rec++){
      // This flag is True as long as all the previous conditions are correctly
      // matched, and the query is thus true for this record.
      // If matched is true after all conditions are checked, the record is
      // part of the query set for this query.
      unsigned int matched = TRUE;
      for(int attr=0; (attr<n_attrs) && matched; attr++){
        // Check if this attribute matches: its condition in +-1, and the values don't match.
        int condition_value = q_values[attr],
            qbs_data = GETDATA(data, rec, attr);
        switch(q_conditions[attr]){
          case 0:  // No condition on the attribute.
            matched = TRUE;
            break;
          case 1:  // Equality condition.
            matched = (qbs_data == condition_value);
            break;
          case 2:  // Greater than.
            matched = (qbs_data > condition_value);
            break;
          case 3:  // Greater than or equal.
            matched = (qbs_data >= condition_value);
            break;
          case -1:  // Different from.
            matched = (qbs_data != condition_value);
            break;
          case -2:  // Smaller than.
            matched = (qbs_data < condition_value);
            break;
          case -3:  // Smaller than or equal.
            matched = (qbs_data <= condition_value);
            break;
        }
      }
      // If the record matches, add it to the user set.
      if(matched){
        user_set_size += 1;
        // Also compute the live hash, as a XOR of hash(indices).
        user_set_hash = user_set_hash ^ simple_hash(rec + 1);
      }
    }
    // Second, add noise, and use bucket suppression.
    double result = user_set_size;  // Real answer.
    // First, fetch the corresponding query-based system.
    if(qbs->type == SIMPLE){
      // Fetch the simple instance from the QBS.
      SimpleQBSInstance *simpleqbs = (SimpleQBSInstance*) qbs->instance;
      // Check the threshold: if under, set to zero.
      if (result <= simpleqbs->bucket_threshold){
        result = 0;
      } else {
        // Add scale * N(0, 1).
        result += simpleqbs->noise_scale * normal_noise(simpleqbs->prev_seed);
        // Trickity trick in the implementation: increment prev_seed to generate independent noises.
        simpleqbs->prev_seed += 1;
      }
    }
    else if (qbs->type == DIFFIX) {
      /*
        To protect the intellectual property of Aircloak, we have removed this
        part of the code, which computes the query answer for the Diffix QBS.

        If you want to implement this method yourself, use the following:
         - qbs->instance is a "DiffixInstance" (after type-casting). This
           contains the seed of the RNG.
         - `user_set_size` is the size of the user set of the query.
         - `user_set_hash` is a hash of the user set of the query.
         - the variable `result` is the output of the query, and should be
           modified using the logic of the QBS.
      */
      printf("Diffix QBS not implemented.");
    } else if (qbs->type == TABLEBUILDER){
        // Fetch the TableBuilder instance from the QBS.
        TableBuilderInstance *tablebuilder = (TableBuilderInstance*) qbs->instance;
        // Bucket suppression: thresholding here.
        if(result <= tablebuilder->threshold){
          result = 0;
        } else {
          // Add uniform noise in [-c, +c].
          // The noise is seeded with the user set (XORs of individual seeds), as
          //  suggested in design docs.
          result += uniform_noise(tablebuilder->noise_scale, tablebuilder->seed ^ user_set_hash);
        }
    } else {
      // This is highly abnormal -- abort everything.
      printf("Error: bad qbs->type.\n");
      return;
    }
    // Round, trim, and save the output to an array.
    output[q] = round(result);
    if(output[q] < 0){ output[q] = 0; };
  }
}

/*
  Extension of performQueries with budget, for Differential Privacy.
  This uses performQueries on qbs->true_qbs to obtain the exact results,
   then .
  The syntax is the same as performQueries, except a budget_fractions argument is added.
*/
void performQueriesWithBudget(DPLaplaceInstance* qbs, int n_queries, int *values, int *conditions,
  double *budget_fractions, int *output){
  // First, perform the queries truthfully, using the inner "true QBS".
  performQueries(&(qbs->true_qbs), n_queries, values, conditions, output);
  // output is now populated with the correct answers, as int.
  double result;
  for(int i=0; i < n_queries; i++){
    // result, a double, incorporates the noise added to ints.
    result = output[i] + laplace_noise(qbs->seed) / (budget_fractions[i] * qbs->epsilon);
    // Increment the seed to generate new noises.
    qbs->seed += 1;
    // Round and trim the output.
    output[i] = round(result);
    if(output[i] < 0){ output[i] = 0; };
  }
}
