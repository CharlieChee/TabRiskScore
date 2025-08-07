# QuerySnout 🐽: Automating the Discovery of Attribute Inference Attacks against Query-Based Systems

Source code repository for the "QuerySnout: Automating the Discovery of Attribute Inference Attacks against Query-Based Systems" paper by Ana-Maria Cretu*, Florimond Houssiau*, Antoine Cully and Yves-Alexandre de Montjoye presented at the ACM CCS 2022 conference https://dl.acm.org/doi/abs/10.1145/3548606.3560581 (*denotes equal contribution).

## 1 - Environment

To reproduce the paper experiments, please install the `conda` environment from the .yml file as specified below.

```bsh
conda env create -f apa.yml
```

## 2 - Dataset preprocessing

Run the ```notebooks/dataset_discretization.ipynb``` notebook; the discretized dataset will be saved automatically under the ```datasets``` folder.

## 3 - Installing the QBS software

QuerySnout uses an optimized implementation of query-based systems, implemented in a C module interfaced with Python (the `cqbs` module). To build and install this package, go to the ```src/optimized_qbs``` directory and run the following command:

```bsh
python setup.py install
```

Note that this repository does not contain an implementation of Diffix for intellectual property reasons. Using the Diffix QBS in the library will produce a warning and return unnoised results. We provide instructions in the code on how to adapt it to add noise similarly to Diffix (or other QBSes).

## 4 - Running the attack (with several repetitions)

For the AUXILIARY and EXACT-BUT-1 scenarios, run each script by replacing DATASET with "census", "adults" and "insurance".

```bsh
./automated_attack_table_builder.sh DATASET NUM_PROCS

./automated_attack_dp_laplace.sh DATASET EPSILON NUM_PROCS

./automated_attack_qbs_simple.sh DATASET THRESHOLD SIGMA NUM_PROCS
```

Note: to run the attack against Diffix, you need to first update the C QBS code as described in ```src/optimized_qbs/qbs.c```. You may then run the script:

```bsh
./automated_attack_diffix.sh DATASET NUM_PROCS
```

# How to cite

If you re-use our code, please cite our paper:

```
@inproceedings{cretu2022querysnout,
  title={Querysnout: Automating the discovery of attribute inference attacks against query-based systems},
  author={Cretu, Ana-Maria and Houssiau, Florimond and Cully, Antoine and de Montjoye, Yves-Alexandre},
  booktitle={Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security},
  pages={623--637},
  year={2022}
}
```




