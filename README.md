# Paper
To repoduce the computations of the submitted paper.
1) Install physmm: (works on Ubuntu, does not work on windows)
- git clone pyhsmm
- install pybasicbayes
- go to folder pyhsmm
- python setup.py build_ext --inplace
- python setup.py install --user
- test: example folder -> python hsmm.p
2) Computations:
- Clone this report and put the tool-tracking-data folder in the models folder. 
- Change the working directory to the models folder.
- Then run the script: paper_computations.py
- The results of the test set will be saved as csv in the models folder.

3) Benchmarks:




# Hidden Markov Models

Useful Links:
- Problem Statement: https://cmutschler.de/datafest-tool-tracking-with-machine-learning

- Data Description: https://cmutschler.de/datasets/tool-tracking-dataset

- Data Repository: https://github.com/mutschcr/tool-tracking

- HDP-HSMM: https://github.com/mattjj/pyhsmm

## References

## Authors

Amadeu Scheppach, Matthias Gruber, Siddharth Bhargava, Stefan Depperschmidt
