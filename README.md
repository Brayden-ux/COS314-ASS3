# COS314-ASS3

PREREQUISITES
- Java 11 or later 

FILES
- GPArithmetic.jar
- GPDecisionTree.jar
- MultiRunEvaluator.jar
- src for source code

-------------------------------
PROGRAM EXECUTION
-------------------------------
Run:  

    java -jar GPArithmetic.jar or GPDecisionTree.jar 
    > Enter random seed (long): 42
    > Enter training CSV filepath: Breast_train.csv
    > Enter test CSV filepath: Breast_test.csv


 
OUTPUT per generation:
```
Gen N | BestFit=0.XXXX | Acc=... Prec=... Rec=... F1=... [TP=... FP=... FN=... TN=...]
    Expression: (symbolic expression)
```

-------------------------------
MULTI-RUN EVALUATOR (30 independant runs)
-------------------------------

Run:

    java -jar MultiRunEvaluator.jar 
    > Enter random seed (long): 42
    > Enter training CSV filepath: Breast_train.csv
    > Enter test CSV filepath: Breast_test.csv

Note: Generates 30 different seeds based on given seed for reproducibility 

OUTPUT:
- Per-run test accuracy and F1 for both algorithms
- Summary statistics (mean ± std, best run)
- Best run details (expression / tree)
- Table comparison
- Wilcoxon signed-rank test result
- Saves to:  multirun_results.txt
