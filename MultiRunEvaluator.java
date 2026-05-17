import java.io.*;
import java.util.*;

public class MultiRunEvaluator {

    private static final int NUM_RUNS = 30;

    public static void main(String[] args) throws Exception {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter base seed (long): ");
        long baseSeed = scanner.nextLong();

        String trainPath = "Breast_train.csv";
        String testPath = "Breast_test.csv";

        System.out.println("\nLoading data...");
        DataLoader train = new DataLoader(trainPath);
        DataLoader test  = new DataLoader(testPath);
        System.out.printf("Train: %d | Test: %d%n", train.n, test.n);

        double[] arithTestAcc = new double[NUM_RUNS];
        double[] arithF1 = new double[NUM_RUNS];
        double[] dtTestAcc = new double[NUM_RUNS];
        double[] dtF1 = new double[NUM_RUNS];
        long[]   seeds = new long[NUM_RUNS];

        // Seeds are base + run index for reproducibility
        for (int i = 0; i < NUM_RUNS; i++) {
            seeds[i] = baseSeed + i * 1000L;
        }

        // GP Arithmetic
        System.out.println(" GP ARITHMETIC : 30 Runs");

        int bestArithRun = 0; double bestArithAcc = -1;
        GPArithmetic.Individual bestArithInd = null;

        for (int i = 0; i < NUM_RUNS; i++) {
            long seed = seeds[i];
            GPArithmetic gp = new GPArithmetic(seed, train.X, train.y);
            GPArithmetic.Individual best = gp.evolve(false);  // silent

            Metrics testM = new Metrics(best.predict(test.X), test.y);
            arithTestAcc[i] = testM.accuracy;
            arithF1[i] = testM.fMeasure;

            System.out.printf("Run %2d | seed=%-12d | TestAcc=%.4f | F1=%.4f%n",
                    i + 1, seed, testM.accuracy, testM.fMeasure);

            if (testM.accuracy > bestArithAcc) {
                bestArithAcc = testM.accuracy;
                bestArithRun = i;
                bestArithInd = best;
            }
        }

        // GP Decision Tree
        System.out.println("GP DECISION TREE : 30 Runs ");

        int bestDTRun = 0; double bestDTAcc = -1;
        GPDecisionTree.Individual bestDTInd = null;

        for (int i = 0; i < NUM_RUNS; i++) {
            long seed = seeds[i];
            GPDecisionTree gp = new GPDecisionTree(seed, train.X, train.y);
            GPDecisionTree.Individual best = gp.evolve(false);  // silent

            Metrics testM = new Metrics(best.predict(test.X), test.y);
            dtTestAcc[i] = testM.accuracy;
            dtF1[i] = testM.fMeasure;

            System.out.printf("Run %2d | seed=%-12d | TestAcc=%.4f | F1=%.4f%n",
                    i + 1, seed, testM.accuracy, testM.fMeasure);

            if (testM.accuracy > bestDTAcc) {
                bestDTAcc = testM.accuracy;
                bestDTRun = i;
                bestDTInd = best;
            }
        }

        // Summary Statistics
        System.out.println("SUMMARY STATISTICS  (30 runs) ");

        System.out.printf("%n%-20s %8s %8s %8s%n", "Algorithm", "MeanAcc", "StdAcc", "BestAcc");
        System.out.printf("%-20s %8.4f %8.4f %8.4f  (Run %d, seed %d)%n","GP Arithmetic",mean(arithTestAcc), std(arithTestAcc), arithTestAcc[bestArithRun],
                bestArithRun + 1, seeds[bestArithRun]);
        System.out.printf("%-20s %8.4f %8.4f %8.4f  (Run %d, seed %d)%n","GP Decision Tree",mean(dtTestAcc), std(dtTestAcc), dtTestAcc[bestDTRun],
                bestDTRun + 1, seeds[bestDTRun]);

        // Best Run Details
        System.out.println("  BEST RUN DETAILS ");

        System.out.println("\n-Best GP Arithmetic");
        System.out.println("Seed      : " + seeds[bestArithRun]);
        System.out.println("Expression: " + bestArithInd.root);
        Metrics arithTrainM = new Metrics(bestArithInd.predict(train.X), train.y);
        Metrics arithTestM  = new Metrics(bestArithInd.predict(test.X),  test.y);
        System.out.println("Train : " + arithTrainM);
        System.out.println("Test  : " + arithTestM);

        System.out.println("\n Best GP Decision Tree ");
        System.out.println("Seed: " + seeds[bestDTRun]);
        // System.out.print(bestDTInd.root.prettyPrint(""));
        Metrics dtTrainM = new Metrics(bestDTInd.predict(train.X), train.y);
        Metrics dtTestM  = new Metrics(bestDTInd.predict(test.X),  test.y);
        System.out.println("Train : " + dtTrainM);
        System.out.println("Test  : " + dtTestM);

        // Table 2 - Comparison of Classification Performance
        System.out.println("\nTABLE 2 – Comparison of Classification Perf.");
        System.out.printf("%-20s %10s %10s %10s %10s%n",
                "Algorithm", "Train(%)", "Test(%)", "F-measure", "Runtime");
        System.out.printf("%-20s %10.2f %10.2f %10.4f %10s%n",
                "GP Decision Tree",
                dtTrainM.accuracy * 100, dtTestM.accuracy * 100, dtTestM.fMeasure, "see log");
        System.out.printf("%-20s %10.2f %10.2f %10.4f %10s%n",
                "GP Arithmetic",
                arithTrainM.accuracy * 100, arithTestM.accuracy * 100, arithTestM.fMeasure, "see log");

        // Wilcoxon signed-rank test for statistical significance
        System.out.println("\nWILCOXON SIGNED-RANK TEST");
        wilcoxon(arithTestAcc, dtTestAcc);

        // Write results file
        try (PrintWriter pw = new PrintWriter("multirun_results.txt")) {
            pw.println("COS314 Assignment 3 – Multi-Run Results");
            pw.println("Base seed: " + baseSeed);
            pw.printf("GP Arithmetic  best seed: %d (Run %d) | TestAcc=%.4f F1=%.4f%n",
                    seeds[bestArithRun], bestArithRun+1, arithTestAcc[bestArithRun], arithF1[bestArithRun]);
            pw.printf("GP DecisionTree best seed: %d (Run %d) | TestAcc=%.4f F1=%.4f%n",
                    seeds[bestDTRun], bestDTRun+1, dtTestAcc[bestDTRun], dtF1[bestDTRun]);
        }
        System.out.println("\nResults saved to multirun_results.txt");
    }

    // Mean of array
    static double mean(double[] v) {
        double s = 0;
        for (double x : v) {
            s += x;
        }
        return s / v.length;
    }

    // Standard deviation of array
    static double std(double[] v) {
        double m = mean(v), s = 0;
        for (double x : v) {
            s += (x - m) * (x - m);
        }
        return Math.sqrt(s / v.length);
    }

    // Wilcoxon signed-rank test (two-tailed)
    // Null hypothesis: medians are equal
    // Reports W statistic and approximate p-value via normal approximation
    static void wilcoxon(double[] a, double[] b) {
        int n = a.length;
        double[] diff = new double[n];
        for (int i = 0; i < n; i++) {
            diff[i] = a[i] - b[i];
        }

        // Remove zeros, rank remaining absolute differences
        List<Double> nonZero = new ArrayList<>();
        for (double d : diff) {
            if (d != 0) {
                nonZero.add(d);
            }
        }
        int m = nonZero.size();

        if (m == 0) {
            System.out.println("All differences are zero – no test possible.");
            return;
        }

        // Sort by absolute value
        nonZero.sort(Comparator.comparingDouble(Math::abs));

        // Assign ranks (average for ties)
        double[] ranks = new double[m];
        for (int i = 0; i < m; ) {
            int j = i;
            while (j < m && Math.abs(nonZero.get(j)) == Math.abs(nonZero.get(i))) {
                j++;
            }
            double avgRank = (i + 1 + j) / 2.0;
            for (int k = i; k < j; k++) {
                ranks[k] = avgRank;
            }
            i = j;
        }

        double Wplus = 0, Wminus = 0;
        for (int i = 0; i < m; i++) {
            if (nonZero.get(i) > 0) {
                Wplus += ranks[i];
            }
            else {
                Wminus += ranks[i];
            }
        }
        double W = Math.min(Wplus, Wminus);

        // Normal approximation
        double mu    = m * (m + 1) / 4.0;
        double sigma = Math.sqrt(m * (m + 1) * (2 * m + 1) / 24.0);
        double z     = (W - mu) / sigma;
        double pApprox = 2.0 * normalCDF(z);   // two-tailed

        System.out.printf("W+ = %.1f  W- = %.1f  W = %.1f  n = %d%n", Wplus, Wminus, W, m);
        System.out.printf("Normal approx: z = %.4f  p ≈ %.4f%n", z, pApprox);
        if (pApprox < 0.05) {
            System.out.println("Result: SIGNIFICANT difference (p < 0.05)");
        } else {
            System.out.println("Result: No significant difference (p >= 0.05)");
        }
    }

    // Standard normal CDF using error function approximation
    static double normalCDF(double z) {
        return 0.5 * erfc(-z / Math.sqrt(2));
    }

    // Abramowitz & Stegun approximation
    static double erfc(double x) {
        double t = 1.0 / (1.0 + 0.3275911 * Math.abs(x));
        double poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                + t * (-1.453152027 + t * 1.061405429))));
        double result = poly * Math.exp(-x * x);
        if (x >= 0) {
            return result;
        } else {
            return 2.0 - result;
        }
    }
}
