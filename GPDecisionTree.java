import java.io.*;
import java.util.*;

public class GPDecisionTree {
    //parameters
    private static final int POP_SIZE = 75;
    private static final int MAX_GENERATIONS = 75;
    private static final int INIT_MAX_DEPTH = 4;
    private static final int MAX_DEPTH = 8;
    private static final int MUTATION_DEPTH = 3;
    private static final int TOURNAMENT_SIZE = 5;
    private static final double CROSSOVER_RATE = 0.80;
    private static final double MUTATION_RATE = 0.20;
    private static final int NUM_FEATURES = DataLoader.NUM_FEATURES;

    //Tree node
    static class Node{
        boolean isLeaf;
        int featureIdx;
        double threshold;
        String op;
        Node left; //true condition
        Node right; //false
        int label;

        //decision node
        Node(int featureIdx, double threshold, String op, Node left, Node right){
            this.isLeaf = false;
            this.featureIdx = featureIdx;
            this.threshold = threshold;
            this.op = op;
            this.left = left;
            this.right = right;
        }

        //leaf node
        Node(int label){
            this.isLeaf = true;
            this.label = label;
        }

        Node copy(){
            if(isLeaf){
                return new Node(label);
            }
            return new Node(featureIdx, threshold, op, left.copy(), right.copy());
        }

        int classify(double[] x){
            if(isLeaf){
                return label;
            }
            boolean cond = op.equals("<") ? x[featureIdx] < threshold : x[featureIdx] >= threshold;
            return cond ? left.classify(x) : right.classify(x);
        }

        //nodes in subtree
        int size(){
            if(isLeaf) return 1;
            return 1 + left.size() + right.size();
        }

        int depth(){
            if(isLeaf) return 0;
            return 1 + Math.max(left.depth(), right.depth());
        }

        String formatTree(String indent){
            if(isLeaf){
                return indent + "CLASS=" + label + "\n";
            }
            String cond = DataLoader.featureName(featureIdx)
                    + " " + op + " " + String.format("%.2f", threshold);
            return indent + "IF " + cond + ":\n"
                    + left.formatTree(indent + "  |  ")
                    + indent + "ELSE:\n"
                    + right.formatTree(indent + "  |  ");
        }

        @Override
        public String toString(){
            if(isLeaf){
                return "CLASS=" + label;
            } 
            return "IF(" + DataLoader.featureName(featureIdx)
                    + op + String.format("%.2f", threshold)
                    + "," + left + "," + right + ")";
        }
    }

    static class Individual implements Comparable<Individual> {
        Node root;
        double fitness;

        Individual(Node root){
            this.root = root;
        }

        int[] predict(double[][] X){
            int[] preds = new int[X.length];
            for(int i = 0; i < X.length; i++){
                preds[i] = root.classify(X[i]);
            }
            return preds;
        }

        void evaluate(double[][] X, int[] y){
            int correct = 0;
            for(int i = 0; i < y.length; i++){
                if (root.classify(X[i]) == y[i]) correct++;
            }
            fitness = (double) correct / y.length;
        }

        @Override
        public int compareTo(Individual ind){
            return Double.compare(ind.fitness, this.fitness); // descending
        }
    }

    //GP fields
    private final Random rng;
    private final double[][] trainX;
    private final int[] trainY;

    // Per-feature [min, max] computed from training data
    private final double[] featMin;
    private final double[] featMax;

    public GPDecisionTree(long seed, double[][] trainX, int[] trainY){
        this.rng = new Random(seed);
        this.trainX = trainX;
        this.trainY = trainY;
        this.featMin = new double[NUM_FEATURES];
        this.featMax = new double[NUM_FEATURES];
        computeFeatureRanges();
    }

    private void computeFeatureRanges(){
        Arrays.fill(featMin, Double.MAX_VALUE);
        Arrays.fill(featMax, -Double.MAX_VALUE);
        for(double[] row : trainX){
            for(int j = 0; j < NUM_FEATURES; j++){
                if(row[j] < featMin[j]) featMin[j] = row[j];
                if(row[j] > featMax[j]) featMax[j] = row[j];
            }
        }
    }
    
    //sample threshold
    private double randomThreshold(int featureIdx){
        double lo = featMin[featureIdx];
        double hi = featMax[featureIdx];
        double raw = (lo >= hi) ? lo : lo + rng.nextDouble() * (hi - lo);
        return Math.round(raw * 10.0) / 10.0;
    }

    private Node grow(int maxDepth, int currentDepth){
        boolean forceLeaf = (currentDepth >= maxDepth)
                || (currentDepth > 0 && rng.nextDouble() < 0.3);
        return forceLeaf ? randomLeaf() : randomInternal(maxDepth, currentDepth);
    }

    private Node full(int depth, int currentDepth){
        if(currentDepth >= depth) return randomLeaf();
        return randomInternal(depth, currentDepth);
    }

    private Node randomLeaf(){
        return new Node(rng.nextInt(2)); // label 0 or 1
    }

    private Node randomInternal(int maxDepth, int currentDepth){
        int feat = rng.nextInt(NUM_FEATURES);
        double thresh = randomThreshold(feat);
        String op = rng.nextBoolean() ? "<" : ">=";
        Node left = grow(maxDepth, currentDepth + 1);
        Node right = grow(maxDepth, currentDepth + 1);
        return new Node(feat, thresh, op, left, right);
    }
    //ramped half and half
    private List<Individual> initPopulation(){
        List<Individual> pop = new ArrayList<>(POP_SIZE);
        int depthRange = INIT_MAX_DEPTH - 1;  // depths 2, 3, 4
        int halfPop = POP_SIZE / 2;
        for(int i = 0; i < POP_SIZE; i++){
            int d = 2 + (i % depthRange);
            Node root = (i < halfPop) ? full(d, 0) : grow(d, 0);
            pop.add(new Individual(root));
        }
        return pop;
    }

    //selection
    private Individual tournament(List<Individual> pop){
        Individual best = null;
        for(int i = 0; i < TOURNAMENT_SIZE; i++){
            Individual c = pop.get(rng.nextInt(pop.size()));
            if(best == null || c.fitness > best.fitness){
                best = c;
            }
        }
        return best;
    }

    private static Node getNthNode(Node root, int[] counter){
        if(counter[0] == 0){
            counter[0]--;
            return root; 
        }
        counter[0]--;
        if(root.isLeaf) return null;
        Node r = getNthNode(root.left, counter);
        if(r != null){
            return r;
        }
        return getNthNode(root.right, counter);
    }

    //return copy 
    private static Node replaceNode(Node root, int[] counter, Node replacement){
        if(counter[0] == 0){
            counter[0]--; return replacement;
        }
        counter[0]--;
        if(root.isLeaf) return root;
        Node newLeft = replaceNode(root.left,  counter, replacement);
        if(counter[0] < -1){
            return new Node(root.featureIdx, root.threshold, root.op, newLeft, root.right.copy());
        }
        Node newRight = replaceNode(root.right, counter, replacement);
        return new Node(root.featureIdx, root.threshold, root.op, newLeft, newRight);
    }

    //crossover
    private Individual crossover(Individual p1, Individual p2){
        Node child = p1.root.copy();
        Node sub2 = getNthNode(p2.root.copy(), new int[]{rng.nextInt(p2.root.size())});
        if(sub2 == null){
            return new Individual(child);
        }

        int insertIdx = rng.nextInt(child.size());
        Node newRoot = replaceNode(child, new int[]{insertIdx}, sub2.copy());

        // Reject offspring that exceed the depth limit
        if (newRoot.depth() > MAX_DEPTH) newRoot = p1.root.copy();
        return new Individual(newRoot);
    }

    //point mutation
    private Individual mutate(Individual ind){
        Node child = ind.root.copy();
        int idx = rng.nextInt(child.size());

        // Replace the chosen node with a freshly grown subtree (depth <= MUTATION_DEPTH)
        Node newSub = grow(MUTATION_DEPTH, 0);
        Node newRoot = replaceNode(child, new int[]{idx}, newSub);
        if(newRoot.depth() > MAX_DEPTH){
            newRoot = child;
        }
        return new Individual(newRoot);
    }

    public Individual evolve(boolean verbose){
        List<Individual> pop = initPopulation();
        for(Individual ind : pop){
            ind.evaluate(trainX, trainY);
        }
        Collections.sort(pop);

        Individual globalBest = new Individual(pop.get(0).root.copy());
        globalBest.fitness = pop.get(0).fitness;

        for(int gen = 0; gen < MAX_GENERATIONS; gen++){
            List<Individual> nextPop = new ArrayList<>(POP_SIZE);

            // Elitism: carry global best forward unchanged
            Individual elite = new Individual(globalBest.root.copy());
            elite.fitness = globalBest.fitness;
            nextPop.add(elite);

            while(nextPop.size() < POP_SIZE){
                Individual child;
                if(rng.nextDouble() < CROSSOVER_RATE){
                    child = crossover(tournament(pop), tournament(pop));
                }else{
                    child = mutate(tournament(pop));
                }
                child.evaluate(trainX, trainY);
                nextPop.add(child);
            }

            pop = nextPop;
            Collections.sort(pop);

            Individual genBest = pop.get(0);
            if(genBest.fitness > globalBest.fitness){
                globalBest = new Individual(genBest.root.copy());
                globalBest.fitness = genBest.fitness;
            }

            if(verbose){
                Metrics m = new Metrics(globalBest.predict(trainX), trainY);
                System.out.printf("Gen %3d | BestFit=%.4f | PopBest=%.4f | Depth=%d | %s%n",
                        gen + 1, globalBest.fitness, genBest.fitness,
                        globalBest.root.depth(), m);
            }

            // Early stop if perfect training accuracy
            if(globalBest.fitness >= 1.0) break;
        }
        return globalBest;
    }

    public static void main(String[] args) throws Exception{
        Scanner scanner = new Scanner(System.in);
 
        System.out.print("Enter random seed (long): ");
        long seed = scanner.nextLong();
 
        String trainPath = "Breast_train.csv";
        String testPath  = "Breast_test.csv";
 
        System.out.println("\nLoading data...");
        DataLoader train = new DataLoader(trainPath);
        DataLoader test = new DataLoader(testPath);
        System.out.printf("Train: %d samples | Test: %d samples | Features: %d%n", train.n, test.n, DataLoader.NUM_FEATURES);
 
        System.out.println("\n=== GP DECISION TREE CLASSIFIER - TRAINING ===");
        System.out.printf("Seed=%d | PopSize=%d | MaxGen=%d | MaxDepth=%d | CrossRate=%.0f%% | MutRate=%.0f%%%n",
                seed, POP_SIZE, MAX_GENERATIONS, MAX_DEPTH,
                CROSSOVER_RATE * 100, MUTATION_RATE * 100);
        System.out.println();
 
        long startTime = System.currentTimeMillis();
        GPDecisionTree gp = new GPDecisionTree(seed, train.X, train.y);
        Individual best = gp.evolve(true);
        long elapsed = System.currentTimeMillis() - startTime;
 
        System.out.println("\n=== BEST INDIVIDUAL ===");
        System.out.println("\nDecision Tree Structure:\n");
        System.out.print(best.root.formatTree(""));
 
        Metrics trainMetrics = new Metrics(best.predict(train.X), train.y);
        Metrics testMetrics = new Metrics(best.predict(test.X),  test.y);
 
        System.out.println("\n--- Training Metrics ---");
        System.out.println(trainMetrics);
        System.out.println("--- Test Metrics ---");
        System.out.println(testMetrics);
        System.out.printf("%nRuntime: %.2f seconds%n", elapsed / 1000.0);
 
        System.out.println("\n--- Table 2 Row (for report) ---");
        System.out.printf("GP Decision Tree | Train=%.2f%% | Test=%.2f%% | F1=%.4f | Runtime=%.2fs%n",
                trainMetrics.accuracy * 100, testMetrics.accuracy * 100,
                testMetrics.fMeasure, elapsed / 1000.0);
 
        try(PrintWriter pw = new PrintWriter("best_decisiontree_model.txt")){
            pw.println("Seed: " + seed);
            pw.println("Tree depth: " + best.root.depth());
            pw.println("Tree size (nodes): " + best.root.size());
            pw.println("\nDecision Tree Structure:");
            pw.print(best.root.formatTree(""));
            pw.printf("%nTrain accuracy : %.4f%n", trainMetrics.accuracy);
            pw.printf("Test accuracy : %.4f%n", testMetrics.accuracy);
            pw.printf("F-measure : %.4f%n", testMetrics.fMeasure);
            pw.printf("Runtime (s) : %.2f%n", elapsed / 1000.0);
        }
        System.out.println("\nModel saved to best_decisiontree_model.txt");
    }
}