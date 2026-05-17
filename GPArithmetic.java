import java.io.*;
import java.util.*;

public class GPArithmetic{

    private static final int POP_SIZE = 75;
    private static final int MAX_GENERATIONS = 75;
    private static final int INIT_MAX_DEPTH = 4;
    private static final int MAX_DEPTH = 8;
    private static final int TOURNAMENT_SIZE = 5;
    private static final double CROSSOVER_RATE = 0.80;
    private static final double MUTATION_RATE = 0.20;
    private static final int NUM_FEATURES = DataLoader.NUM_FEATURES;
    private static final String[] FUNCTIONS = {"+","-","*","/"};

    static class Node {
        boolean isFunction;
        String op;
        int featureIdx;  //>= 0 : terminal variable
        double constant; //used when featureIdx == -1
        boolean isConst;
        Node left, right;

        //Function node 
        Node(String op, Node left, Node right){
            isFunction = true;
            this.op = op;
            this.left = left; 
            this.right = right;
        }

        // Variable terminal
        Node(int featureIdx){
            isFunction = false; 
            this.featureIdx = featureIdx; 
            isConst = false;
        }

        //Constant terminal
        Node(double constant){
            isFunction = false; 
            this.constant = constant;
            featureIdx = -1; 
            isConst = true;
        }

        //Deep copy
        Node copy() {

            if (!isFunction){
                if(isConst) return new Node(constant);
                return new Node(featureIdx);
            }
            return new Node(op, left.copy(), right.copy());
        }

        //Evaluate expression on one sample
        double eval(double[] x){

            if (!isFunction) return isConst ? constant : x[featureIdx];
            
            double l = left.eval(x);
            double r = right.eval(x);

            switch (op) {
                case "+": return l + r;
                case "-": return l - r;
                case "*": return l * r;
                case "/": return (Math.abs(r) < 1e-10) ? 1.0 : l / r;
                default:  return 0.0;
            }
        }

        // Count nodes
        int size() {

            if (!isFunction) return 1;
            return 1 + left.size() + right.size();
        }

        // Depth of tree
        int depth() {
            if (!isFunction) return 0;
            return 1 + Math.max(left.depth(), right.depth());
        }

        //Readable expression
        @Override
        public String toString() {
            if (!isFunction) return isConst? String.format("%.3f", constant) : DataLoader.featureName(featureIdx);
            return "(" + left + " " + op + " " + right + ")";
        }
    }

    static class Individual implements Comparable<Individual> {
        Node root;
        double fitness;

        Individual(Node root) {
             this.root = root;
        }

        int[] predict(double[][] X) {

            int[] preds = new int[X.length];
            for (int i = 0; i < X.length; i++)
                preds[i] = (root.eval(X[i]) > 0) ? 1 : 0;
            return preds;
        }

        void evaluate(double[][] X, int[] y) {
            int correct = 0;
            for (int i=0; i<y.length; i++){
                if(predict(new double[][]{X[i]})[0] == y[i]) correct++;
            }

            fitness = (double) correct/y.length;
        }

        @Override public int compareTo(Individual other) {
            return Double.compare(other.fitness, this.fitness); // descending
        }
    }

    private final Random rng;
    private final double[][] trainX;
    private final int[] trainY;

    public GPArithmetic(long seed, double[][] trainX, int[] trainY) {
        this.rng = new Random(seed);
        this.trainX = trainX;
        this.trainY = trainY;
    }

    private Node grow(int maxDepth, int currentDepth) {
        boolean leaf = (currentDepth >= maxDepth) || (currentDepth > 0 && rng.nextDouble() < 0.3);

        if(leaf) return randomTerminal();
        String op = FUNCTIONS[rng.nextInt(FUNCTIONS.length)];
        return new Node(op, grow(maxDepth, currentDepth + 1), grow(maxDepth, currentDepth + 1));
    }

    private Node full(int depth, int currentDepth) {
        if(currentDepth >= depth) return randomTerminal();

        String op = FUNCTIONS[rng.nextInt(FUNCTIONS.length)];
        return new Node(op, full(depth, currentDepth + 1), full(depth, currentDepth + 1));
    }

    private Node randomTerminal(){
        //70% chance of variable, 30% constant (range -5 to 5)
        if(rng.nextDouble() < 0.7){
            return new Node(rng.nextInt(NUM_FEATURES));
        }

        return new Node(rng.nextDouble() * 10 - 5);
    }

    //ramped half and half popullation initialization
    private List<Individual> initPopulation() {

        List<Individual> pop = new ArrayList<>();

        int depthRange = INIT_MAX_DEPTH - 1;
        int halfPop = POP_SIZE/2;

        for(int i=0; i<POP_SIZE; i++){
            int d = 2 + (i % depthRange);
            Node root = (i < halfPop) ? full(d, 0) : grow(d, 0);
            pop.add(new Individual(root));
        }

        return pop;
    }

    //tournemant selection
    private Individual tournament(List<Individual> pop) {
        Individual best = null;

        for (int i=0; i<TOURNAMENT_SIZE; i++){
            Individual c = pop.get(rng.nextInt(pop.size()));

            if (best==null || c.fitness>best.fitness) best = c;
        }

        return best;
    }

    //Crossover helpers
    private static Node getNthNode(Node root, int[] counter){

        if(counter[0]==0){ 
            counter[0]--; 
            return root; 
        }

        counter[0]--;
        if(!root.isFunction) return null;

        Node r = getNthNode(root.left, counter);
        if (r != null) return r;

        return getNthNode(root.right, counter);
    }

    private Node getRandomSubtree(Node root){
        int size = root.size();
        int idx  = rng.nextInt(size);
        int[] c  = {idx};

        return getNthNode(root, c);
    }

    private Node replaceNode(Node root, int[] counter, Node replacement) {
        if (counter[0]==0){ 
            counter[0]--; 
            return replacement; 
        }

        counter[0]--;
        if (!root.isFunction) return root;
        Node newLeft  = replaceNode(root.left,  counter, replacement);

        if (counter[0]<-1) {
            return new Node(root.op, newLeft, root.right.copy());
        }

        Node newRight = replaceNode(root.right, counter, replacement);
        return new Node(root.op, newLeft, newRight);
    }

    //Crossover
    private Individual crossover(Individual parent1, Individual parent2){

        Node child = parent1.root.copy();
        Node sub2  = getRandomSubtree(parent2.root).copy();

        int idx = rng.nextInt(child.size());
        int[] c = {idx};
        Node newRoot = replaceNode(child, c, sub2);

        //depth limit enforcement
        if (newRoot.depth() > MAX_DEPTH) newRoot = parent1.root.copy();
        return new Individual(newRoot);
    }

    //Mutation
    private Individual mutate(Individual sol){

        Node child = sol.root.copy();
        int  size = child.size();
        int  idx = rng.nextInt(size);

        Node newSubtree = grow(3, 0);   //mutation offspring depth=3
        int[] c = {idx};
        Node newRoot = replaceNode(child, c, newSubtree);

        if(newRoot.depth() > MAX_DEPTH) newRoot = child;
        return new Individual(newRoot);
    }

    public Individual evolve(boolean verbose){
        List<Individual> pop = initPopulation();

        for(Individual sol : pop){ 
            sol.evaluate(trainX, trainY);
            }

        Collections.sort(pop);

        Individual globalBest = pop.get(0);

        for(int gen=0; gen<MAX_GENERATIONS; gen++){

            List<Individual> newPop = new ArrayList<>();

            //Elitism
            newPop.add(new Individual(globalBest.root.copy()));
            newPop.get(0).fitness = globalBest.fitness;

            while(newPop.size()<POP_SIZE){
                Individual child;
                double r = rng.nextDouble();

                if(r<CROSSOVER_RATE){
                    
                    child = crossover(tournament(pop), tournament(pop));

                }else{
                    child = mutate(tournament(pop));
                }
                child.evaluate(trainX, trainY);
                newPop.add(child);
            }

            pop = newPop;
            Collections.sort(pop);

            if(pop.get(0).fitness>globalBest.fitness){
                globalBest = pop.get(0);
            }

            if (verbose){
                Metrics m = new Metrics(globalBest.predict(trainX), trainY);

                System.out.printf("Gen %3d | BestFit=%.4f | %s%n", gen + 1, globalBest.fitness, m);
                System.out.println("        Expression: " + globalBest.root+"\n");
            }
        }

        return globalBest;
    }

    public static void main(String[] args) throws Exception {

        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter random seed(long): ");
        long seed = scanner.nextLong();

        System.out.print("Enter training CSV filepath: ");
        String trainPath = scanner.next();

        System.out.print("Enter test CSV filepath: ");
        String testPath = scanner.next();

        System.out.println("\nLoading data");
        DataLoader train = new DataLoader(trainPath);
        DataLoader test  = new DataLoader(testPath);
        System.out.printf("Train: %d samples | Test: %d samples%n", train.n, test.n);

        System.out.println("\n=== GP ARITHMETIC CLASSIFIER – TRAINING ===");
        System.out.printf("Seed=%d | PopSize=%d | MaxGen=%d | CrossRate=%.0f%% | MutRate=%.0f%%%n", seed, POP_SIZE, MAX_GENERATIONS,
            CROSSOVER_RATE*100, MUTATION_RATE*100);
        System.out.println();

        long startTime = System.currentTimeMillis();
        GPArithmetic gp = new GPArithmetic(seed, train.X, train.y);

        Individual best = gp.evolve(true);
        long timeElapsed = System.currentTimeMillis()-startTime;

        System.out.println("\n=== FINAL RESULTS ===");
        System.out.println("Best Expression: "+best.root);

        Metrics trainMetrics = new Metrics(best.predict(train.X), train.y);
        Metrics testMetrics  = new Metrics(best.predict(test.X),  test.y);

        System.out.println("\n--- Training ---");
        System.out.println(trainMetrics);
        System.out.println("--- Test ---");
        System.out.println(testMetrics);
        System.out.printf("%nRuntime: %.2f seconds%n", timeElapsed / 1000.0);

        System.out.printf("%nSummary:%n");
        System.out.printf("GP Arithmetic | Train=%.2f%% | Test=%.2f%% | F1=%.4f | Runtime=%.2fs%n", trainMetrics.accuracy * 100, testMetrics.accuracy * 100, testMetrics.fMeasure, timeElapsed / 1000.0);

        // Save best model expression to file
        try (PrintWriter pw = new PrintWriter("best_arithmetic_model.txt")) {
            pw.println("Seed: " + seed);
            pw.println("Expression: " + best.root);
            pw.printf("Train accuracy: %.4f%n", trainMetrics.accuracy);
            pw.printf("Test  accuracy: %.4f%n", testMetrics.accuracy);
            pw.printf("F-measure     : %.4f%n", testMetrics.fMeasure);
            pw.printf("Runtime (s)   : %.2f%n", timeElapsed / 1000.0);
        }
        System.out.println("\nModel saved to best_arithmetic_model.txt");
    }
    
}