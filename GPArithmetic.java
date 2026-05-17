import java.io.*;
import java.util.*;

public class GPArithmetic{

    private static final int POP_SIZE = 200;
    private static final int MAX_GENERATIONS = 100;
    private static final int INIT_MAX_DEPTH = 4;
    private static final int MAX_DEPTH = 8;
    private static final int TOURNAMENT_SIZE = 5;
    private static final double CROSSOVER_RATE = 0.80;
    private static final double MUTATION_RATE = 0.20;
    private static final int NUM_FEATURES = DataLoader.NUM_FEATURES;
    private static final String[] FUNCTIONS = {"+","-","*","/"};

    static class Node {
        boolean isFunction;
        String  operator;
        int featureIdx;  //>= 0 : terminal variable
        double  constant; //used when featureIdx == -1
        boolean isConst;
        Node left, right;

        //Function node 
        Node(String operator, Node left, Node right){
            isFunction = true;
            this.operator = operator;
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
            return new Node(operator, left.copy(), right.copy());
        }

        //Evaluate expression on one sample
        double eval(double[] x){

            if (!isFunction) return isConst ? constant : x[featureIdx];
            
            double l = left.eval(x);
            double r = right.eval(x);

            switch (operator) {
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
            return "(" + left + " " + operator + " " + right + ")";
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

    public GPArithmetic(long seed, double[][] trainX, int[] trainY) {
        this.rng = new Random(seed);
        this.trainX = trainX;
        this.trainY = trainY;
    }
    
}