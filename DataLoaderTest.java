public class DataLoaderTest {
    public static void main(String[] args) throws Exception {
        DataLoader loader = new DataLoader("Breast_train.csv");
        System.out.println("  DataLoader works");
        System.out.println("  Loaded " + loader.n + " samples");
        System.out.println("  Features per sample: " + loader.X[0].length);
        System.out.println("  First label: " + loader.y[0]);
    }
}