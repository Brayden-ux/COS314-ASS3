import java.io.*;
import java.util.*;

public class DataLoader {

    public static final int NUM_FEATURES = 9;

    public final double[][] X;   // [samples][features]
    public final int[] y;    // [samples]
    public final int  n;    // number of samples

    public DataLoader(String filepath) throws IOException {
        List<double[]> rows  = new ArrayList<>();
        List<Integer>  labels = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filepath))) {
            String line = br.readLine();
            if (line == null) {
                throw new IOException("Empty file: " + filepath);
            } 

            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] parts = line.split(",");
                if (parts.length < NUM_FEATURES + 1) continue;

                labels.add(Integer.parseInt(parts[0].trim()));
                double[] feats = new double[NUM_FEATURES];
                for (int i = 0; i < NUM_FEATURES; i++) {
                    feats[i] = Double.parseDouble(parts[i + 1].trim());
                }
                rows.add(feats);
            }
        }

        n = rows.size();
        X = rows.toArray(new double[0][]);
        y = new int[n];
        for (int i = 0; i < n; i++) {
            y[i] = labels.get(i);
        }
    }

    // Feature names for display purposes
    public static String featureName(int idx) {
        String[] names = {"age", "menopause", "tumor_size", "inv_nodes",
                "node_caps", "deg_malig", "breast", "breast_quad", "irradiat"};
        return (idx >= 0 && idx < names.length) ? names[idx] : "x" + idx;
    }
}
