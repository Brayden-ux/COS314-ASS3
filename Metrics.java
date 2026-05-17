
// Classification metrics: accuracy, precision, recall, F1-measure
public class Metrics {

    public final double accuracy;
    public final double precision;
    public final double recall;
    public final double fMeasure;

    // tp/fp/fn/tn for class=1 (positive = recurrence)
    public final int tp, fp, fn, tn;

    public Metrics(int[] predicted, int[] actual) {
        int tp = 0, fp = 0, fn = 0, tn = 0;
        for (int i = 0; i < actual.length; i++) {
            if (predicted[i] == 1 && actual[i] == 1) tp++;
            else if (predicted[i] == 1 && actual[i] == 0) fp++;
            else if (predicted[i] == 0 && actual[i] == 1) fn++;
            else tn++;
        }
        this.tp = tp;
        this.fp = fp;
        this.fn = fn;
        this.tn = tn;

        accuracy = (double)(tp + tn) / actual.length;
        double prec;
        if (tp + fp == 0) {
            prec = 0.0;
        } else {
            prec = (double) tp / (tp + fp);
        }
        double rec;
        if (tp + fn == 0) {
            rec = 0.0;
        } else {
            rec = (double) tp / (tp + fn);
        }
        precision = prec;
        recall = rec;
        double fMeasure_val;
        if (prec + rec == 0) {
            fMeasure_val = 0.0;
        } else {
            fMeasure_val = 2.0 * prec * rec / (prec + rec);
        }
        fMeasure = fMeasure_val;
    }

    @Override
    public String toString() {
        return String.format("Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f  [TP=%d FP=%d FN=%d TN=%d]",
                accuracy, precision, recall, fMeasure, tp, fp, fn, tn);
    }
}
