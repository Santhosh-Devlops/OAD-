package dymeter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

/**
 * Evaluation metrics for DyMETER.
 *
 * Ground-truth metrics:
 *   AUCROC
 *   AUCPR
 *   Precision
 *   Recall
 *   F1
 *   FPR
 *
 * Dataset-independent metrics:
 *   Mean score
 *   Score standard deviation
 *   Mean threshold
 *   Flagged ratio
 */
final class Metrics {

    private Metrics() {}

    static final class Summary {

        final int rows;
        final int positives;
        final int negatives;
        final int flagged;

        final double aucRoc;
        final double aucPr;

        final double falsePositiveRate;
        final double falseNegativeRate;

        final double precision;
        final double recall;
        final double f1;

        final double meanScore;
        final double stdScore;
        final double meanThreshold;
        final double flaggedRate;

        final boolean labelled;
        final boolean aucAvailable;

        final String metricStatus;

        Summary(
                int rows,
                int positives,
                int negatives,
                int flagged,
                double aucRoc,
                double aucPr,
                double falsePositiveRate,
                double falseNegativeRate,
                double precision,
                double recall,
                double f1,
                double meanScore,
                double stdScore,
                double meanThreshold,
                double flaggedRate,
                boolean labelled,
                boolean aucAvailable,
                String metricStatus
        ) {

            this.rows = rows;
            this.positives = positives;
            this.negatives = negatives;
            this.flagged = flagged;

            this.aucRoc = aucRoc;
            this.aucPr = aucPr;

            this.falsePositiveRate = falsePositiveRate;
            this.falseNegativeRate = falseNegativeRate;

            this.precision = precision;
            this.recall = recall;
            this.f1 = f1;

            this.meanScore = meanScore;
            this.stdScore = stdScore;
            this.meanThreshold = meanThreshold;
            this.flaggedRate = flaggedRate;

            this.labelled = labelled;
            this.aucAvailable = aucAvailable;
            this.metricStatus = metricStatus;
        }
    }

    /**
     * Complete evaluation.
     *
     * AUCROC/AUCPR require both positive and negative ground-truth
     * examples. They are NOT fabricated for unlabelled datasets.
     */
    static Summary summarise(
            boolean[] labels,
            double[] scores,
            boolean[] predictions,
            double[] thresholds
    ) {

        int rows = scores.length;

        boolean labelled = labels != null && labels.length == rows;

        int positives = 0;
        int negatives = 0;

        int truePositives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;

        int flagged = 0;

        if (labelled) {

            for (int i = 0; i < rows; i++) {

                if (labels[i]) {
                    positives++;
                } else {
                    negatives++;
                }

                if (predictions[i]) {

                    flagged++;

                    if (labels[i]) {
                        truePositives++;
                    } else {
                        falsePositives++;
                    }

                } else if (labels[i]) {

                    falseNegatives++;
                }
            }

        } else {

            for (boolean prediction : predictions) {
                if (prediction) {
                    flagged++;
                }
            }
        }

        boolean aucAvailable =
                labelled &&
                positives > 0 &&
                negatives > 0;

        double aucRoc = Double.NaN;
        double aucPr = Double.NaN;

        double fpr = Double.NaN;
        double fnr = Double.NaN;
        double precision = Double.NaN;
        double recall = Double.NaN;
        double f1 = Double.NaN;

        if (labelled) {

            if (aucAvailable) {
                aucRoc = aucRoc(labels, scores);
                aucPr = aucPr(labels, scores);
            }

            if (negatives > 0) {
                fpr = falsePositives / (double) negatives;
            }

            if (positives > 0) {
                fnr = falseNegatives / (double) positives;
                recall = truePositives / (double) positives;
            }

            if (flagged > 0) {
                precision = truePositives / (double) flagged;
            }

            if (!Double.isNaN(precision)
                    && !Double.isNaN(recall)
                    && precision + recall > 0) {

                f1 = 2.0 * precision * recall
                        / (precision + recall);
            }
        }

        double meanScore = mean(scores);
        double stdScore = standardDeviation(scores);
        double meanThreshold = mean(thresholds);

        double flaggedRate =
                rows == 0 ? 0.0 : flagged / (double) rows;

        String metricStatus;

        if (!labelled) {

            metricStatus =
                    "Ranking and classification metrics require a ground-truth label column; " +
                    "score and threshold summaries are available.";

        } else if (!aucAvailable) {

            metricStatus =
                    "The evaluated rows contain only one label class; both normal and anomalous examples " +
                    "are required to calculate ranking metrics.";

        } else {

            metricStatus = "Ground-truth evaluation available.";
        }

        return new Summary(
                rows,
                positives,
                negatives,
                flagged,
                aucRoc,
                aucPr,
                fpr,
                fnr,
                precision,
                recall,
                f1,
                meanScore,
                stdScore,
                meanThreshold,
                flaggedRate,
                labelled,
                aucAvailable,
                metricStatus
        );
    }

    /**
     * Backward-compatible helper.
     */
    static Summary summarise(
            boolean[] labels,
            double[] scores,
            boolean[] predictions
    ) {

        double[] thresholds = new double[scores.length];

        Arrays.fill(thresholds, Double.NaN);

        return summarise(
                labels,
                scores,
                predictions,
                thresholds
        );
    }

    /**
     * AUCROC using the Mann-Whitney rank formulation.
     */
    static double aucRoc(
            boolean[] labels,
            double[] scores
    ) {

        int positives = 0;
        int negatives = 0;

        List<Integer> order = ascendingOrder(scores);

        double[] ranks = new double[scores.length];

        int index = 0;

        while (index < order.size()) {

            int end = index;

            while (
                    end + 1 < order.size()
                    && Double.compare(
                    scores[order.get(end + 1)],
                    scores[order.get(index)]
            ) == 0
            ) {
                end++;
            }

            double average =
                    (index + end) / 2.0 + 1.0;

            for (int k = index; k <= end; k++) {
                ranks[order.get(k)] = average;
            }

            index = end + 1;
        }

        double rankSum = 0;

        for (int i = 0; i < labels.length; i++) {

            if (labels[i]) {

                positives++;
                rankSum += ranks[i];

            } else {

                negatives++;
            }
        }

        if (positives == 0 || negatives == 0) {
            return Double.NaN;
        }

        return (
                rankSum
                        - positives * (positives + 1) / 2.0
        ) / (positives * (double) negatives);
    }

    /**
     * Average precision / PR-AUC.
     */
    static double aucPr(
            boolean[] labels,
            double[] scores
    ) {

        int positives = 0;

        for (boolean label : labels) {

            if (label) {
                positives++;
            }
        }

        if (positives == 0) {
            return Double.NaN;
        }

        List<Integer> order = ascendingOrder(scores);

        double precisionSum = 0.0;
        double previousRecall = 0.0;

        int truePositives = 0;
        int seen = 0;

        int index = order.size() - 1;

        while (index >= 0) {

            int end = index;

            while (
                    end - 1 >= 0
                    && Double.compare(
                    scores[order.get(end - 1)],
                    scores[order.get(index)]
            ) == 0
            ) {
                end--;
            }

            for (int k = end; k <= index; k++) {

                seen++;

                if (labels[order.get(k)]) {
                    truePositives++;
                }
            }

            double precision =
                    truePositives / (double) seen;

            double recall =
                    truePositives / (double) positives;

            precisionSum +=
                    (recall - previousRecall)
                            * precision;

            previousRecall = recall;

            index = end - 1;
        }

        return precisionSum;
    }

    static double mean(double[] values) {

        if (values == null || values.length == 0) {
            return Double.NaN;
        }

        double total = 0;

        int count = 0;

        for (double value : values) {

            if (!Double.isFinite(value)) {
                continue;
            }

            total += value;
            count++;
        }

        return count == 0
                ? Double.NaN
                : total / count;
    }

    static double standardDeviation(double[] values) {

        if (values == null || values.length < 2) {
            return 0;
        }

        double average = mean(values);

        double total = 0;
        int count = 0;

        for (double value : values) {

            if (!Double.isFinite(value)) {
                continue;
            }

            total +=
                    (value - average)
                            * (value - average);

            count++;
        }

        if (count < 2) {
            return 0;
        }

        return Math.sqrt(total / (count - 1));
    }

    private static List<Integer> ascendingOrder(
            final double[] scores
    ) {

        List<Integer> order =
                new ArrayList<>(scores.length);

        for (int i = 0; i < scores.length; i++) {
            order.add(i);
        }

        order.sort(
                Comparator.comparingDouble(
                        index -> scores[index]
                )
        );

        return order;
    }

    static String format(double value) {

        if (!Double.isFinite(value)) {
            return "Not scored";
        }

        return String.format(
                Locale.US,
                "%.4f",
                value
        );
    }

    static boolean[] asBooleanArray(
            boolean[] labels,
            int from,
            int to
    ) {

        return Arrays.copyOfRange(
                labels,
                from,
                to
        );
    }
}