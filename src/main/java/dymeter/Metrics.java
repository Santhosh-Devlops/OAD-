package dymeter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/** AUCROC, AUCPR and decision-quality metrics of the paper's evaluation section. */
final class Metrics {
    private Metrics(){}

    static final class Summary {
        final int rows,positives,negatives,flagged;
        final double aucRoc,aucPr,falsePositiveRate,falseNegativeRate,precision,recall,f1;
        Summary(int rows,int positives,int negatives,int flagged,double aucRoc,double aucPr,double falsePositiveRate,
                double falseNegativeRate,double precision,double recall,double f1){
            this.rows=rows;this.positives=positives;this.negatives=negatives;this.flagged=flagged;this.aucRoc=aucRoc;
            this.aucPr=aucPr;this.falsePositiveRate=falsePositiveRate;this.falseNegativeRate=falseNegativeRate;
            this.precision=precision;this.recall=recall;this.f1=f1;
        }
    }

    /** AUCROC by the Mann-Whitney rank formulation, with 0.5 credit for tied scores. */
    static double aucRoc(boolean[] labels,double[] scores){
        int positives=0,negatives=0;
        List<Integer> order=ascendingOrder(scores);
        double[] ranks=new double[scores.length];
        int index=0;
        while(index<order.size()){
            int end=index;
            while(end+1<order.size()&&scores[order.get(end+1)]==scores[order.get(index)])end++;
            double average=(index+end)/2.0+1;
            for(int k=index;k<=end;k++)ranks[order.get(k)]=average;
            index=end+1;
        }
        double rankSum=0;
        for(int i=0;i<labels.length;i++){
            if(labels[i]){positives++;rankSum+=ranks[i];}else negatives++;
        }
        if(positives==0||negatives==0)return Double.NaN;
        return (rankSum-positives*(positives+1)/2.0)/(positives*(double)negatives);
    }

    /** AUCPR: average precision over the thresholds induced by the distinct scores (step-wise). */
    static double aucPr(boolean[] labels,double[] scores){
        int positives=0;
        for(boolean label:labels)if(label)positives++;
        if(positives==0)return Double.NaN;
        List<Integer> order=ascendingOrder(scores);
        double precisionSum=0,previousRecall=0;
        int truePositives=0,seen=0;
        int index=order.size()-1;
        while(index>=0){
            int end=index;
            while(end-1>=0&&scores[order.get(end-1)]==scores[order.get(index)])end--;
            for(int k=end;k<=index;k++){
                seen++;
                if(labels[order.get(k)])truePositives++;
            }
            double precision=truePositives/(double)seen,recall=truePositives/(double)positives;
            precisionSum+=(recall-previousRecall)*precision;
            previousRecall=recall;
            index=end-1;
        }
        return precisionSum;
    }

    /** Decision quality of the adaptive threshold decisions produced by the DTO. */
    static Summary summarise(boolean[] labels,double[] scores,boolean[] predictions){
        int positives=0,negatives=0,truePositives=0,falsePositives=0,falseNegatives=0,flagged=0;
        for(int i=0;i<labels.length;i++){
            if(labels[i])positives++;else negatives++;
            if(predictions[i]){
                flagged++;
                if(labels[i])truePositives++;else falsePositives++;
            }else if(labels[i])falseNegatives++;
        }
        int trueNegatives=negatives-falsePositives;
        double falsePositiveRate=negatives==0?Double.NaN:falsePositives/(double)negatives;
        double falseNegativeRate=positives==0?Double.NaN:falseNegatives/(double)positives;
        double precision=flagged==0?Double.NaN:truePositives/(double)flagged;
        double recall=positives==0?Double.NaN:truePositives/(double)positives;
        double f1=Double.isNaN(precision)||Double.isNaN(recall)||precision+recall==0?Double.NaN
                :2*precision*recall/(precision+recall);
        return new Summary(labels.length,positives,negatives,flagged,aucRoc(labels,scores),aucPr(labels,scores),
                falsePositiveRate,falseNegativeRate,precision,recall,f1);
    }

    private static List<Integer> ascendingOrder(final double[] scores){
        List<Integer> order=new ArrayList<>(scores.length);
        for(int i=0;i<scores.length;i++)order.add(i);
        order.sort(Comparator.comparingDouble(index -> scores[index]));
        return order;
    }

    static double mean(double[] values){
        if(values.length==0)return Double.NaN;
        double total=0;
        for(double value:values)total+=value;
        return total/values.length;
    }

    static double standardDeviation(double[] values){
        if(values.length<2)return 0;
        double mean=mean(values),total=0;
        for(double value:values)total+=(value-mean)*(value-mean);
        return Math.sqrt(total/(values.length-1));
    }

    static String format(double value){
        return Double.isNaN(value)?"n/a":String.format(java.util.Locale.US,"%.4f",value);
    }

    static boolean[] asBooleanArray(boolean[] labels,int from,int to){return Arrays.copyOfRange(labels,from,to);}
}
