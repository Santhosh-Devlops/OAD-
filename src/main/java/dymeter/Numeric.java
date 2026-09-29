package dymeter;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

/** Shared statistics used by the IEC uncertainty (Eq. 8) and by the DTO rank statistics (Eq. 17-19). */
final class Numeric {
    private Numeric(){}
    /** Empirical CDF inverse: F^-1(q) with the paper's F_N(A)=|W_N|^-1 * sum 1(A_i<=A) (Eq. 17). */
    static double quantile(Collection<Double> values,double q){
        if(values.isEmpty())return Double.NaN;
        List<Double> sorted=new ArrayList<>(values);
        Collections.sort(sorted);
        int index=Math.min(sorted.size()-1,Math.max(0,(int)Math.ceil(q*sorted.size())-1));
        return sorted.get(index);
    }
    /** Median anomaly score of the candidate window W_C, the paper's bAm (Eq. 18). */
    static double median(Collection<Double> values){return quantile(values,0.5);}
    /** Mean anomaly score, used by the paper's DyMETER-A ablation of the median design. */
    static double mean(Collection<Double> values){
        if(values.isEmpty())return Double.NaN;
        double total=0;for(double v:values)total+=v;return total/values.size();
    }
    /** Median absolute deviation of W_N, the paper's delta. */
    static double mad(Collection<Double> values){
        if(values.isEmpty())return Double.NaN;
        double m=median(values),total=0;
        for(double v:values)total+=Math.abs(v-m);
        return total/values.size();
    }
    /** Digamma, the paper's Phi(.), needed by the Dirichlet mutual information of Eq. (8). */
    static double digamma(double x){
        double result=0;
        while(x<8){result-=1/x;x++;}
        double z=1/x,z2=z*z;
        return result+Math.log(x)-0.5*z-z2*(1.0/12-z2*(1.0/120-z2/252));
    }
    /** Numerically stable softplus, log(1+exp(v)), used for Dirichlet evidence. */
    static double softplus(double v){
        if(v>20)return v;
        if(v<-20)return Math.exp(v);
        return Math.log1p(Math.exp(v));
    }
    static double relu(double v){return v>0?v:0;}
    static double reluDerivative(double pre){return pre>0?1:0;}
}
