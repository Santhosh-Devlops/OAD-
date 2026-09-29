package dymeter;

import java.util.Arrays;
import java.util.List;

/**
 * Principal component analysis used to size the SCD latent width: the paper sets the latent
 * dimension to the number of principal components that explain at least 70% of the variance.
 * Implemented with a Jacobi eigen-decomposition of the covariance matrix, no external libraries.
 */
final class Pca {
    private Pca(){}
    static final class Result {
        final int components;
        final double[] eigenvalues;
        final double explainedVariance;
        Result(int components,double[] eigenvalues,double explainedVariance){
            this.components=components;this.eigenvalues=eigenvalues;this.explainedVariance=explainedVariance;
        }
    }

    static Result analyse(List<double[]> rows,double targetVariance){
        int dimensions=rows.get(0).length,count=rows.size();
        double[] mean=new double[dimensions];
        for(double[] row:rows)for(int j=0;j<dimensions;j++)mean[j]+=row[j]/count;
        double[][] covariance=Matrix.zeros(dimensions,dimensions);
        for(double[] row:rows)for(int i=0;i<dimensions;i++){
            double left=row[i]-mean[i];
            for(int j=0;j<dimensions;j++)covariance[i][j]+=left*(row[j]-mean[j]);
        }
        for(int i=0;i<dimensions;i++)for(int j=0;j<dimensions;j++)covariance[i][j]/=count;
        double[] eigenvalues=jacobiEigenvalues(covariance);
        Arrays.sort(eigenvalues);
        double total=0;
        for(double value:eigenvalues)total+=value;
        double running=0;
        int components=dimensions;
        for(int k=1;k<=dimensions;k++){
            running+=eigenvalues[dimensions-k];
            if(total>0&&running/total>=targetVariance){components=k;break;}
        }
        double ratio=total>0?running/total:1;
        return new Result(Math.max(1,components),eigenvalues,ratio);
    }

    /** Cyclic Jacobi rotation sweep; returns the diagonal (eigenvalues) of a symmetric matrix. */
    private static double[] jacobiEigenvalues(double[][] matrix){
        int n=matrix.length;
        double[][] a=Matrix.copy(matrix);
        for(int iteration=0;iteration<Math.max(1,n*n*20);iteration++){
            int p=0,q=1;
            double maximum=0;
            for(int i=0;i<n;i++)for(int j=i+1;j<n;j++)if(Math.abs(a[i][j])>maximum){maximum=Math.abs(a[i][j]);p=i;q=j;}
            if(maximum<1e-12)break;
            double angle=0.5*Math.atan2(2*a[p][q],a[q][q]-a[p][p]);
            double c=Math.cos(angle),s=Math.sin(angle);
            double app=c*c*a[p][p]-2*s*c*a[p][q]+s*s*a[q][q];
            double aqq=s*s*a[p][p]+2*s*c*a[p][q]+c*c*a[q][q];
            for(int k=0;k<n;k++)if(k!=p&&k!=q){
                double akp=c*a[k][p]-s*a[k][q],akq=s*a[k][p]+c*a[k][q];
                a[k][p]=a[p][k]=akp;
                a[k][q]=a[q][k]=akq;
            }
            a[p][p]=app;a[q][q]=aqq;a[p][q]=a[q][p]=0;
        }
        double[] eigenvalues=new double[n];
        for(int i=0;i<n;i++)eigenvalues[i]=Math.max(0,a[i][i]);
        return eigenvalues;
    }
}
