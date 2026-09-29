package dymeter;

import java.util.Random;

/**
 * Dense array helpers. A weight matrix w is stored as w[input][output], matching the
 * convention used throughout the port: y[j] = sum_i x[i] * w[i][j] + b[j].
 */
final class Matrix {
    private Matrix(){}
    static double[][] zeros(int rows,int columns){return new double[rows][columns];}
    static double[] zeros(int size){return new double[size];}
    /** y = W x + b. */
    static double[] affine(double[][] w,double[] b,double[] x){
        double[] y=b.clone();
        for(int i=0;i<x.length;i++){
            double xi=x[i];
            if(xi==0)continue;
            double[] row=w[i];
            for(int j=0;j<y.length;j++)y[j]+=xi*row[j];
        }
        return y;
    }
    /** dL/dx = W dL/dy for the same layout. */
    static double[] affineTranspose(double[][] w,double[] dy){
        double[] dx=new double[w.length];
        for(int i=0;i<w.length;i++){
            double sum=0;double[] row=w[i];
            for(int j=0;j<row.length;j++)sum+=row[j]*dy[j];
            dx[i]=sum;
        }
        return dx;
    }
    /** w[i][j] += scale * x[i] * y[j], the outer-product gradient of an affine layer. */
    static void outerAdd(double[][] w,double[] x,double[] y,double scale){
        for(int i=0;i<x.length;i++){
            if(x[i]==0)continue;
            double s=scale*x[i],row[]=w[i];
            for(int j=0;j<y.length;j++)row[j]+=s*y[j];
        }
    }
    static void addInPlace(double[] target,double[] values){for(int i=0;i<target.length;i++)target[i]+=values[i];}
    static void addInPlace(double[][] target,double[][] values){
        for(int i=0;i<target.length;i++)for(int j=0;j<target[i].length;j++)target[i][j]+=values[i][j];
    }
    static void scaleInPlace(double[] target,double factor){for(int i=0;i<target.length;i++)target[i]*=factor;}
    static void fill(double[][] target,double value){for(double[] row:target)java.util.Arrays.fill(row,value);}
    static double squaredDistance(double[] a,double[] b){
        double sum=0;for(int i=0;i<a.length;i++){double d=a[i]-b[i];sum+=d*d;}return sum;
    }
    /** He initialisation for ReLU layers with fan-in 'inputs'. */
    static void heInit(double[][] w,Random random){
        double scale=Math.sqrt(2.0/w.length);
        for(double[] row:w)for(int j=0;j<row.length;j++)row[j]=random.nextGaussian()*scale;
    }
    static void xavierInit(double[][] w,Random random){
        double scale=Math.sqrt(2.0/(w.length+w[0].length));
        for(double[] row:w)for(int j=0;j<row.length;j++)row[j]=random.nextGaussian()*scale;
    }
    static double[] copy(double[] values){return values.clone();}
    static double[][] copy(double[][] values){
        double[][] out=new double[values.length][];
        for(int i=0;i<values.length;i++)out[i]=values[i].clone();
        return out;
    }
    static double[][][] copy(double[][][] values){
        double[][][] out=new double[values.length][][];
        for(int i=0;i<values.length;i++)out[i]=copy(values[i]);
        return out;
    }
    static double[][] subtract(double[][] first,double[][] second){
        double[][] out=new double[first.length][];
        for(int i=0;i<first.length;i++){
            out[i]=new double[first[i].length];
            for(int j=0;j<first[i].length;j++)out[i][j]=first[i][j]-second[i][j];
        }
        return out;
    }
    static double[][] add(double[][] first,double[][] second){
        double[][] out=copy(first);addInPlace(out,second);return out;
    }
    static String toString(double[] values,int precision){
        StringBuilder b=new StringBuilder("[");
        for(int i=0;i<values.length;i++){if(i>0)b.append(", ");b.append(String.format(java.util.Locale.US,"%."+precision+"f",values[i]));}
        return b.append(']').toString();
    }
}
