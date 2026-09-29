package dymeter;

import java.util.Arrays;

/**
 * Adam optimizer (Kingma and Ba) used by every learnable module.
 * The paper trains DyMETER with "Adam ... initial learning rate of 1e-2, decayed exponentially
 * by a factor of 0.96 over 2000 epochs"; the exponential schedule lives in the training loops,
 * this class holds the per-parameter first/second moment state.
 */
final class Adam {
    private static final double BETA1=0.9,BETA2=0.999,EPS=1e-8;
    private final double[] first,second;
    private final double[][] firstMatrix,secondMatrix;
    private int steps;

    Adam(int size){
        first=new double[size];
        second=new double[size];
        firstMatrix=null;
        secondMatrix=null;
    }

    /** Optimiser state for a weight matrix stored as weight[input][output]. */
    Adam(int rows,int columns){
        first=null;
        second=null;
        firstMatrix=new double[rows][columns];
        secondMatrix=new double[rows][columns];
    }

    /** One Adam step: p <- p - lr * mHat / (sqrt(vHat) + eps). */
    void step(double[] parameter,double[] gradient,double learningRate){
        steps++;
        double bias1=1-Math.pow(BETA1,steps),bias2=1-Math.pow(BETA2,steps),lr=learningRate;
        for(int i=0;i<parameter.length;i++){
            double g=gradient[i];
            first[i]=BETA1*first[i]+(1-BETA1)*g;
            second[i]=BETA2*second[i]+(1-BETA2)*g*g;
            parameter[i]-=lr*(first[i]/bias1)/(Math.sqrt(second[i]/bias2)+EPS);
        }
    }

    void step(double[][] parameter,double[][] gradient,double learningRate){
        steps++;
        double bias1=1-Math.pow(BETA1,steps),bias2=1-Math.pow(BETA2,steps),lr=learningRate;
        for(int i=0;i<parameter.length;i++){
            double[] row=parameter[i],gradientRow=gradient[i],firstRow=firstMatrix[i],secondRow=secondMatrix[i];
            for(int j=0;j<row.length;j++){
                double g=gradientRow[j];
                firstRow[j]=BETA1*firstRow[j]+(1-BETA1)*g;
                secondRow[j]=BETA2*secondRow[j]+(1-BETA2)*g*g;
                row[j]-=lr*(firstRow[j]/bias1)/(Math.sqrt(secondRow[j]/bias2)+EPS);
            }
        }
    }

    void reset(){
        steps=0;
        if(first!=null){Arrays.fill(first,0);Arrays.fill(second,0);}
        if(firstMatrix!=null)for(int i=0;i<firstMatrix.length;i++){
            Arrays.fill(firstMatrix[i],0);
            Arrays.fill(secondMatrix[i],0);
        }
    }
}
