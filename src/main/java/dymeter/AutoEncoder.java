package dymeter;

import java.util.Random;

/**
 * Symmetric DNN autoencoder: the Static Concept-aware Detector (SCD) of Eq. (3)-(4) and, when its
 * weights are replaced by Theta_d = Theta_s + Delta_Theta_d, the Dynamic Shift-aware Detector of
 * Eq. (12)-(14). Hidden layers use ReLU, the output layer is linear, and every parameter is
 * optimised with Adam. Weights are stored as weight[layer][input][output].
 */
final class AutoEncoder {
    /** Back-propagated gradients of Eq. (4)/(14) for the weights that produced the forward pass. */
    static final class Gradient {
        final double[][][] weights;
        final double[][] biases;
        Gradient(double[][][] weights,double[][] biases){this.weights=weights;this.biases=biases;}
    }

    final int[] widths;
    private final double[][][] weights;
    private final double[][] biases;
    private final Adam[] weightOptimizers;
    private final Adam[] biasOptimizers;
    private final int layers;
    private double[][] activation;
    private double[][] pre;
    private double[][][] usedWeights;
    private double[][] usedBiases;

    AutoEncoder(int[] widths,Random random){
        if(widths.length<3)throw new IllegalArgumentException("The autoencoder needs at least two layers.");
        this.widths=widths.clone();
        this.layers=widths.length-1;
        weights=new double[layers][][];
        biases=new double[layers][];
        weightOptimizers=new Adam[layers];
        biasOptimizers=new Adam[layers];
        for(int l=0;l<layers;l++){
            weights[l]=new double[widths[l]][widths[l+1]];
            biases[l]=new double[widths[l+1]];
            if(l==layers-1)Matrix.xavierInit(weights[l],random); else Matrix.heInit(weights[l],random);
            weightOptimizers[l]=new Adam(widths[l],widths[l+1]);
            biasOptimizers[l]=new Adam(widths[l+1]);
        }
    }

    /**
     * Paper's symmetric design: the encoder has 'encoderLayers' layers ending in the latent width and
     * the decoder mirrors it, so the layer sizes read
     * [d, h1, ..., latent, ..., h1, d]. Hidden widths shrink geometrically from the input width to
     * the latent width and never fall below the latent code.
     */
    static int[] symmetricWidths(int dimensions,int latentWidth,int encoderLayers){
        int n=Math.max(1,Math.min(encoderLayers,10));
        int latent=Math.max(1,Math.min(latentWidth,dimensions));
        int[] encoder=new int[n];
        encoder[n-1]=latent;
        for(int k=1;k<n;k++){
            int width=(int)Math.round(dimensions*Math.pow(latent/(double)dimensions,(double)k/n));
            encoder[k-1]=Math.max(latent,Math.min(dimensions,width));
            if(k>1)encoder[k-1]=Math.min(encoder[k-1],encoder[k-2]);
        }
        int[] sizes=new int[2*n+1];
        sizes[0]=dimensions;
        for(int k=0;k<n;k++)sizes[k+1]=encoder[k];
        for(int j=1;j<=n;j++)sizes[n+j]=j==n?dimensions:encoder[n-1-j];
        return sizes;
    }

    int inputSize(){return widths[0];}
    int latentSize(){return widths[widths.length/2];}
    int parameterCount(){
        int total=0;
        for(int l=0;l<layers;l++)total+=widths[l]*widths[l+1]+widths[l+1];
        return total;
    }

    double[] reconstruct(double[] x){return forward(x,null,null);}

    /** Eq. (3)/(13): x -> E(x) -> z -> D(z) -> y, with an optional dynamic parameter set. */
    double[] forward(double[] x,double[][][] weightOverride,double[][] biasOverride){
        usedWeights=weightOverride==null?weights:weightOverride;
        usedBiases=biasOverride==null?biases:biasOverride;
        activation=new double[layers+1][];
        pre=new double[layers][];
        activation[0]=x.clone();
        for(int l=0;l<layers;l++){
            pre[l]=Matrix.affine(usedWeights[l],usedBiases[l],activation[l]);
            double[] out=new double[widths[l+1]];
            boolean outputLayer=l==layers-1;
            for(int j=0;j<out.length;j++)out[j]=outputLayer?pre[l][j]:Numeric.relu(pre[l][j]);
            activation[l+1]=out;
        }
        return activation[layers];
    }

    /** Latent representation z (respectively z') of the most recent forward pass. */
    double[] latent(){return activation[layers/2].clone();}

    /** Encoder part only, used when reporting the learned latent space. */
    double[] encode(double[] x){
        double[] current=x;
        for(int l=0;l<layers/2;l++){
            double[] out=Matrix.affine(weights[l],biases[l],current);
            for(int i=0;i<out.length;i++)out[i]=Numeric.relu(out[i]);
            current=out;
        }
        return current.clone();
    }

    /** Eq. (4)/(14): L_2(x,y) = sum_i (x_i - y_i)^2 / n with n the input dimensionality. */
    static double reconstructionLoss(double[] x,double[] y){return Matrix.squaredDistance(x,y)/x.length;}

    /** dL_2/dy of the mean squared reconstruction loss used by SCD and DSD. */
    static double[] reconstructionGradient(double[] x,double[] y){
        double[] gradient=new double[x.length];
        for(int i=0;i<x.length;i++)gradient[i]=2*(y[i]-x[i])/x.length;
        return gradient;
    }

    /** Eq. (4)/(14) chain rule: dLoss/dW for the weights that produced the last forward pass. */
    Gradient backward(double[] outputGradient){
        double[][][] weightGradients=new double[layers][][];
        double[][] biasGradients=new double[layers][];
        double[] delta=outputGradient;
        for(int l=layers-1;l>=0;l--){
            double[] dz=new double[widths[l+1]];
            boolean outputLayer=l==layers-1;
            for(int j=0;j<dz.length;j++)dz[j]=outputLayer?delta[j]:delta[j]*Numeric.reluDerivative(pre[l][j]);
            weightGradients[l]=Matrix.zeros(widths[l],widths[l+1]);
            Matrix.outerAdd(weightGradients[l],activation[l],dz,1.0);
            biasGradients[l]=dz.clone();
            if(l>0)delta=Matrix.affineTranspose(usedWeights[l],dz);
        }
        return new Gradient(weightGradients,biasGradients);
    }

    /** Adam update of Theta_s (Eq. 12: Theta_s is trained together with the shift generators). */
    void applyGradient(Gradient gradient,double learningRate){
        for(int l=0;l<layers;l++){
            weightOptimizers[l].step(weights[l],gradient.weights[l],learningRate);
            biasOptimizers[l].step(biases[l],gradient.biases[l],learningRate);
        }
    }

    void resetOptimizers(){for(int l=0;l<layers;l++){weightOptimizers[l].reset();biasOptimizers[l].reset();}}

    /** Used by the offline refresh to move a freshly fine-tuned detector into the live engine. */
    void absorb(AutoEncoder other){
        for(int l=0;l<layers;l++){
            for(int i=0;i<widths[l];i++)System.arraycopy(other.weights[l][i],0,weights[l][i],0,widths[l+1]);
            System.arraycopy(other.biases[l],0,biases[l],0,widths[l+1]);
        }
        resetOptimizers();
    }

    double[][][] weightsRef(){return weights;}
    double[][] biasesRef(){return biases;}
}
