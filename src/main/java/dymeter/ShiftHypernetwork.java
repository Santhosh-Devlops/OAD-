package dymeter;

import java.util.Random;

/**
 * Dynamic Shift-aware Detector (DSD) hypernetwork. Instead of the random vector r^(n) of Eq. (1),
 * Eq. (9) derives the per-layer feature e^(n) = L_layer^(n)(E_share(x)) from the current instance,
 * Eq. (10) maps it to the parameter matrix W^(n) = (W1 e^(n) + b1) W2 + b2 with two MLP layers and
 * Eq. (11) adds the layer bias, K^(n) = W^(n) + b^(n). The shifts are added to the SCD parameters,
 * Theta_d = Theta_s + Delta_Theta_d (Eq. 12). All parameters are flat row-major arrays.
 */
final class ShiftHypernetwork {
    /** dLoss/dTheta_h of the hypernetwork for every tensor, plus the shared encoder parameters. */
    static final class Gradient {
        final double[][] layerW,layerB,mlp1W,mlp1B,mlp2W,mlp2B,base,embedding;
        final double[] sharedW,sharedB;
        Gradient(double[][] layerW,double[][] layerB,double[][] mlp1W,double[][] mlp1B,double[][] mlp2W,
                 double[][] mlp2B,double[][] base,double[][] embedding,double[] sharedW,double[] sharedB){
            this.layerW=layerW;this.layerB=layerB;this.mlp1W=mlp1W;this.mlp1B=mlp1B;this.mlp2W=mlp2W;
            this.mlp2B=mlp2B;this.base=base;this.embedding=embedding;this.sharedW=sharedW;this.sharedB=sharedB;
        }
    }

    private final int inputSize,sharedWidth,mlpWidth,tensors;
    private final int[] inputs,outputs,sizes;
    private final boolean instanceConditioned;
    private final double[] sharedW,sharedB;
    private final double[][] layerW,layerB,mlp1W,mlp1B,mlp2W,mlp2B,base,embedding;
    private final Adam sharedWOptimizer,sharedBOptimizer;
    private final Adam[] layerWOptimizer,layerBOptimizer,mlp1WOptimizer,mlp1BOptimizer,mlp2WOptimizer,mlp2BOptimizer,baseOptimizer,embeddingOptimizer;
    private double[] sharedPre,sharedActivation;
    private double[][] feature,hidden,flatShift;

    ShiftHypernetwork(int[] inputDimensions,int[] outputDimensions,int sharedWidth,int mlpWidth,
                      boolean instanceConditioned,Random random){
        this.tensors=inputDimensions.length;
        this.inputs=inputDimensions.clone();
        this.outputs=outputDimensions.clone();
        this.inputSize=inputDimensions[0];
        this.sharedWidth=Math.max(2,sharedWidth);
        this.mlpWidth=Math.max(2,mlpWidth);
        this.instanceConditioned=instanceConditioned;
        sizes=new int[tensors];
        for(int n=0;n<tensors;n++)sizes[n]=inputs[n]*outputs[n];
        sharedW=new double[inputSize*this.sharedWidth];
        sharedB=new double[this.sharedWidth];
        double scale=Math.sqrt(2.0/inputSize);
        for(int i=0;i<sharedW.length;i++)sharedW[i]=random.nextGaussian()*scale;
        layerW=new double[tensors][];layerB=new double[tensors][];mlp1W=new double[tensors][];mlp1B=new double[tensors][];
        mlp2W=new double[tensors][];mlp2B=new double[tensors][];base=new double[tensors][];embedding=new double[tensors][];
        layerWOptimizer=new Adam[tensors];layerBOptimizer=new Adam[tensors];mlp1WOptimizer=new Adam[tensors];mlp1BOptimizer=new Adam[tensors];
        mlp2WOptimizer=new Adam[tensors];mlp2BOptimizer=new Adam[tensors];baseOptimizer=new Adam[tensors];embeddingOptimizer=new Adam[tensors];
        for(int n=0;n<tensors;n++){
            layerW[n]=new double[this.sharedWidth*this.sharedWidth];
            layerB[n]=new double[this.sharedWidth];
            mlp1W[n]=new double[this.mlpWidth*this.sharedWidth];
            mlp1B[n]=new double[this.mlpWidth];
            mlp2W[n]=new double[this.mlpWidth*sizes[n]];
            mlp2B[n]=new double[sizes[n]];
            base[n]=new double[sizes[n]];
            embedding[n]=new double[this.sharedWidth];
            double layerScale=Math.sqrt(2.0/this.sharedWidth);
            for(int i=0;i<layerW[n].length;i++)layerW[n][i]=random.nextGaussian()*layerScale;
            for(int i=0;i<mlp1W[n].length;i++)mlp1W[n][i]=random.nextGaussian()*Math.sqrt(2.0/this.sharedWidth);
            // Eq. (11): with W2 = 0 and b = 0 the dynamic detector starts exactly at Theta_s.
            for(int i=0;i<embedding[n].length;i++)embedding[n][i]=random.nextGaussian();
            layerWOptimizer[n]=new Adam(layerW[n].length);layerBOptimizer[n]=new Adam(layerB[n].length);
            mlp1WOptimizer[n]=new Adam(mlp1W[n].length);mlp1BOptimizer[n]=new Adam(mlp1B[n].length);
            mlp2WOptimizer[n]=new Adam(mlp2W[n].length);mlp2BOptimizer[n]=new Adam(mlp2B[n].length);
            baseOptimizer[n]=new Adam(base[n].length);embeddingOptimizer[n]=new Adam(embedding[n].length);
        }
        sharedWOptimizer=new Adam(sharedW.length);
        sharedBOptimizer=new Adam(sharedB.length);
    }

    int tensorCount(){return tensors;}
    int mlpWidth(){return mlpWidth;}
    int sharedWidth(){return sharedWidth;}
    boolean instanceConditioned(){return instanceConditioned;}

    int parameterCount(){
        int total=sharedW.length+sharedB.length;
        for(int n=0;n<tensors;n++)total+=layerW[n].length+layerB[n].length+mlp1W[n].length+mlp1B[n].length
                +mlp2W[n].length+mlp2B[n].length+base[n].length+embedding[n].length;
        return total;
    }

    private double shiftMagnitude;
    private double[] input;

    /** Eq. (9)-(11): adds Delta_Theta_d to the provided dynamic parameter set (Eq. 12). */
    void addShift(double[] x,double[][][] target){
        input=x;
        if(instanceConditioned){
            sharedPre=new double[sharedWidth];
            for(int i=0;i<inputSize;i++){
                double xi=x[i];
                if(xi==0)continue;
                int offset=i*sharedWidth;
                for(int s=0;s<sharedWidth;s++)sharedPre[s]+=xi*sharedW[offset+s];
            }
            sharedActivation=new double[sharedWidth];
            for(int s=0;s<sharedWidth;s++)sharedActivation[s]=Numeric.relu(sharedPre[s]+sharedB[s]);
        }
        feature=new double[tensors][];
        hidden=new double[tensors][];
        flatShift=new double[tensors][];
        double magnitude=0;
        for(int n=0;n<tensors;n++){
            double[] e=new double[sharedWidth];
            if(instanceConditioned){
                for(int s=0;s<sharedWidth;s++){
                    double sum=layerB[n][s];
                    int layerOffset=s*sharedWidth;
                    for(int k=0;k<sharedWidth;k++)sum+=layerW[n][layerOffset+k]*sharedActivation[k];
                    e[s]=sum;
                }
            }else{
                System.arraycopy(embedding[n],0,e,0,sharedWidth);
            }
            feature[n]=e;
            double[] u=new double[mlpWidth];
            for(int k=0;k<sharedWidth;k++){
                double ek=e[k];
                if(ek==0)continue;
                int offset=k*mlpWidth;
                for(int p=0;p<mlpWidth;p++)u[p]+=ek*mlp1W[n][offset+p];
            }
            for(int p=0;p<mlpWidth;p++)u[p]+=mlp1B[n][p];
            hidden[n]=u;
            double[] flat=new double[sizes[n]];
            for(int p=0;p<mlpWidth;p++){
                double up=u[p];
                if(up==0)continue;
                int offset=p*sizes[n];
                for(int q=0;q<sizes[n];q++)flat[q]+=up*mlp2W[n][offset+q];
            }
            for(int q=0;q<sizes[n];q++)flat[q]+=mlp2B[n][q]+base[n][q];
            flatShift[n]=flat;
            for(int i=0;i<inputs[n];i++)for(int j=0;j<outputs[n];j++){
                double delta=flat[i*outputs[n]+j];
                target[n][i][j]+=delta;
                magnitude+=delta*delta;
            }
        }
        shiftMagnitude=Math.sqrt(magnitude);
    }

    /** ||Delta_Theta_d|| of the most recent instance-conditioned shift, reported to the UI. */
    double lastShiftMagnitude(){return shiftMagnitude;}

    /** Back-propagation of dLoss/dDelta_Theta_d (from the DSD loss) into Theta_h. */
    Gradient backward(double[][][] shiftGradients){
        double[][] dLayerW=new double[tensors][],dLayerB=new double[tensors][],dMlp1W=new double[tensors][];
        double[][] dMlp1B=new double[tensors][],dMlp2W=new double[tensors][],dMlp2B=new double[tensors][];
        double[][] dBase=new double[tensors][],dEmbedding=new double[tensors][];
        double[] dSharedActivation=new double[sharedWidth];
        for(int n=0;n<tensors;n++){
            double[] dFlat=flatten(shiftGradients[n]);
            dBase[n]=dFlat.clone();
            dMlp2B[n]=dFlat.clone();
            dMlp2W[n]=new double[mlpWidth*sizes[n]];
            for(int p=0;p<mlpWidth;p++){
                double up=hidden[n][p];
                int offset=p*sizes[n];
                for(int q=0;q<sizes[n];q++)dMlp2W[n][offset+q]=up*dFlat[q];
            }
            double[] du=new double[mlpWidth];
            for(int p=0;p<mlpWidth;p++){
                double sum=0;
                int offset=p*sizes[n];
                for(int q=0;q<sizes[n];q++)sum+=mlp2W[n][offset+q]*dFlat[q];
                du[p]=sum;
            }
            dMlp1B[n]=du.clone();
            double[] dFeature=new double[sharedWidth];
            dMlp1W[n]=new double[sharedWidth*mlpWidth];
            for(int k=0;k<sharedWidth;k++){
                double ek=feature[n][k];
                int offset=k*mlpWidth;
                for(int p=0;p<mlpWidth;p++){dMlp1W[n][offset+p]=ek*du[p];dFeature[k]+=mlp1W[n][offset+p]*du[p];}
            }
            if(instanceConditioned){
                dLayerB[n]=dFeature.clone();
                dLayerW[n]=new double[sharedWidth*sharedWidth];
                for(int s=0;s<sharedWidth;s++){
                    double activationValue=sharedActivation[s];
                    int offset=s*sharedWidth;
                    for(int k=0;k<sharedWidth;k++){dLayerW[n][offset+k]=activationValue*dFeature[k];dSharedActivation[k]+=layerW[n][offset+k]*dFeature[k];}
                }
                dEmbedding[n]=new double[sharedWidth];
            }else{
                dLayerB[n]=new double[sharedWidth];
                dLayerW[n]=new double[sharedWidth*sharedWidth];
                dEmbedding[n]=dFeature.clone();
            }
        }
        double[] dSharedPre=new double[sharedWidth];
        if(instanceConditioned)for(int s=0;s<sharedWidth;s++)dSharedPre[s]=dSharedActivation[s]*Numeric.reluDerivative(sharedPre[s]);
        double[] dSharedW=new double[sharedW.length];
        if(instanceConditioned)for(int i=0;i<inputSize;i++){
            double xi=input[i];
            int offset=i*sharedWidth;
            for(int s=0;s<sharedWidth;s++)dSharedW[offset+s]=xi*dSharedPre[s];
        }
        return new Gradient(dLayerW,dLayerB,dMlp1W,dMlp1B,dMlp2W,dMlp2B,dBase,dEmbedding,dSharedW,dSharedPre.clone());
    }

    // Eq. (10)-(11): the two MLP layers turn the per-layer feature vector e^(n) into the complete
    // N_in x N_out shift matrix, to which the layer bias b^(n) is added.
    void applyGradient(Gradient gradient,double learningRate){
        for(int n=0;n<tensors;n++){
            layerWOptimizer[n].step(layerW[n],gradient.layerW[n],learningRate);
            layerBOptimizer[n].step(layerB[n],gradient.layerB[n],learningRate);
            mlp1WOptimizer[n].step(mlp1W[n],gradient.mlp1W[n],learningRate);
            mlp1BOptimizer[n].step(mlp1B[n],gradient.mlp1B[n],learningRate);
            mlp2WOptimizer[n].step(mlp2W[n],gradient.mlp2W[n],learningRate);
            mlp2BOptimizer[n].step(mlp2B[n],gradient.mlp2B[n],learningRate);
            baseOptimizer[n].step(base[n],gradient.base[n],learningRate);
            embeddingOptimizer[n].step(embedding[n],gradient.embedding[n],learningRate);
        }
        sharedWOptimizer.step(sharedW,gradient.sharedW,learningRate);
        sharedBOptimizer.step(sharedB,gradient.sharedB,learningRate);
    }

    void resetOptimizers(){
        for(int n=0;n<tensors;n++){
            layerWOptimizer[n].reset();layerBOptimizer[n].reset();mlp1WOptimizer[n].reset();mlp1BOptimizer[n].reset();
            mlp2WOptimizer[n].reset();mlp2BOptimizer[n].reset();baseOptimizer[n].reset();embeddingOptimizer[n].reset();
        }
        sharedWOptimizer.reset();sharedBOptimizer.reset();
    }

    void absorb(ShiftHypernetwork other){
        for(int n=0;n<tensors;n++){
            System.arraycopy(other.layerW[n],0,layerW[n],0,layerW[n].length);
            System.arraycopy(other.layerB[n],0,layerB[n],0,layerB[n].length);
            System.arraycopy(other.mlp1W[n],0,mlp1W[n],0,mlp1W[n].length);
            System.arraycopy(other.mlp1B[n],0,mlp1B[n],0,mlp1B[n].length);
            System.arraycopy(other.mlp2W[n],0,mlp2W[n],0,mlp2W[n].length);
            System.arraycopy(other.mlp2B[n],0,mlp2B[n],0,mlp2B[n].length);
            System.arraycopy(other.base[n],0,base[n],0,base[n].length);
            System.arraycopy(other.embedding[n],0,embedding[n],0,embedding[n].length);
        }
        System.arraycopy(other.sharedW,0,sharedW,0,sharedW.length);
        System.arraycopy(other.sharedB,0,sharedB,0,sharedB.length);
        resetOptimizers();
    }

    private static double[] flatten(double[][] matrix){
        double[] flat=new double[matrix.length*matrix[0].length];
        int index=0;
        for(double[] row:matrix){System.arraycopy(row,0,flat,index,row.length);index+=row.length;}
        return flat;
    }

    /** Live parameter views in a fixed order, used by the numerical gradient check of the self-test. */
    java.util.List<double[]> parameterViews(){
        java.util.List<double[]> views=new java.util.ArrayList<>();
        views.add(sharedW);
        views.add(sharedB);
        for(int n=0;n<tensors;n++){
            views.add(layerW[n]);
            views.add(layerB[n]);
            views.add(mlp1W[n]);
            views.add(mlp1B[n]);
            views.add(mlp2W[n]);
            views.add(mlp2B[n]);
            views.add(base[n]);
            views.add(embedding[n]);
        }
        return views;
    }

    /** The gradient of one backward pass flattened in the same order as parameterViews(). */
    double[] gradientVector(Gradient gradient){
        java.util.List<double[]> view=new java.util.ArrayList<>();
        view.add(gradient.sharedW);
        view.add(gradient.sharedB);
        for(int n=0;n<tensors;n++){
            view.add(gradient.layerW[n]);
            view.add(gradient.layerB[n]);
            view.add(gradient.mlp1W[n]);
            view.add(gradient.mlp1B[n]);
            view.add(gradient.mlp2W[n]);
            view.add(gradient.mlp2B[n]);
            view.add(gradient.base[n]);
            view.add(gradient.embedding[n]);
        }
        int size=0;
        for(double[] part:view)size+=part.length;
        double[] flat=new double[size];
        int index=0;
        for(double[] part:view){
            System.arraycopy(part,0,flat,index,part.length);
            index+=part.length;
        }
        return flat;
    }

    /** Total parameter count of the flattened view, used to size the numerical checks. */
    int flattenedParameterCount(){
        int size=0;
        for(double[] part:parameterViews())size+=part.length;
        return size;
    }
}
