package dymeter;

import java.util.Random;

/**
 * Intelligent Evolution Controller (IEC): the lightweight evidential classifier of Eq. (2), (6)-(8),
 * implemented as the paper's two-layer ReLU DNN that maps an instance to Dirichlet evidence
 * alpha = g(f(x)) with the exponential non-negativity map g (Eq. 2). Class 1 marks poorly
 * reconstructed (pseudo-positive) instances, class 0 well reconstructed (pseudo-negative) ones.
 * Parameters are kept flat (row-major) so that Adam updates them in place.
 */
final class EvidentialController {
    static final int POSITIVE=1,NEGATIVE=0,CLASSES=2;

    /** dLoss/dTheta_c, with Theta_c = (W1,b1,W2,b2) of the two-layer evidential DNN. */
    static final class Gradient {
        final double[] w1,b1,w2,b2;
        Gradient(double[] w1,double[] b1,double[] w2,double[] b2){this.w1=w1;this.b1=b1;this.w2=w2;this.b2=b2;}
    }

    private final int dimensions,hidden;
    private final boolean exponentialEvidence;
    private final double[] w1,b1,w2,b2;
    private final Adam w1Optimizer,b1Optimizer,w2Optimizer,b2Optimizer;
    private double[] input,hiddenPre,hiddenActivation,logits,concentration,evidenceDerivative;

    EvidentialController(int dimensions,int hiddenWidth,boolean exponentialEvidence,Random random){
        this.dimensions=dimensions;
        this.hidden=Math.max(1,hiddenWidth);
        this.exponentialEvidence=exponentialEvidence;
        w1=new double[dimensions*hidden];
        b1=new double[hidden];
        w2=new double[hidden*CLASSES];
        b2=new double[CLASSES];
        double scale1=Math.sqrt(2.0/dimensions),scale2=Math.sqrt(2.0/hidden);
        for(int i=0;i<w1.length;i++)w1[i]=random.nextGaussian()*scale1;
        for(int i=0;i<w2.length;i++)w2[i]=random.nextGaussian()*scale2;
        w1Optimizer=new Adam(w1.length);
        b1Optimizer=new Adam(hidden);
        w2Optimizer=new Adam(w2.length);
        b2Optimizer=new Adam(CLASSES);
    }

    int hiddenWidth(){return hidden;}

    /**
     * Eq. (2): alpha = g(f(x)) with f the two-layer ReLU DNN and g the non-negativity map of the
     * evidence. Two readings of g are supported and selectable:
     *   SOFTPLUS (default): alpha_c = 1 + log(1 + exp(z_c)), the map used by evidential deep learning;
     *   EXPONENTIAL       : alpha_c = 1 + exp(z_c).
     * The paper's own search range for mu_e ([0.005, 0.4], Fig. 4b) shows that its concept
     * uncertainties live in that band, which only holds for the softplus map - the exponential map
     * drives U_ctr to ~0 as soon as the classifier is confident.
     */
    double[] concentration(double[] x){
        input=x;
        hiddenPre=new double[hidden];
        for(int i=0;i<dimensions;i++){
            double xi=x[i];
            if(xi==0)continue;
            int offset=i*hidden;
            for(int h=0;h<hidden;h++)hiddenPre[h]+=xi*w1[offset+h];
        }
        hiddenActivation=new double[hidden];
        for(int h=0;h<hidden;h++)hiddenActivation[h]=Numeric.relu(hiddenPre[h]+b1[h]);
        logits=new double[CLASSES];
        for(int h=0;h<hidden;h++){
            double ah=hiddenActivation[h];
            int offset=h*CLASSES;
            for(int c=0;c<CLASSES;c++)logits[c]+=ah*w2[offset+c];
        }
        concentration=new double[CLASSES];
        evidenceDerivative=new double[CLASSES];
        for(int c=0;c<CLASSES;c++){
            double z=Math.max(-30,Math.min(30,logits[c]+b2[c]));
            if(exponentialEvidence){
                double e=Math.exp(z);
                concentration[c]=1+e;
                evidenceDerivative[c]=e;
            }else{
                concentration[c]=1+Numeric.softplus(z);
                evidenceDerivative[c]=1/(1+Math.exp(-z));
            }
        }
        return concentration;
    }

    double[] alpha(double[] x){return concentration(x);}

    /** Eq. (6): p̂_c = alpha_c / sum_k alpha_k. */
    static double predictiveProbability(double[] alpha,int c){
        double total=0;for(double value:alpha)total+=value;
        return alpha[c]/total;
    }

    static double evidenceSum(double[] alpha){
        double total=0;for(double value:alpha)total+=value;
        return total;
    }

    /**
     * Eq. (8): concept uncertainty, i.e. the mutual information of the Dirichlet distribution,
     * U_ctr = sum_c p̂_c (Phi(alpha_c + 1) - Phi(sum_k alpha_k + 1)) - sum_c p̂_c log p̂_c.
     */
    static double uncertainty(double[] alpha){
        double total=evidenceSum(alpha),digammaSum=Numeric.digamma(total+1),value=0;
        for(int c=0;c<alpha.length;c++){
            double p=alpha[c]/total;
            value+=p*(Numeric.digamma(alpha[c]+1)-digammaSum);
            if(p>0)value-=p*Math.log(p);
        }
        return Math.max(0,value);
    }

    /** Eq. (7): focal evidential loss (1 - p̂_c)^gamma (log(sum_k alpha_k) - log alpha_c). */
    static double focalLoss(double[] alpha,int label,double gamma){
        double total=evidenceSum(alpha),p=alpha[label]/total;
        return Math.pow(1-p,gamma)*(Math.log(total)-Math.log(alpha[label]));
    }

    /** Cross-entropy on the Dirichlet predictive probabilities, the paper's DyMETER-ce ablation. */
    static double crossEntropyLoss(double[] alpha,int label){
        return -Math.log(Math.max(1e-12,predictiveProbability(alpha,label)));
    }

    /**
     * Back-propagation of the IEC objective (Eq. 7, or the cross-entropy of DyMETER-ce) through the
     * exponential evidence map and the two-layer DNN. Returns dLoss/dTheta_c.
     */
    Gradient backward(int label,boolean focal,double gamma){
        double total=evidenceSum(concentration),p=concentration[label]/total;
        double logTerm=Math.log(total)-Math.log(concentration[label]);
        double[] dAlpha=new double[CLASSES];
        for(int c=0;c<CLASSES;c++){
            if(focal){
                double q=Math.max(1e-12,1-p);
                double dq=(c==label)?-(total-concentration[label])/(total*total):concentration[label]/(total*total);
                double dLog=(c==label)?(1/total-1/concentration[label]):(1/total);
                dAlpha[c]=gamma*Math.pow(q,gamma-1)*dq*logTerm+Math.pow(q,gamma)*dLog;
            }else{
                dAlpha[c]=(c==label)?(1/total-1/concentration[label]):(1/total);
            }
        }
        double[] dLogit=new double[CLASSES];
        for(int c=0;c<CLASSES;c++)dLogit[c]=dAlpha[c]*evidenceDerivative[c];
        double[] dW2=new double[hidden*CLASSES],dB2=dLogit.clone();
        for(int h=0;h<hidden;h++){
            double ah=hiddenActivation[h];
            int offset=h*CLASSES;
            for(int c=0;c<CLASSES;c++)dW2[offset+c]=ah*dLogit[c];
        }
        double[] dHidden=new double[hidden];
        for(int h=0;h<hidden;h++){
            double sum=0;
            int offset=h*CLASSES;
            for(int c=0;c<CLASSES;c++)sum+=w2[offset+c]*dLogit[c];
            dHidden[h]=sum;
        }
        double[] dHiddenPre=new double[hidden];
        for(int h=0;h<hidden;h++)dHiddenPre[h]=dHidden[h]*Numeric.reluDerivative(hiddenPre[h]+b1[h]);
        double[] dW1=new double[dimensions*hidden];
        for(int i=0;i<dimensions;i++){
            double xi=input[i];
            if(xi==0)continue;
            int offset=i*hidden;
            for(int h=0;h<hidden;h++)dW1[offset+h]=xi*dHiddenPre[h];
        }
        return new Gradient(dW1,dHiddenPre.clone(),dW2,dB2);
    }

    void applyGradient(Gradient gradient,double learningRate){
        w1Optimizer.step(w1,gradient.w1,learningRate);
        b1Optimizer.step(b1,gradient.b1,learningRate);
        w2Optimizer.step(w2,gradient.w2,learningRate);
        b2Optimizer.step(b2,gradient.b2,learningRate);
    }

    void resetOptimizers(){w1Optimizer.reset();b1Optimizer.reset();w2Optimizer.reset();b2Optimizer.reset();}

    /** Live parameter views, used by the numerical gradient checks of the self-test. */
    double[] w1Ref(){return w1;}
    double[] b1Ref(){return b1;}
    double[] w2Ref(){return w2;}
    double[] b2Ref(){return b2;}

    void absorb(EvidentialController other){
        System.arraycopy(other.w1,0,w1,0,w1.length);
        System.arraycopy(other.b1,0,b1,0,b1.length);
        System.arraycopy(other.w2,0,w2,0,w2.length);
        System.arraycopy(other.b2,0,b2,0,b2.length);
        resetOptimizers();
    }
}
