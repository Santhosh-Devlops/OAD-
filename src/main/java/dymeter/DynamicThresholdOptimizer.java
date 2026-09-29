package dymeter;

import java.util.ArrayDeque;
import java.util.Collection;
import java.util.Deque;

/**
 * Dynamic Threshold Optimization (DTO). Implements the uncertainty-aware anomaly score of Eq. (15),
 * the empirical normal-score distribution of Eq. (16)-(17), the candidate-window regularization of
 * Eq. (18), the adaptive threshold of Eq. (19) and the accumulated-uncertainty offline update signal
 * of Eq. (20). When the IEC signals concept drift the procedure is re-initialised: stale entries of
 * W_N and W_C are discarded and the quantile statistics are re-estimated from the new concept.
 */
final class DynamicThresholdOptimizer {
    /** Everything the caller needs to explain a single decision. */
    static final class Decision {
        final double score,preliminary,regularization,threshold,medianCandidate,deviation;
        final int normalWindow,candidateWindow,driftResets;
        final boolean anomaly,drift,reset,offlineUpdate;
        Decision(double score,double preliminary,double regularization,double threshold,double medianCandidate,
                 double deviation,int normalWindow,int candidateWindow,int driftResets,boolean anomaly,
                 boolean drift,boolean reset,boolean offlineUpdate){
            this.score=score;this.preliminary=preliminary;this.regularization=regularization;this.threshold=threshold;
            this.medianCandidate=medianCandidate;this.deviation=deviation;this.normalWindow=normalWindow;
            this.candidateWindow=candidateWindow;this.driftResets=driftResets;this.anomaly=anomaly;this.drift=drift;
            this.reset=reset;this.offlineUpdate=offlineUpdate;
        }
    }

    private static final int MINIMUM_CALIBRATION=8;

    private final int windowSize,maxInterval;
    private final double tau,kappa,lambda,muO,emaRate,muT;
    private final boolean quantileOnly,meanCandidate;
    private final Deque<Double> normalScores=new ArrayDeque<>(),candidateScores=new ArrayDeque<>(),uncertainty=new ArrayDeque<>();
    private double referenceError,medianCandidate=Double.NaN,deviation=Double.NaN;
    private double preliminary=Double.NaN,regularization,threshold=Double.NaN,accumulated;
    private int rowsSinceUpdate,driftResets,reseedRemaining,staleRows;

    DynamicThresholdOptimizer(int windowSize,double tau,double kappa,double lambda,double muO,int maxInterval,
                              double emaRate,double muT,double referenceError,boolean quantileOnly,boolean meanCandidate){
        this.windowSize=windowSize;
        this.tau=tau;this.kappa=kappa;this.lambda=lambda;this.muO=muO;this.maxInterval=Math.max(1,maxInterval);
        this.emaRate=emaRate;this.muT=muT;this.referenceError=referenceError;
        this.quantileOnly=quantileOnly;this.meanCandidate=meanCandidate;
    }

    /** Eq. (15): A(x) = R_e exp(lambda * U_ctr * (r_e - R_e)). */
    double score(double error,double uncertaintyValue){return error*Math.exp(lambda*uncertaintyValue*(referenceError-error));}

    /** Warm start from the confident normal pseudo-labels of the historical segment. */
    void seed(Collection<Double> scores){
        for(double value:scores){normalScores.addLast(value);trim(normalScores);}
        deviation=Numeric.mad(normalScores);
        preliminary=Numeric.quantile(normalScores,1-tau);
        threshold=preliminary;
    }

    void setReferenceError(double value){referenceError=value;}

    double referenceError(){return referenceError;}
    double preliminary(){return preliminary;}
    double threshold(){return threshold;}
    double candidateMedian(){return medianCandidate;}
    double deviation(){return deviation;}
    int normalWindowSize(){return normalScores.size();}
    int candidateWindowSize(){return candidateScores.size();}
    int driftResets(){return driftResets;}
    double accumulatedUncertainty(){return accumulated;}

    /**
     * Inference step of Algorithm 2: score, calibrate mu*_a (Eq. 19), decide, then update windows.
     *
     * Eq. (16) cost-sensitive thresholding: the expected cost
     *   F_c(mu_a) = pi_N Cfp INT_{mu_a..inf} p_N(A) dA + pi_A Cfn INT_{0..mu_a} p_A(A) dA
     * is minimised on the normal distribution only (the paper's simplification for pi_N >> pi_A), so the
     * decision boundary is derived from the empirical normal-score CDF F_N of Eq. (17) below.
     */
    Decision process(double error,double uncertaintyValue,boolean startNewDriftEpisode){
        boolean reset=false;
        if(startNewDriftEpisode){
            resetStatistics();
            reset=true;
            driftResets++;
        }
        double score=score(error,uncertaintyValue);
        // r_e is initialised with the maximum training error and updated by an exponential moving average.
        referenceError=(1-emaRate)*referenceError+emaRate*error;
        // Stale-window guard: if the boundary has rejected every instance for a full window it cannot be
        // tracking the stream, so F_N(A) is rebuilt from the current observations before deciding. This
        // is the general counterpart of the drift re-initialisation and prevents the unrecoverable state
        // in which a stale threshold flags everything and W_N never refills (measured: FPR = 1.0).
        if(staleRows>windowSize){
            resetStatistics();
            reseedRemaining=Math.max(reseedRemaining,windowSize/4);
            staleRows=0;
        }
        double mu0=currentPreliminary(),delta=currentDeviation();
        // W_C of Eq. (18): uncertain scores close to mu0, i.e. U_ctr > mu_t and A in [mu0-delta, mu0+delta].
        if(uncertaintyValue>muT&&Math.abs(score-mu0)<=delta){
            candidateScores.addLast(score);
            trim(candidateScores);
        }
        double rawCandidate=meanCandidate?Numeric.mean(candidateScores):Numeric.median(candidateScores);
        if(!Double.isNaN(rawCandidate))medianCandidate=Double.isNaN(medianCandidate)?rawCandidate
                :(1-emaRate)*medianCandidate+emaRate*rawCandidate;
        double muR=candidateScores.isEmpty()||quantileOnly?0:kappa*(mu0-medianCandidate);
        double muStar=mu0+muR;
        boolean anomaly=score>muStar;
        // W_N holds the recently classified normal scores. Right after a drift reset the window is
        // empty, so the first 'reseedRemaining' scores of the new concept rebuild the empirical
        // distribution F_N(A) of Eq. (17); without this bootstrap the stale threshold could flag the
        // whole new concept and the window would never fill again.
        boolean reseeding=reseedRemaining>0;
        if(reseeding)reseedRemaining--;
        if(reseeding||!anomaly){normalScores.addLast(score);trim(normalScores);staleRows=0;}
        else staleRows++;
        preliminary=mu0;regularization=muR;threshold=muStar;
        // Eq. (20): sum of U_ctr above mu_e inside the sliding window, or the maximum interval T_max.
        uncertainty.addLast(uncertaintyValue>muT?uncertaintyValue:0);
        trim(uncertainty);
        accumulated=0;
        for(double value:uncertainty)accumulated+=value;
        rowsSinceUpdate++;
        // Eq. (20): UP = 1 when the accumulated concept uncertainty above mu_e in W_N exceeds mu_o or when
        // the time since the last update exceeds T_max; the refresh then runs in a parallel worker.
        boolean offlineUpdate=accumulated>muO||rowsSinceUpdate>maxInterval;
        if(offlineUpdate)rowsSinceUpdate=0;
        return new Decision(score,mu0,muR,muStar,medianCandidate,delta,normalScores.size(),candidateScores.size(),
                driftResets,anomaly,startNewDriftEpisode,reset,offlineUpdate);
    }

    /**
     * Algorithm 2 drift branch: discard the stale entries of W_N and W_C and re-estimate the
     * statistics as the new concept arrives. The engine calls this once per confirmed drift episode,
     * so a single drifting noise spike cannot wipe the calibration window. The cold start after the
     * reset is undefined in the paper; we keep serving the last calibrated preliminary threshold
     * until the fresh window holds MINIMUM_CALIBRATION normal scores (implementation bootstrap).
     */
    void resetStatistics(){
        normalScores.clear();
        candidateScores.clear();
        uncertainty.clear();
        accumulated=0;
        medianCandidate=Double.NaN;
        deviation=Double.NaN;
        rowsSinceUpdate=0;
        reseedRemaining=Math.max(MINIMUM_CALIBRATION,windowSize/4);
        staleRows=0;
    }

    /** Eq. (17): empirical CDF F_N(A) = |W_N|^-1 Sum 1(A_i <= A) over the recent normal scores. */
    private double currentPreliminary(){
        if(normalScores.isEmpty())return Double.isFinite(preliminary)?preliminary:referenceError;
        return Numeric.quantile(normalScores,1-tau);
    }

    /** delta = MAD(W_N): the paper's exponentially decaying rank statistic (Eq. 18). */
    private double currentDeviation(){
        double value=Numeric.mad(normalScores);
        return Double.isFinite(value)?value:(Double.isFinite(deviation)?deviation:referenceError);
    }

    private <T> void trim(Deque<T> deque){while(deque.size()>windowSize)deque.removeFirst();}
}
