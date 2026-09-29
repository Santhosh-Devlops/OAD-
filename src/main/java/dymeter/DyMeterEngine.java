package dymeter;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * DyMETER: dynamic concept adaptation for online anomaly detection. The engine owns the four modules
 * of the paper - SCD (Eq. 3-4), IEC (Eq. 2, 5-8), DSD (Eq. 9-14) and DTO (Eq. 15-20) - and runs
 * Algorithm 1 for training and Algorithm 2 for inference, including the parallel offline update.
 */
public final class DyMeterEngine {
    /**
     * Run settings. Every value fixed by the paper is the default here; the remaining switches exist
     * so that the ablation variants of the paper can be reproduced from the same engine.
     */
    public static final class Config {
        public int dimensions,historicalRows,encoderLayers,latentWidth,epochs=2000,batchSize=64,windowSize=64;
        public int iecHidden,hyperSharedWidth=8,hyperMlpWidth=8,pseudoLabelRounds=4,maxInterval=256;
        public double lambda=0.6,gamma=2.0,kappa=0.8,tau=0.05,muO=32,pseudoProportion=0.05;
        public double learningRate=1e-2,learningRateDecay=0.96,emaRate=2.0/65;
        /** Concept uncertainty threshold mu_e: the paper searches it in [0.005, 0.4]. When it is left
         *  unset (NaN) the port derives it as the muEQuantile-quantile of the training uncertainties. */
        public double muE=Double.NaN;
        /** Quantile of the training uncertainty distribution used when mu_e is not given explicitly.
         *  0.75 marks the upper edge of the historical uncertainty band, so the historical concept itself
         *  is not declared a new one while a genuinely unfamiliar region still crosses the threshold. */
        public double muEQuantile=0.75;
        public long seed=42;
        public boolean useIec=true,useDsd=true,useDto=true,focalLoss=true,instanceConditioned=true;
        public boolean quantileOnly=false,meanCandidate=false,priorLabels=false,dynamicAlways=false,trainStaticBackbone=true;
        /** Eq. (2) evidence map: softplus (evidential deep learning default) or the exponential reading. */
        public boolean exponentialEvidence=false;
        public String variant="DyMETER";

        public Config copy(){
            Config c=new Config();
            c.dimensions=dimensions;c.historicalRows=historicalRows;c.encoderLayers=encoderLayers;c.latentWidth=latentWidth;
            c.epochs=epochs;c.batchSize=batchSize;c.windowSize=windowSize;c.iecHidden=iecHidden;
            c.hyperSharedWidth=hyperSharedWidth;c.hyperMlpWidth=hyperMlpWidth;c.pseudoLabelRounds=pseudoLabelRounds;
            c.maxInterval=maxInterval;c.lambda=lambda;c.gamma=gamma;c.kappa=kappa;c.tau=tau;c.muO=muO;
            c.pseudoProportion=pseudoProportion;c.learningRate=learningRate;c.learningRateDecay=learningRateDecay;
            c.emaRate=emaRate;c.muE=muE;c.muEQuantile=muEQuantile;c.seed=seed;c.useIec=useIec;c.useDsd=useDsd;c.useDto=useDto;
            c.focalLoss=focalLoss;c.instanceConditioned=instanceConditioned;c.quantileOnly=quantileOnly;
            c.meanCandidate=meanCandidate;c.priorLabels=priorLabels;c.dynamicAlways=dynamicAlways;
            c.trainStaticBackbone=trainStaticBackbone;c.variant=variant;c.exponentialEvidence=exponentialEvidence;
            return c;
        }
    }

    /** One processed stream record together with every intermediate quantity of Algorithm 2. */
    public record Result(int row,double[] features,int concept,double uncertainty,double error,double score,double threshold,
                         double preliminary,double regularization,double candidateMedian,double deviation,double referenceError,
                         double shiftMagnitude,int normalWindow,int candidateWindow,boolean anomaly,boolean drift,
                         boolean thresholdReset,boolean offlineUpdate,String detector){}

    private static final double HISTORICAL_RATIO=0.20,MIN_MU_E=0.005,MAX_MU_E=0.4;
    private static final double PSEUDO_LABEL_TOLERANCE=1e-3;
    private static final int DRIFT_CONFIRMATION=5;
    private static final double DRIFT_SMOOTHING=0.2,DRIFT_MARGIN=0.25;

    private final Config config;
    private final Random random;
    private AutoEncoder detector;
    private EvidentialController controller;
    private ShiftHypernetwork hypernetwork;
    private DynamicThresholdOptimizer threshold;
    private double[][][] dynamicWeights;
    private double muE=Double.NaN,muP=Double.NaN,muT=Double.NaN,referenceError=Double.NaN,fixedThreshold=Double.NaN;
    private double smoothedUncertainty=Double.NaN;
    private List<double[]> baseHistory;
    private final Deque<double[]> acceptedNormals=new ArrayDeque<>();
    private int conceptNumber=1;
    private int driftRun,normalRun,rowsSinceEpisode,rowsProcessed;
    private boolean driftEpisodeActive,initialized;
    private int[] tensorInputs,tensorOutputs;
    private long lastUncertainty;
    private volatile boolean refreshing;
    private ExecutorService offlineExecutor;
    private List<Double> trainingNormalScores=new ArrayList<Double>();

    public DyMeterEngine(Config settings){
        this.config=settings.copy();
        this.random=new Random(config.seed);
    }

    /** Settings derived from the data: latent width from PCA, depths, widths and window budgets. */
    public static Config configure(List<double[]> rows){
        if(rows.size()<10)throw new IllegalArgumentException("At least ten numeric rows are needed for the historical segment and the stream.");
        int dimensions=rows.get(0).length;
        int history=Math.max(2,(int)Math.ceil(rows.size()*HISTORICAL_RATIO));
        if(history>=rows.size())history=rows.size()-1;
        Pca.Result pca=Pca.analyse(rows.subList(0,history),0.70);
        Config config=new Config();
        config.dimensions=dimensions;
        config.historicalRows=history;
        config.latentWidth=Math.min(pca.components,dimensions);
        config.encoderLayers=dimensions<=8?2:3;
        config.iecHidden=Math.max(2,Math.min(16,(dimensions+1)/2));
        config.hyperSharedWidth=Math.max(2,Math.min(8,dimensions));
        config.hyperMlpWidth=Math.max(2,Math.min(8,dimensions));
        int streamRows=rows.size()-history;
        config.maxInterval=Math.max(32,(streamRows+1)/2);
        config.muO=config.windowSize*0.5;
        config.emaRate=2.0/(config.windowSize+1);
        return config;
    }

    public Config config(){return config.copy();}
    public boolean trained(){return initialized;}
    public int conceptCount(){return conceptNumber;}
    public double muE(){return muE;}
    public double muP(){return muP;}
    public double muT(){return muT;}
    public double referenceError(){return referenceError;}
    public int latentWidth(){return detector==null?config.latentWidth:detector.latentSize();}
    public int[] networkWidths(){return detector==null?new int[0]:detector.widths.clone();}
    public int detectorParameterCount(){return detector==null?0:detector.parameterCount();}
    public int hypernetworkParameterCount(){return hypernetwork==null?0:hypernetwork.parameterCount();}
    public int controllerParameterCount(){return (config.dimensions+1)*config.iecHidden+2*config.iecHidden+2;}
    public long uncertaintyEvaluations(){return lastUncertainty;}
    public double fixedThreshold(){return fixedThreshold;}
    public boolean refreshing(){return refreshing;}

    /** SCD, IEC, DSD and DTO instantiated from the resolved settings (Algorithm 1, initialization). */
    private void buildModules(boolean trainStaticBackbone){
        detector=new AutoEncoder(AutoEncoder.symmetricWidths(config.dimensions,config.latentWidth,config.encoderLayers),random);
        controller=new EvidentialController(config.dimensions,config.iecHidden,config.exponentialEvidence,random);
        tensorInputs=new int[detector.widths.length-1];
        tensorOutputs=new int[detector.widths.length-1];
        for(int l=0;l<tensorInputs.length;l++){
            tensorInputs[l]=detector.widths[l];
            tensorOutputs[l]=detector.widths[l+1];
        }
        hypernetwork=new ShiftHypernetwork(tensorInputs,tensorOutputs,config.hyperSharedWidth,config.hyperMlpWidth,
                config.instanceConditioned,random);
        dynamicWeights=new double[tensorInputs.length][][];
        for(int l=0;l<tensorInputs.length;l++)dynamicWeights[l]=new double[tensorInputs[l]][tensorOutputs[l]];
        threshold=new DynamicThresholdOptimizer(config.windowSize,config.tau,config.kappa,config.lambda,config.muO,
                config.maxInterval,config.emaRate,Double.NaN,1,config.quantileOnly,config.meanCandidate);
    }

    /** Eq. (10)-(12): Theta_d = Theta_s + Delta_Theta_d into a reused buffer. */
    private void loadDynamicWeights(double[] x){
        double[][][] source=detector.weightsRef();
        for(int l=0;l<source.length;l++)
            for(int i=0;i<source[l].length;i++)System.arraycopy(source[l][i],0,dynamicWeights[l][i],0,source[l][i].length);
        hypernetwork.addShift(x,dynamicWeights);
    }

    private double learningRate(int step){
        return config.learningRate*Math.pow(config.learningRateDecay,(double)step/config.epochs);
    }

    private List<double[]> sampleBatch(List<double[]> rows){
        int size=rows.size();
        if(size<=config.batchSize)return rows;
        int[] order=new int[size];
        for(int i=0;i<size;i++)order[i]=i;
        for(int i=size-1;i>0;i--){
            int j=random.nextInt(i+1),swap=order[i];
            order[i]=order[j];order[j]=swap;
        }
        List<double[]> batch=new ArrayList<>(config.batchSize);
        for(int i=0;i<config.batchSize;i++)batch.add(rows.get(order[i]));
        return batch;
    }

    private static final int UNKNOWN=-1;

    /** Algorithm 1: train the SCD, pseudo-label with Eq. (5), then optimise IEC and DSD jointly. */
    public void train(List<double[]> historical){train(historical,null);}

    /**
     * @param labels optional ground-truth anomaly flags used only by the paper's DyMETER-pl ablation
     *               (1% of the labelled anomalies are injected into the pseudo-labelled set).
     */
    public void train(List<double[]> historical,boolean[] labels){
        if(historical.isEmpty())throw new IllegalArgumentException("The historical training portion is empty.");
        if(baseHistory==null)baseHistory=copyRows(historical);
        if(!initialized){buildModules(config.trainStaticBackbone);initialized=true;}
        long started=System.nanoTime();
        // Algorithm 1, lines 3-4: update Theta_s by minimising L_SCD of Eq. (4).
        if(config.trainStaticBackbone)trainStaticDetector(historical);
        // Reconstruction errors of the static detector for the pseudo-labelling rule of Eq. (5).
        double[] errors=new double[historical.size()];
        double maximum=Double.NEGATIVE_INFINITY;
        for(int i=0;i<historical.size();i++){
            double[] x=historical.get(i);
            errors[i]=AutoEncoder.reconstructionLoss(x,detector.reconstruct(x));
            maximum=Math.max(maximum,errors[i]);
        }
        muP=Numeric.quantile(toList(errors),1-config.pseudoProportion);
        referenceError=maximum;
        // Algorithm 1, lines 6-12: pseudo-labelling, IEC objective of Eq. (7), joint DSD objective of Eq. (14).
        int[] pseudoLabels=config.useIec?trainController(historical,errors,labels):new int[historical.size()];
        trainDynamicDetector(historical,errors,pseudoLabels);
        // mu_t of Eq. (18): the maximum concept uncertainty observed across the training set after training.
        double observed=0;
        for(double[] x:historical)observed=Math.max(observed,uncertainty(x));
        muT=observed;
        seedThresholdStatistics(historical,errors);
        trainingNormalScores=collectNormalScores(historical,errors);
        fixedThreshold=Numeric.quantile(trainingNormalScores,1-config.tau);
        trainingMillis+=(System.nanoTime()-started)/1000000;
    }

    private long trainingMillis;

    /** Lines 3-4 of Algorithm 1: Theta_s <- argmin L_SCD with Adam and the paper's decayed schedule. */
    private void trainStaticDetector(List<double[]> rows){
        RunProgress.stage(1,"SCD training","symmetric autoencoder, Adam, Eq. (3)-(4)",config.epochs);
        for(int step=0;step<config.epochs;step++){
            RunProgress.epoch(step+1);
            double learningRate=learningRate(step);
            for(double[] x:sampleBatch(rows)){
                double[] y=detector.forward(x,null,null);
                detector.applyGradient(detector.backward(AutoEncoder.reconstructionGradient(x,y)),learningRate);
            }
        }
    }

    /** Lines 6-11 of Algorithm 1: repeat the pseudo-labelling of Eq. (5) and the focal evidential
     * objective of Eq. (7) until the uncertainty threshold mu_e converges. Samples with
     * U_ctr > mu_e stay unlabelled ("Unknown" in Eq. 5) and never enter the IEC training set.
     */
    private int[] trainController(List<double[]> rows,double[] errors,boolean[] labels){
        int[] pseudoLabels=new int[rows.size()];
        for(int round=0;round<config.pseudoLabelRounds;round++){
            List<Integer> labelled=new ArrayList<>();
            for(int i=0;i<rows.size();i++){
                double[] x=rows.get(i);
                double u=uncertainty(x);
                // Eq. (5) pseudo-labelling: L_2 above mu_p is pseudo-positive, at or below is pseudo-negative, and
        // instances with U_ctr > mu_e stay unknown so that only high-confidence samples train the IEC.
        if(u>muE){pseudoLabels[i]=UNKNOWN;continue;}
                pseudoLabels[i]=errors[i]>muP?EvidentialController.POSITIVE:EvidentialController.NEGATIVE;
                labelled.add(i);
            }
            if(config.priorLabels&&labels!=null)injectPriorLabels(rows,labels,pseudoLabels,labelled);
            if(labelled.isEmpty())break;
            RunProgress.stage(2,"Pseudo-labelling + IEC","evidential focal loss, Eq. (5)-(8), round "+(round+1),
                    config.epochs);
            for(int step=0;step<config.epochs;step++){
                RunProgress.epoch(step+1);
                double learningRate=learningRate(step);
                for(int pick=0;pick<config.batchSize;pick++){
                    int index=labelled.get(random.nextInt(labelled.size()));
                    double[] x=rows.get(index);
                    controller.concentration(x);
                    controller.applyGradient(controller.backward(pseudoLabels[index],config.focalLoss,config.gamma),learningRate);
                }
            }
            List<Double> uncertainties=new ArrayList<>(labelled.size());
            for(int index:labelled)uncertainties.add(uncertainty(rows.get(index)));
            // mu_e is the searched hyperparameter of the paper ([0.005, 0.4]); when it is not given the
            // port derives it as the muEQuantile-quantile of the training uncertainty distribution, i.e.
            // the boundary between "confident" and "unreliable" instances of Eq. (5).
            double observed=Numeric.quantile(uncertainties,config.muEQuantile);
            double target=config.muE>0?config.muE:observed;
            double next=Math.min(MAX_MU_E,Math.max(MIN_MU_E,target));
            boolean converged=Math.abs(next-muE)<PSEUDO_LABEL_TOLERANCE;
            muE=next;
            if(converged)break;
        }
        return pseudoLabels;
    }

    /**
     * Line 12 of Algorithm 1: update Theta_s, Theta_c and Theta_d by minimising L_DSD (Eq. 14) with
     * the gradients jointly propagated through the SCD and the IEC.
     */
    private void trainDynamicDetector(List<double[]> rows,double[] errors,int[] pseudoLabels){
        if(!config.useDsd)return;
        RunProgress.stage(3,"DSD hypernetwork","instance-aware shifts, Eq. (9)-(14)",config.epochs);
        for(int step=0;step<config.epochs;step++){
            RunProgress.epoch(step+1);
            double learningRate=learningRate(step);
            for(int pick=0;pick<config.batchSize;pick++){
                int index=random.nextInt(rows.size());
                double[] x=rows.get(index);
                loadDynamicWeights(x);
                double[] y=detector.forward(x,dynamicWeights,null);
                AutoEncoder.Gradient gradient=detector.backward(AutoEncoder.reconstructionGradient(x,y));
                detector.applyGradient(gradient,learningRate);
                hypernetwork.applyGradient(hypernetwork.backward(gradient.weights),learningRate);
                if(config.useIec&&pseudoLabels.length>index&&pseudoLabels[index]!=UNKNOWN){
                    controller.concentration(x);
                    controller.applyGradient(controller.backward(pseudoLabels[index],config.focalLoss,config.gamma),learningRate);
                }
            }
        }
    }

    /** DyMETER-pl: 1% of the labelled anomalies join the pseudo-labelled training set. */
    private void injectPriorLabels(List<double[]> rows,boolean[] labels,int[] pseudoLabels,List<Integer> labelled){
        List<Integer> anomalies=new ArrayList<>();
        for(int i=0;i<rows.size()&&i<labels.length;i++)
            if(labels[i]&&pseudoLabels[i]!=EvidentialController.POSITIVE)anomalies.add(i);
        int count=Math.max(1,(int)Math.round(rows.size()*0.01));
        for(int k=0;k<count&&!anomalies.isEmpty();k++){
            int index=anomalies.remove(random.nextInt(anomalies.size()));
            pseudoLabels[index]=EvidentialController.POSITIVE;
            if(!labelled.contains(index))labelled.add(index);
        }
    }

    /** Eq. (8) for one instance; identically zero when the IEC module is switched off. */
    private double uncertainty(double[] x){
        if(!config.useIec)return 0;
        lastUncertainty++;
        return EvidentialController.uncertainty(controller.alpha(x));
    }

    /** U_ctr of the flat Dirichlet prior alpha=(2,2): the uncertainty of "no evidence at all". */
    public static double uninformativeUncertainty(){return EvidentialController.uncertainty(new double[]{2,2});}

    /** DTO warm start plus the fixed threshold used by the variants that omit the DTO. */
    private void seedThresholdStatistics(List<double[]> rows,double[] errors){
        List<Double> normalScores=collectNormalScores(rows,errors);
        if(normalScores.isEmpty()){
            // mu_e was set tighter than the training uncertainty band: fall back to the most confident half
            // of the historical segment so the DTO always has a calibration window.
            List<Double> uncertainties=new ArrayList<>(rows.size());
            for(double[] row:rows)uncertainties.add(uncertainty(row));
            double relaxed=Numeric.quantile(uncertainties,0.5);
            for(int i=0;i<rows.size();i++){
                double u=uncertainty(rows.get(i));
                if(u<=relaxed&&errors[i]<=muP)normalScores.add(score(errors[i],u));
            }
        }
        if(normalScores.isEmpty())
            throw new IllegalArgumentException("No high-confidence normal training rows were found, so the DTO cannot be calibrated.");
        threshold=new DynamicThresholdOptimizer(config.windowSize,config.tau,config.kappa,config.lambda,config.muO,
                config.maxInterval,config.emaRate,muT,referenceError,config.quantileOnly,config.meanCandidate);
        threshold.setReferenceError(referenceError);
        threshold.seed(normalScores);
    }

    /** Confident normal pseudo-labels of Eq. (5): U_ctr <= mu_e and R_e <= mu_p. */
    private List<Double> collectNormalScores(List<double[]> rows,double[] errors){
        List<Double> scores=new ArrayList<>();
        for(int i=0;i<rows.size();i++){
            double u=uncertainty(rows.get(i));
            boolean confident=!config.useIec||u<=muE;
            if(confident&&errors[i]<=muP)scores.add(score(errors[i],u));
        }
        return scores;
    }

    /**
     * Algorithm 2: concept uncertainty from the IEC (line 2), escalation to the DSD when
     * U_ctr > mu_e (lines 3-8), the uncertainty-aware anomaly score of Eq. (15), the adaptive
     * threshold of Eq. (19) and the accumulated-uncertainty update condition of Eq. (20).
     *
     * Eq. (15): A(x) = R_e exp(lambda U_ctr (r_e - R_e)), computed by DynamicThresholdOptimizer.score,
     * where R_e is the SCD error L_2(x,y) of Eq. (4) or the DSD error L_2(x,y') of Eq. (14).
     */
    public Result process(int rowNumber,double[] x){
        AutoEncoder scd=detector;
        DynamicThresholdOptimizer dto=threshold;
        double u=uncertainty(x);
        // The drift signal is the (smoothed) concept uncertainty compared with mu_e, so a single instance
        // flickering around mu_e cannot open or close a concept; a 25 % margin keeps the historical band
        // itself from being declared a new concept (Algorithm 2, line 3).
        smoothedUncertainty=Double.isNaN(smoothedUncertainty)?u:(1-DRIFT_SMOOTHING)*smoothedUncertainty+DRIFT_SMOOTHING*u;
        DriftState state=evaluateDrift(config.useIec&&smoothedUncertainty>muE*(1+DRIFT_MARGIN));
        boolean drift=state.drifting(),newDriftEpisode=state.rising();
        boolean dynamic=config.useDsd&&(drift||config.dynamicAlways);
        double[] y;
        String detectorName="SCD";
        if(dynamic){
            loadDynamicWeights(x);
            y=scd.forward(x,dynamicWeights,null);
            detectorName="DSD";
        }else{
            y=scd.forward(x,null,null);
        }
        double error=AutoEncoder.reconstructionLoss(x,y);
        double shiftMagnitude=dynamic?hypernetwork.lastShiftMagnitude():0;
        double scoreValue,mu0,muR,muStar,candidateMedian,deviationValue;boolean anomaly,reset=false,offlineUpdate=false;
        if(config.useDto){
            DynamicThresholdOptimizer.Decision decision=dto.process(error,u,newDriftEpisode);
            scoreValue=decision.score;mu0=decision.preliminary;muR=decision.regularization;muStar=decision.threshold;
            candidateMedian=decision.medianCandidate;deviationValue=decision.deviation;anomaly=decision.anomaly;
            reset=decision.reset;offlineUpdate=decision.offlineUpdate;
        }else{
            scoreValue=score(error,u);
            mu0=fixedThreshold;muR=0;muStar=fixedThreshold;candidateMedian=Double.NaN;deviationValue=Double.NaN;
            anomaly=scoreValue>muStar;
        }
        if(!anomaly){
            acceptedNormals.addLast(x.clone());
            while(acceptedNormals.size()>config.windowSize)acceptedNormals.removeFirst();
        }
        if(offlineUpdate)scheduleOfflineUpdate();
        return new Result(rowNumber,x.clone(),conceptNumber,u,error,scoreValue,muStar,mu0,muR,candidateMedian,deviationValue,
                config.useDto?dto.referenceError():referenceError,shiftMagnitude,dto.normalWindowSize(),dto.candidateWindowSize(),
                anomaly,drift,reset,offlineUpdate,detectorName);
    }

    /**
     * Drift signal with hysteresis. A concept is declared after DRIFT_CONFIRMATION consecutive
     * uncertain instances and the episode only closes after a sustained normal run, otherwise a
     * single flicker of U_ctr around mu_e would open thousands of one-row concepts and repeatedly
     * wipe the DTO window. The rising edge is what re-initialises the threshold statistics.
     */
    private DriftState evaluateDrift(boolean rawDrift){
        rowsProcessed++;
        if(rawDrift){
            driftRun++;
            normalRun=0;
        }else{
            driftRun=0;
            normalRun++;
        }
        boolean rising=false;
        // Warm-up: the first windowSize/2 records calibrate the stream against the historical concept and
        // cannot yet justify declaring a new one.
        if(!driftEpisodeActive&&driftRun>=DRIFT_CONFIRMATION&&rowsProcessed>=Math.max(DRIFT_CONFIRMATION,config.windowSize/2)){
            driftEpisodeActive=true;
            conceptNumber++;
            rowsSinceEpisode=0;
            rising=true;
        }else if(driftEpisodeActive&&normalRun>=driftClosure()){
            driftEpisodeActive=false;
        }
        rowsSinceEpisode++;
        return new DriftState(driftEpisodeActive,rising);
    }

    private int driftClosure(){return Math.max(5,config.windowSize/8);}

    private record DriftState(boolean drifting,boolean rising){}

    /**
     * Eq. (20): fine-tune the modules in a parallel worker and swap them into the live engine when
     * they are ready, so that online detection is never interrupted (the paper's parallel offline
     * training strategy).
     */
    private void scheduleOfflineUpdate(){
        if(refreshing||baseHistory==null)return;
        refreshing=true;
        List<double[]> refresh=new ArrayList<>(baseHistory);
        for(double[] row:acceptedNormals)refresh.add(row.clone());
        if(offlineExecutor==null)offlineExecutor=Executors.newSingleThreadExecutor(runnable->{
            Thread thread=new Thread(runnable,"dymeter-offline-update");
            thread.setDaemon(true);
            return thread;
        });
        offlineExecutor.submit(()->{
            try{
                // The paper fine-tunes the modules on the recent window in parallel and keeps serving
                // with the current parameters, so the refresh continues from the live model instead of
                // re-initialising it.
                DyMeterEngine trained=fork();
                trained.fineTune(refresh);
                applyRefresh(trained);
            }catch(RuntimeException failure){
                // A failing background refresh must never disturb the online detection service.
            }finally{
                refreshing=false;
            }
        });
    }

    private int minimumEpisodeGap(){return Math.max(8,config.windowSize/4);}

    /** Snapshot of the live model for the parallel offline update. */
    public DyMeterEngine fork(){
        DyMeterEngine copy=new DyMeterEngine(config);
        copy.buildModules(config.trainStaticBackbone);
        copy.detector.absorb(detector);
        copy.controller.absorb(controller);
        copy.hypernetwork.absorb(hypernetwork);
        copy.muE=muE;
        copy.muP=muP;
        copy.muT=muT;
        copy.referenceError=referenceError;
        copy.fixedThreshold=fixedThreshold;
        copy.trainingNormalScores=trainingNormalScores;
        copy.baseHistory=copyRows(baseHistory);
        copy.initialized=true;
        return copy;
    }

    /** Fine-tunes every module of an existing snapshot; used by the parallel offline update. */
    public void fineTune(List<double[]> rows){
        if(!initialized)throw new IllegalStateException("Fine-tuning requires an already trained engine.");
        RunProgress.offline(true);
        boolean rememberedBackbone=config.trainStaticBackbone;
        int rememberedEpochs=config.epochs;
        config.trainStaticBackbone=true;
        // The paper only states that the refresh fine-tunes the modules; a shorter pass keeps the
        // background worker light compared with the full historical training.
        config.epochs=Math.max(150,rememberedEpochs/5);
        try{
            train(rows);
        }finally{
            config.epochs=rememberedEpochs;
            config.trainStaticBackbone=rememberedBackbone;
            RunProgress.offline(false);
        }
    }

    private synchronized void applyRefresh(DyMeterEngine trained){
        detector.absorb(trained.detector);
        if(config.useIec)controller.absorb(trained.controller);
        if(config.useDsd)hypernetwork.absorb(trained.hypernetwork);
        if(!Double.isNaN(trained.muE))muE=trained.muE;
        muP=trained.muP;
        muT=trained.muT;
        referenceError=trained.referenceError;
        fixedThreshold=trained.fixedThreshold;
        // Re-initialise the DTO so that no stale error statistic survives the model refresh.
        threshold=new DynamicThresholdOptimizer(config.windowSize,config.tau,config.kappa,config.lambda,config.muO,
                config.maxInterval,config.emaRate,trained.muT,referenceError,config.quantileOnly,config.meanCandidate);
        threshold.setReferenceError(referenceError);
        threshold.seed(trained.trainingNormalScores);
        trainingNormalScores=trained.trainingNormalScores;
        refreshCount++;
    }

    private int refreshCount;

    public int refreshCount(){return refreshCount;}
    public long trainingMillis(){return trainingMillis;}
    public List<Double> trainingNormalScores(){return new ArrayList<>(trainingNormalScores);}
    public String variant(){return config.variant;}

    /** Releases the parallel offline-update worker. */
    public void shutdown(){
        if(offlineExecutor!=null){offlineExecutor.shutdownNow();offlineExecutor=null;}
    }

    private double score(double error,double u){return error*Math.exp(config.lambda*u*(referenceError-error));}

    private List<Double> toList(double[] values){
        List<Double> list=new ArrayList<>(values.length);
        for(double value:values)list.add(value);
        return list;
    }

    private static List<double[]> copyRows(List<double[]> rows){
        List<double[]> out=new ArrayList<>(rows.size());
        for(double[] row:rows)out.add(row.clone());
        return out;
    }
}
