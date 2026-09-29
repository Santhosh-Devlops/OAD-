package dymeter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;

/**
 * The dedicated ablation study of the paper ("Module effectiveness", "Focal objective function",
 * "Prior knowledge", "Instance-specific information", plus the DyMETER-Q and DyMETER-A threshold
 * designs) and the shared train/evaluate harness used by the CLI, the self-test and the web lab.
 */
final class Ablation {
    /** A named configuration of the same engine. */
    static final class Variant {
        final String name,description;
        private final Consumer<DyMeterEngine.Config> adjuster;
        Variant(String name,String description,Consumer<DyMeterEngine.Config> adjuster){
            this.name=name;this.description=description;this.adjuster=adjuster;
        }
        DyMeterEngine.Config build(DyMeterEngine.Config base){
            DyMeterEngine.Config config=base.copy();
            config.variant=name;
            adjuster.accept(config);
            return config;
        }
    }

    /** Evaluation outcome for one variant: metrics, drift behaviour and the per-row traces. */
    static final class Outcome {
        final String variant;
        final Metrics.Summary summary;
        final int streamRows,driftRows,driftEpisodes,thresholdResets,offlineUpdates;
        final long trainingMillis,inferenceMillis;
        final double[] scores,uncertainties,thresholds,errors;
        final boolean[] labels,decisions,drift,resets,updates;
        final int[] concepts;
        final double muE,muT,muP,meanUncertainty,uncertaintyP95,uncertaintyAboveRatio;
        Outcome(String variant,Metrics.Summary summary,int streamRows,int driftRows,int driftEpisodes,int thresholdResets,
                int offlineUpdates,long trainingMillis,long inferenceMillis,double[] scores,double[] uncertainties,
                double[] thresholds,double[] errors,boolean[] labels,boolean[] decisions,boolean[] drift,
                boolean[] resets,boolean[] updates,int[] concepts,double muE,double muT,double muP){
            this.variant=variant;this.summary=summary;this.streamRows=streamRows;this.driftRows=driftRows;
            this.driftEpisodes=driftEpisodes;this.thresholdResets=thresholdResets;this.offlineUpdates=offlineUpdates;
            this.trainingMillis=trainingMillis;this.inferenceMillis=inferenceMillis;this.scores=scores;
            this.uncertainties=uncertainties;this.thresholds=thresholds;this.errors=errors;this.labels=labels;
            this.decisions=decisions;this.drift=drift;this.resets=resets;this.updates=updates;this.concepts=concepts;
            this.muE=muE;this.muT=muT;this.muP=muP;this.meanUncertainty=Metrics.mean(uncertainties);
            this.uncertaintyP95=Numeric.quantile(boxed(uncertainties),0.95);
            int above=0;
            for(double value:uncertainties)if(value>muE)above++;
            this.uncertaintyAboveRatio=uncertainties.length==0?0:above/(double)uncertainties.length;
        }
        double aucRoc(){return summary.aucRoc;}
        double aucPr(){return summary.aucPr;}
    }

    private Ablation(){}

    static Variant full(){
        return new Variant("DyMETER","Full model: SCD + IEC + DSD + DTO",config -> {});
    }

    static List<Variant> variants(){
        List<Variant> variants=new ArrayList<>();
        variants.add(full());
        variants.add(new Variant("DyMETER-S","SCD only",config -> {
            config.useIec=false;config.useDsd=false;config.useDto=false;
        }));
        variants.add(new Variant("DyMETER-D","DSD only, static backbone untrained",config -> {
            config.useIec=false;config.useDsd=true;config.useDto=false;config.dynamicAlways=true;
            config.trainStaticBackbone=false;
        }));
        variants.add(new Variant("DyMETER-SD","SCD + DSD, fixed threshold",config -> {
            config.useIec=false;config.useDsd=true;config.useDto=false;config.dynamicAlways=true;
        }));
        variants.add(new Variant("DyMETER-SDT","SCD + DSD + DTO, no IEC",config -> {
            config.useIec=false;config.useDsd=true;config.useDto=true;config.dynamicAlways=true;
        }));
        variants.add(new Variant("DyMETER w/o IEC","full model without the IEC",config -> {
            config.useIec=false;config.useDsd=true;config.useDto=true;config.dynamicAlways=true;
        }));
        variants.add(new Variant("DyMETER w/o DTO","full model without the DTO",config -> config.useDto=false));
        variants.add(new Variant("DyMETER-ce","cross-entropy instead of the focal objective",config -> config.focalLoss=false));
        variants.add(new Variant("DyMETER-pl","1% of labelled anomalies injected as prior knowledge",config -> config.priorLabels=true));
        variants.add(new Variant("DyMETER-re","random embeddings instead of instance-conditioned shifts",config -> config.instanceConditioned=false));
        variants.add(new Variant("DyMETER-Q","quantile-only thresholding, mu_r = 0",config -> config.quantileOnly=true));
        variants.add(new Variant("DyMETER-A","mean instead of median for the candidate window",config -> config.meanCandidate=true));
        return variants;
    }

    static Variant byName(String name){
        for(Variant variant:variants())if(variant.name.equalsIgnoreCase(name))return variant;
        throw new IllegalArgumentException("Unknown variant '"+name+"'. Known variants: "+names());
    }

    static String names(){
        StringBuilder builder=new StringBuilder();
        for(Variant variant:variants()){
            if(builder.length()>0)builder.append(", ");
            builder.append(variant.name);
        }
        return builder.toString();
    }

    private static List<Double> boxed(double[] values){
        List<Double> list=new ArrayList<>(values.length);
        for(double value:values)list.add(value);
        return list;
    }

    /**
     * Trains one variant on the historical segment of the stream and then processes the remaining
     * records online, collecting the anomaly scores, decisions and drift signals.
     */
    static Outcome run(StreamData.Table table,DyMeterEngine.Config base,Variant variant,long seed){
        DyMeterEngine.Config config=variant.build(base);
        config.seed=seed;
        DyMeterEngine engine=new DyMeterEngine(config);
        int history=Math.max(2,Math.min(config.historicalRows,table.rows()-1));
        List<double[]> historyRows=new ArrayList<>(table.features.subList(0,history));
        boolean[] historyLabels=table.labelled()?Arrays.copyOfRange(table.labels,0,history):null;
        long started=System.nanoTime();
        engine.train(historyRows,historyLabels);
        long trained=System.nanoTime();
        int streamRows=table.rows()-history;
        double[] scores=new double[streamRows],uncertainties=new double[streamRows],thresholds=new double[streamRows],errors=new double[streamRows];
        boolean[] labels=new boolean[streamRows],decisions=new boolean[streamRows],driftFlags=new boolean[streamRows];
        boolean[] resets=new boolean[streamRows],updates=new boolean[streamRows];
        int[] concepts=new int[streamRows];
        int driftRows=0,resetCount=0,updateCount=0;
        RunProgress.stream(0,streamRows);
        for(int i=0;i<streamRows;i++){
            if(i%20==0)RunProgress.stream(i,streamRows);
            DyMeterEngine.Result result=engine.process(i+1,table.features.get(history+i));
            scores[i]=result.score();
            uncertainties[i]=result.uncertainty();
            thresholds[i]=result.threshold();
            errors[i]=result.error();
            decisions[i]=result.anomaly();
            driftFlags[i]=result.drift();
            resets[i]=result.thresholdReset();
            updates[i]=result.offlineUpdate();
            concepts[i]=result.concept();
            if(result.drift())driftRows++;
            if(result.thresholdReset())resetCount++;
            if(result.offlineUpdate())updateCount++;
            if(table.labelled())labels[i]=table.labels[history+i];
        }
        long finished=System.nanoTime();
        engine.shutdown();
        Metrics.Summary summary=table.labelled()
                ? Metrics.summarise(labels,scores,decisions)
                : new Metrics.Summary(streamRows,0,0,0,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN,Double.NaN);
        return new Outcome(variant.name,summary,streamRows,driftRows,engine.conceptCount(),resetCount,updateCount,
                (trained-started)/1000000,(finished-trained)/1000000,scores,uncertainties,thresholds,errors,
                labels,decisions,driftFlags,resets,updates,concepts,engine.muE(),engine.muT(),engine.muP());
    }

    /** Mean, standard deviation and the raw values of one metric across independent runs. */
    static final class Repeated {
        final String variant;
        final double[] aucRoc,aucPr;
        final double meanAucRoc,stdAucRoc,meanAucPr,meanFpr,meanRecall;
        final int driftEpisodes,driftRows,thresholdResets,offlineUpdates;
        final long trainingMillis,inferenceMillis;
        Repeated(String variant,double[] aucRoc,double[] aucPr,double meanFpr,double meanRecall,int driftEpisodes,
                 int driftRows,int thresholdResets,int offlineUpdates,long trainingMillis,long inferenceMillis){
            this.variant=variant;this.aucRoc=aucRoc;this.aucPr=aucPr;
            this.meanAucRoc=Metrics.mean(aucRoc);this.stdAucRoc=Metrics.standardDeviation(aucRoc);
            this.meanAucPr=Metrics.mean(aucPr);this.meanFpr=meanFpr;this.meanRecall=meanRecall;
            this.driftEpisodes=driftEpisodes;this.driftRows=driftRows;this.thresholdResets=thresholdResets;
            this.offlineUpdates=offlineUpdates;this.trainingMillis=trainingMillis;this.inferenceMillis=inferenceMillis;
        }
        double meanAucRoc(){return meanAucRoc;}
    }

    /** The paper reports the mean over five independent runs; this repeats one variant accordingly. */
    static Repeated repeat(StreamData.Table table,DyMeterEngine.Config base,Variant variant,int runs,long seed){
        double[] aucRoc=new double[runs],aucPr=new double[runs];
        double fprTotal=0,recallTotal=0;
        int episodes=0,driftRows=0,resets=0,updates=0;
        long training=0,inference=0;
        for(int run=0;run<runs;run++){
            Outcome outcome=run(table,base,variant,seed+run);
            aucRoc[run]=outcome.aucRoc();
            aucPr[run]=outcome.aucPr();
            fprTotal+=outcome.summary.falsePositiveRate;
            recallTotal+=outcome.summary.recall;
            episodes=Math.max(episodes,outcome.driftEpisodes);
            driftRows=Math.max(driftRows,outcome.driftRows);
            resets+=outcome.thresholdResets;
            updates+=outcome.offlineUpdates;
            training+=outcome.trainingMillis;
            inference+=outcome.inferenceMillis;
        }
        return new Repeated(variant.name,aucRoc,aucPr,fprTotal/runs,recallTotal/runs,episodes,driftRows,resets,updates,
                training/runs,inference/runs);
    }
}
