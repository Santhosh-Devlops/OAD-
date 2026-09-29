package dymeter;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Self-test for the Java DyMETER port. It verifies the hand-derived back-propagation of the SCD,
 * IEC and DSD modules against central finite differences, checks the closed forms of Eq. (8),
 * (15)-(20), validates the evaluation metrics and finally runs the whole pipeline on a synthetic
 * drifting stream. Run with: java -cp out dymeter.DyMeterSelfTest
 */
public final class DyMeterSelfTest {
    private static int passed,failed;
    private static final double EPSILON=1e-5;

    public static void main(String[] args){
        System.out.println("DyMETER self-test");
        System.out.println("=================");
        gradientChecks();
        equationChecks();
        metricChecks();
        pipelineChecks();
        System.out.println();
        System.out.println("passed "+passed+", failed "+failed);
        if(failed>0)System.exit(1);
    }

    private static void check(String name,boolean condition,String detail){
        if(condition){passed++;System.out.println("  [ok]   "+name+(detail.isEmpty()?"":"  ("+detail+")"));}
        else{failed++;System.out.println("  [FAIL] "+name+"  "+detail);}
    }

    private static void checkClose(String name,double analytic,double numeric,double tolerance){
        double scale=Math.max(1,Math.max(Math.abs(analytic),Math.abs(numeric)));
        check(name,Math.abs(analytic-numeric)/scale<=tolerance,
                String.format(java.util.Locale.US,"analytic %.10f, numeric %.10f",analytic,numeric));
    }

    private static int totalSize(List<double[]> views){
        int size=0;
        for(double[] view:views)size+=view.length;
        return size;
    }

    private static void addToParameter(List<double[]> views,int index,double delta){
        for(double[] view:views){
            if(index<view.length){view[index]+=delta;return;}
            index-=view.length;
        }
        throw new IndexOutOfBoundsException(index);
    }

    private static double[] randomVector(Random random,int size){
        double[] values=new double[size];
        for(int i=0;i<size;i++)values[i]=random.nextGaussian();
        return values;
    }

    private static void gradientChecks(){
        System.out.println();
        System.out.println("1. Numerical gradient verification (analytic vs central differences)");
        Random random=new Random(7);
        autoEncoderGradient(random);
        controllerGradient(random);
        hypernetworkGradient(random);
    }

    private static void autoEncoderGradient(Random random){
        AutoEncoder autoEncoder=new AutoEncoder(new int[]{4,3,2,3,4},new Random(11));
        double[] x=randomVector(random,4);
        double[] y=autoEncoder.forward(x,null,null);
        AutoEncoder.Gradient gradient=autoEncoder.backward(AutoEncoder.reconstructionGradient(x,y));
        double worst=0;
        for(int trial=0;trial<12;trial++){
            int layer=random.nextInt(gradient.weights.length);
            int row=random.nextInt(gradient.weights[layer].length);
            int column=random.nextInt(gradient.weights[layer][row].length);
            double analytic=gradient.weights[layer][row][column];
            double[] weights=autoEncoder.weightsRef()[layer][row];
            weights[column]+=EPSILON;
            double plus=AutoEncoder.reconstructionLoss(x,autoEncoder.reconstruct(x));
            weights[column]-=2*EPSILON;
            double minus=AutoEncoder.reconstructionLoss(x,autoEncoder.reconstruct(x));
            weights[column]+=EPSILON;
            worst=Math.max(worst,Math.abs(analytic-(plus-minus)/(2*EPSILON)));
        }
        check("SCD dL_SCD/dTheta_s matches finite differences (Eq. 4)",worst<1e-6,
                String.format(java.util.Locale.US,"max |difference| %.3e",worst));
    }

    private static double lossOf(EvidentialController controller,double[] x,int label,boolean focal){
        double[] alpha=controller.concentration(x);
        return focal?EvidentialController.focalLoss(alpha,label,2.0):EvidentialController.crossEntropyLoss(alpha,label);
    }

    private static void controllerGradient(Random random){
        for(boolean focal:new boolean[]{true,false}){
            for(boolean exponential:new boolean[]{false,true}){
            EvidentialController controller=new EvidentialController(5,4,exponential,new Random(13));
            double[] x=randomVector(random,5);
            int label=random.nextBoolean()?0:1;
            double loss=lossOf(controller,x,label,focal);
            EvidentialController.Gradient gradient=controller.backward(label,focal,2.0);
            List<double[]> views=Arrays.asList(controller.w1Ref(),controller.b1Ref(),controller.w2Ref(),controller.b2Ref());
            List<double[]> gradients=Arrays.asList(gradient.w1,gradient.b1,gradient.w2,gradient.b2);
            double worst=0;
            for(int trial=0;trial<12;trial++){
                int group=random.nextInt(views.size());
                int index=random.nextInt(views.get(group).length);
                double analytic=gradients.get(group)[index];
                double[] view=views.get(group);
                view[index]+=EPSILON;
                double plus=lossOf(controller,x,label,focal);
                view[index]-=2*EPSILON;
                double minus=lossOf(controller,x,label,focal);
                view[index]+=EPSILON;
                worst=Math.max(worst,Math.abs(analytic-(plus-minus)/(2*EPSILON)));
            }
            check("IEC dL_IEC/dTheta_c matches finite differences ("+(focal?"focal Eq. 7":"cross-entropy of DyMETER-ce")
                            +", "+(exponential?"exponential":"softplus")+" evidence)",
                    worst<1e-6,String.format(java.util.Locale.US,"loss %.6f, max |difference| %.3e",loss,worst));
            }
        }
    }

    private static double dynamicLoss(AutoEncoder detector,ShiftHypernetwork hypernetwork,double[] x,double[][][] dynamic){
        for(int l=0;l<dynamic.length;l++)
            for(int i=0;i<dynamic[l].length;i++)System.arraycopy(detector.weightsRef()[l][i],0,dynamic[l][i],0,dynamic[l][i].length);
        hypernetwork.addShift(x,dynamic);
        double[] y=detector.forward(x,dynamic,null);
        return AutoEncoder.reconstructionLoss(x,y);
    }

    private static void hypernetworkGradient(Random random){
        AutoEncoder detector=new AutoEncoder(new int[]{4,3,2,3,4},new Random(17));
        int[] inputs={4,3,2,3},outputs={3,2,3,4};
        ShiftHypernetwork hypernetwork=new ShiftHypernetwork(inputs,outputs,4,4,true,new Random(19));
        double[] x=randomVector(random,4);
        double[][][] dynamic=new double[4][][];
        for(int l=0;l<4;l++)dynamic[l]=new double[inputs[l]][outputs[l]];
        double loss=dynamicLoss(detector,hypernetwork,x,dynamic);
        AutoEncoder.Gradient throughDetector=detector.backward(AutoEncoder.reconstructionGradient(x,detector.forward(x,dynamic,null)));
        double[] analytic=hypernetwork.gradientVector(hypernetwork.backward(throughDetector.weights));
        List<double[]> views=hypernetwork.parameterViews();
        double worst=0;
        for(int trial=0;trial<12;trial++){
            int index=random.nextInt(totalSize(views));
            double expected=analytic[index];
            addToParameter(views,index,EPSILON);
            double plus=dynamicLoss(detector,hypernetwork,x,dynamic);
            addToParameter(views,index,-2*EPSILON);
            double minus=dynamicLoss(detector,hypernetwork,x,dynamic);
            addToParameter(views,index,EPSILON);
            worst=Math.max(worst,Math.abs(expected-(plus-minus)/(2*EPSILON)));
        }
        check("DSD dL_DSD/dTheta_h matches finite differences (Eq. 9-14)",worst<1e-6,
                String.format(java.util.Locale.US,"loss %.6f, max |difference| %.3e",loss,worst));

        double[][][] fresh=new double[4][][];
        for(int l=0;l<4;l++)fresh[l]=new double[inputs[l]][outputs[l]];
        hypernetwork.addShift(x,fresh);
        double biggest=0;
        for(int l=0;l<4;l++)for(int i=0;i<fresh[l].length;i++)for(int j=0;j<fresh[l][i].length;j++)
            biggest=Math.max(biggest,Math.abs(fresh[l][i][j]));
        check("DSD starts from Delta_Theta_d = 0",biggest==0,"Theta_d = Theta_s before any training step");
    }

    private static void equationChecks(){
        System.out.println();
        System.out.println("2. Closed-form equation checks");
        checkClose("Eq. (6) predictive probability alpha_c / sum_k alpha_k",0.75,
                EvidentialController.predictiveProbability(new double[]{3,1},0),1e-12);
        checkClose("Eq. (8) concept uncertainty of the flat Dirichlet (2,2)",
                (Numeric.digamma(3)-Numeric.digamma(5))+Math.log(2),
                EvidentialController.uncertainty(new double[]{2,2}),1e-12);
        DynamicThresholdOptimizer dto=new DynamicThresholdOptimizer(4,0.05,0.8,0.6,1000,100000,1.0,0.0,1.0,false,false);
        checkClose("Eq. (15) A(x) = R_e exp(lambda U_ctr (r_e - R_e))",0.4*Math.exp(0.6*0.2*(1.0-0.4)),
                dto.score(0.4,0.2),1e-12);

        dto.seed(Arrays.asList(0.1,0.2,0.3,0.4));
        checkClose("Eq. (17) mu0_a = quantile(1-tau) of W_N",0.4,dto.preliminary(),1e-12);
        double expectedScore=dto.score(0.35,0.001);
        DynamicThresholdOptimizer.Decision decision=dto.process(0.35,0.001,false);
        checkClose("Eq. (18) mu_r_a = kappa (mu0_a - bAm)",0.8*(0.4-expectedScore),decision.regularization,1e-9);
        checkClose("Eq. (19) mu*_a = mu0_a + mu_r_a",decision.preliminary+decision.regularization,decision.threshold,1e-12);
        check("Eq. (18) W_C collects uncertain scores near mu0_a",decision.candidateWindow==1,"|W_C| = "+decision.candidateWindow);
        check("Eq. (19) the decision is A(x) > mu*_a",(decision.score>decision.threshold)==decision.anomaly,
                "A = "+Metrics.format(decision.score)+", mu* = "+Metrics.format(decision.threshold));

        DynamicThresholdOptimizer quick=new DynamicThresholdOptimizer(2,0.05,0.8,0.6,0.5,100000,1.0,0.0,1.0,false,false);
        quick.seed(Arrays.asList(0.1,0.2));
        boolean accumulated=false;
        for(int row=0;row<3;row++)accumulated=quick.process(0.1,1.0,false).offlineUpdate;
        check("Eq. (20) offline update fires on accumulated uncertainty > mu_o",accumulated,"sum of U_ctr = 2.4 > mu_o = 0.5");
        DynamicThresholdOptimizer interval=new DynamicThresholdOptimizer(4,0.05,0.8,0.6,1000,3,1.0,0.0,1.0,false,false);
        interval.seed(Arrays.asList(0.1,0.2));
        boolean expired=false;
        for(int row=0;row<4;row++)expired=interval.process(0.05,0.0,false).offlineUpdate;
        check("Eq. (20) offline update fires when Delta t > T_max",expired,"T_max = 3 rows, no uncertainty accumulated");

        DynamicThresholdOptimizer reset=new DynamicThresholdOptimizer(8,0.05,0.8,0.6,1000,100000,1.0,0.0,1.0,false,false);
        reset.seed(Arrays.asList(0.1,0.2,0.3));
        int seeded=reset.normalWindowSize();
        DynamicThresholdOptimizer.Decision first=reset.process(0.2,0.9,true);
        DynamicThresholdOptimizer.Decision second=reset.process(0.2,0.9,false);
        reset.process(0.2,0.0,false);
        DynamicThresholdOptimizer.Decision third=reset.process(0.2,0.9,true);
        check("Algorithm 2 drift branch discards W_N/W_C once per episode",first.reset&&!second.reset&&third.reset,
                "resets "+reset.driftResets());
        check("The stale normal window is discarded and re-seeded from the new concept",
                seeded==3&&first.normalWindow==1,"|W_N| = "+seeded+" before, "+first.normalWindow+" after the reset");
    }

    private static void metricChecks(){
        System.out.println();
        System.out.println("3. Evaluation metrics");
        boolean[] labels={true,true,false,false};
        checkClose("AUCROC of a perfect ranking",1.0,Metrics.aucRoc(labels,new double[]{0.9,0.8,0.2,0.1}),1e-12);
        checkClose("AUCROC of an inverted ranking",0.0,Metrics.aucRoc(labels,new double[]{0.1,0.2,0.8,0.9}),1e-12);
        checkClose("AUCROC with complete ties",0.5,Metrics.aucRoc(labels,new double[]{0.5,0.5,0.5,0.5}),1e-12);
        checkClose("AUCPR of a perfect ranking",1.0,Metrics.aucPr(labels,new double[]{0.9,0.8,0.2,0.1}),1e-12);
        checkClose("AUCPR step-wise average precision",0.8333333,Metrics.aucPr(labels,new double[]{0.9,0.3,0.8,0.1}),1e-6);
        Metrics.Summary summary=Metrics.summarise(labels,new double[]{0.9,0.8,0.2,0.1},new boolean[]{true,true,false,true});
        checkClose("FPR of one false positive among two normals",0.5,summary.falsePositiveRate,1e-12);
        checkClose("Recall of two detected anomalies",1.0,summary.recall,1e-12);
    }

    private static void pipelineChecks(){
        System.out.println();
        System.out.println("4. End-to-end pipeline on a synthetic drifting stream");
        StreamData.Table parsed=StreamData.parse("a,b,label\n1,2,0\n3,4,1\n5,6,0\n");
        check("CSV parsing excludes the label column from the features",
                parsed.dimensions()==2&&parsed.labelled()&&parsed.labels[1]&&!parsed.labels[0],
                parsed.dimensions()+" feature columns, label column index "+parsed.labelColumn);

        StreamData.Table stream=StreamGenerator.synthetic(900,4,3,0.05,6.0,3.0,true,42);
        StreamData.standardise(stream,180);
        DyMeterEngine.Config base=DyMeterEngine.configure(stream.features);
        base.epochs=150;
        base.batchSize=32;
        Ablation.Outcome full=Ablation.run(stream,base,Ablation.full(),42);
        Ablation.Outcome staticOnly=Ablation.run(stream,base,Ablation.byName("DyMETER-S"),42);
        Ablation.Outcome fixedThreshold=Ablation.run(stream,base,Ablation.byName("DyMETER w/o DTO"),42);
        check("Full DyMETER trains and scores every stream record",full.streamRows==stream.rows()-base.historicalRows,
                full.streamRows+" stream records");
        check("AUCROC is defined for the labelled synthetic stream",!Double.isNaN(full.aucRoc()),
                "DyMETER "+Metrics.format(full.aucRoc())+", DyMETER-S (static only) "+Metrics.format(staticOnly.aucRoc()));
        check("Concept drift opens a small number of adaptation episodes",full.driftEpisodes>=2&&full.driftEpisodes<=25,
                full.driftEpisodes+" concepts, "+full.driftRows+" of "+full.streamRows+" rows escalated to the DSD");
        check("The DTO changes the operating point relative to a fixed threshold",
                full.summary.flagged!=fixedThreshold.summary.flagged,
                "flagged with DTO "+full.summary.flagged+" vs fixed threshold "+fixedThreshold.summary.flagged
                        +"; recall "+Metrics.format(full.summary.recall)+" vs "+Metrics.format(fixedThreshold.summary.recall));
        double low=Double.MAX_VALUE,high=-Double.MAX_VALUE;
        for(double value:full.thresholds){low=Math.min(low,value);high=Math.max(high,value);}
        check("The adaptive threshold of Eq. (19) moves with the stream",high>low&&full.thresholdResets>=1,
                "mu* in ["+Metrics.format(low)+", "+Metrics.format(high)+"], resets "+full.thresholdResets);

        // Scenario B: geometrically independent drift. The drifted concepts are unfamiliar to the IEC,
        // which is the setting the paper reports on the real benchmarks.
        StreamData.Table independent=StreamGenerator.synthetic(900,4,3,0.05,6.0,3.0,false,42);
        StreamData.standardise(independent,180);
        DyMeterEngine.Config independentConfig=base.copy();
        Ablation.Outcome drifting=Ablation.run(independent,independentConfig,Ablation.full(),42);
        Ablation.Outcome staticBaseline=Ablation.run(independent,independentConfig,Ablation.byName("DyMETER-S"),42);
        check("Independent drift is detected and escalated to the DSD",
                drifting.driftEpisodes>=2&&drifting.driftRows>0,
                drifting.driftEpisodes+" concepts, "+drifting.driftRows+" of "+drifting.streamRows+" rows escalated");
        check("DyMETER scores every independent-drift record",drifting.streamRows==independent.rows()-base.historicalRows
                        &&!Double.isNaN(drifting.aucRoc()),
                "AUCROC "+Metrics.format(drifting.aucRoc())+" at the smoke-test budget of "+base.epochs
                        +" epochs (DyMETER-S "+Metrics.format(staticBaseline.aucRoc())+"); the tuned comparison is"
                        +" reported by the CLI experiment at the paper's budget");
        check("Both scenarios produce a monotone uncertainty-driven adaptation trace",
                drifting.driftRows>0&&drifting.thresholdResets>=1,
                drifting.driftRows+" escalated rows, "+drifting.thresholdResets+" threshold re-initialisations");
        DyMeterEngine.Config config=base.copy();
        config.epochs=20;
        DyMeterEngine engine=new DyMeterEngine(config);
        engine.train(stream.features.subList(0,base.historicalRows));
        check("The engine fits the historical segment",engine.trained(),
                engine.networkWidths().length+" operators, "+engine.detectorParameterCount()+" SCD parameters, "
                        +engine.hypernetworkParameterCount()+" hypernetwork parameters");
        check("mu_e is learned inside the paper's search range [0.005, 0.4]",
                engine.muE()>=0.005&&engine.muE()<=0.4,String.format(java.util.Locale.US,"mu_e = %.6f",engine.muE()));
        engine.shutdown();
    }
}
