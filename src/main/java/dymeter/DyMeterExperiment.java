package dymeter;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Command line experiment runner: reproduces the paper's evaluation protocol (AUCROC/AUCPR averaged
 * over independent runs) for the full model and for every ablation variant, optionally injecting
 * abrupt, gradual, recurrent or mixed concept drift with the stream generator.
 *
 * Example:
 *   java -cp out dymeter.DyMeterExperiment --data data/sample_stream.csv --variant all --runs 5 --drift abrupt
 */
public final class DyMeterExperiment {
    public static void main(String[] args)throws IOException{
        if(hasFlag(args,"--list")){
            System.out.println("Ablation variants: "+Ablation.names());
            return;
        }
        if(hasFlag(args,"-h")||hasFlag(args,"--help")){
            usage();
            return;
        }
        String generate=value(args,"--generate",null);
        if(generate!=null){
            StreamData.Table generated=StreamGenerator.synthetic(
                    Integer.parseInt(value(args,"--rows","1500")),
                    Integer.parseInt(value(args,"--dimensions","8")),
                    Integer.parseInt(value(args,"--concepts","3")),
                    Double.parseDouble(value(args,"--anomaly-rate","0.05")),
                    Double.parseDouble(value(args,"--anomaly-shift","6.0")),
                    Double.parseDouble(value(args,"--concept-shift","3.0")),
                    Long.parseLong(value(args,"--seed","42")));
            Path target=Path.of(generate);
            if(target.getParent()!=null)Files.createDirectories(target.getParent());
            Files.writeString(target,StreamGenerator.toCsv(generated),StandardCharsets.UTF_8);
            System.out.println("Wrote "+generated.rows()+" labelled synthetic rows to "+target.toAbsolutePath());
            if(value(args,"--data",null)==null)return;
        }
        String data=value(args,"--data",null);
        if(data==null){
            usage();
            throw new IllegalArgumentException("--data <csv> is required.");
        }
        Path source=Path.of(data);
        StreamData.Table table=StreamData.load(source);
        int limit=Integer.parseInt(value(args,"--limit","0"));
        if(limit>0&&limit<table.rows())table=table.slice(0,limit);
        int history=Math.max(2,(int)Math.ceil(table.rows()*0.2));
        StreamData.Table prepared=copy(table);
        StreamData.Table stream=prepareStream(prepared,args);
        if(stream!=prepared)StreamData.standardise(stream,Math.min(history,stream.rows()));
        else StreamData.standardise(stream,history);
        DyMeterEngine.Config base=DyMeterEngine.configure(stream.features);
        applyOverrides(base,args);
        List<Ablation.Variant> variants=new ArrayList<>();
        String variantName=value(args,"--variant","DyMETER");
        if(variantName.equalsIgnoreCase("all"))variants.addAll(Ablation.variants());
        else for(String name:variantName.split(","))variants.add(Ablation.byName(name.trim()));
        int runs=Integer.parseInt(value(args,"--runs","1"));
        long seed=Long.parseLong(value(args,"--seed","42"));
        System.out.printf(Locale.US,"Dataset      : %s (%d rows, %d features%s)%n",source.getFileName(),stream.rows(),
                stream.dimensions(),stream.labelled()?", labelled":"");
        System.out.printf(Locale.US,"Historical   : %d rows (20%%), stream %d rows, latent width %d, layers %d%n",
                base.historicalRows,stream.rows()-base.historicalRows,base.latentWidth,base.encoderLayers);
        System.out.printf(Locale.US,"Schedule     : Adam lr %.0e decayed x%.2f over %d epochs, batch %d, window %d, runs %d%n",
                base.learningRate,base.learningRateDecay,base.epochs,base.batchSize,base.windowSize,runs);
        System.out.println();
        System.out.printf(Locale.US,"%-42s %-18s %-9s %-8s %-8s %-6s %-6s %-7s %-8s %-8s%n",
                "Variant","AUCROC (mean±std)","AUCPR","FPR","Recall","Conc.","Drift","Resets","Train s","Infer s");
        System.out.println("-".repeat(130));
        List<String> markdown=new ArrayList<>();
        markdown.add("| Variant | AUCROC | AUCPR | FPR | Recall | Concepts | Drift rows | Threshold resets | Offline updates |");
        markdown.add("|---|---|---|---|---|---|---|---|---|");
        StringBuilder csv=new StringBuilder("variant,aucroc,aucpr,fpr,recall,concepts,drift_rows,threshold_resets,offline_updates,train_ms,inference_ms\n");
        for(Ablation.Variant variant:variants){
            String aucRoc,aucPr,fpr,recall;
            int episodes,driftRows,resets,updates;
            long training,inference;
            if(runs>1){
                Ablation.Repeated repeated=Ablation.repeat(stream,base,variant,runs,seed);
                aucRoc=Metrics.format(repeated.meanAucRoc)+"±"+Metrics.format(repeated.stdAucRoc);
                aucPr=Metrics.format(repeated.meanAucPr);
                fpr=Metrics.format(repeated.meanFpr);
                recall=Metrics.format(repeated.meanRecall);
                episodes=repeated.driftEpisodes;driftRows=repeated.driftRows;resets=repeated.thresholdResets;
                updates=repeated.offlineUpdates;training=repeated.trainingMillis;inference=repeated.inferenceMillis;
            }else{
                Ablation.Outcome outcome=Ablation.run(stream,base,variant,seed);
                aucRoc=Metrics.format(outcome.aucRoc());
                aucPr=Metrics.format(outcome.aucPr());
                fpr=Metrics.format(outcome.summary.falsePositiveRate);
                recall=Metrics.format(outcome.summary.recall);
                episodes=outcome.driftEpisodes;driftRows=outcome.driftRows;resets=outcome.thresholdResets;
                updates=outcome.offlineUpdates;training=outcome.trainingMillis;inference=outcome.inferenceMillis;
            }
            System.out.printf(Locale.US,"%-42s %-18s %-9s %-8s %-8s %-6d %-6d %-7d %-8d %-8d%n",variant.name,aucRoc,aucPr,
                    fpr,recall,episodes,driftRows,resets,training,inference);
            markdown.add(String.format(Locale.US,"| %s | %s | %s | %s | %s | %d | %d | %d | %d |",variant.name,aucRoc,aucPr,
                    fpr,recall,episodes,driftRows,resets,updates));
            csv.append(String.format(Locale.US,"%s,%s,%s,%s,%s,%d,%d,%d,%d,%d,%d%n",variant.name,aucRoc.replace('±',' '),aucPr,
                    fpr,recall,episodes,driftRows,resets,updates,training,inference));
        }
        String out=value(args,"--out","docs/results");
        if(!out.equalsIgnoreCase("none")){
            Path directory=Path.of(out);
            Files.createDirectories(directory);
            String stem=source.getFileName().toString().replaceAll("\\.[^.]+$","");
            Path csvPath=directory.resolve(stem+"-results.csv");
            Path markdownPath=directory.resolve(stem+"-results.md");
            Files.writeString(csvPath,csv.toString(),StandardCharsets.UTF_8);
            Files.writeString(markdownPath,String.join(System.lineSeparator(),markdown)+System.lineSeparator(),StandardCharsets.UTF_8);
            System.out.println();
            System.out.println("Wrote "+csvPath.toAbsolutePath());
            System.out.println("Wrote "+markdownPath.toAbsolutePath());
        }
    }

    private static StreamData.Table copy(StreamData.Table table){
        return new StreamData.Table(table.header,table.labelColumn,new ArrayList<>(table.features),table.labels.clone());
    }

    /** Builds the simulated stream: concept drift pattern, noise and concept-coverage scenarios. */
    private static StreamData.Table prepareStream(StreamData.Table table,String[] args){
        StreamGenerator.Drift drift=StreamGenerator.Drift.valueOf(value(args,"--drift","none").toUpperCase(Locale.ROOT));
        int concepts=Integer.parseInt(value(args,"--concepts","4"));
        int block=Integer.parseInt(value(args,"--block","500"));
        double magnitude=Double.parseDouble(value(args,"--magnitude","2.0"));
        long seed=Long.parseLong(value(args,"--seed","42"));
        StreamData.Table stream=StreamGenerator.generate(table,drift,concepts,block,magnitude,seed);
        int history=Math.max(2,(int)Math.ceil(stream.rows()*0.2));
        double noise=Double.parseDouble(value(args,"--noise","0"));
        double coverage=Double.parseDouble(value(args,"--coverage","1"));
        if(noise>0)StreamGenerator.addNoise(stream,history,noise,seed);
        if(coverage<1)StreamGenerator.subsampleTrainingRows(stream,history,coverage,seed);
        return stream;
    }

    private static void applyOverrides(DyMeterEngine.Config config,String[] args){
        config.epochs=Integer.parseInt(value(args,"--epochs",String.valueOf(config.epochs)));
        config.batchSize=Integer.parseInt(value(args,"--batch",String.valueOf(config.batchSize)));
        config.windowSize=Integer.parseInt(value(args,"--window",String.valueOf(config.windowSize)));
        config.tau=Double.parseDouble(value(args,"--tau",String.valueOf(config.tau)));
        config.lambda=Double.parseDouble(value(args,"--lambda",String.valueOf(config.lambda)));
        config.kappa=Double.parseDouble(value(args,"--kappa",String.valueOf(config.kappa)));
        config.gamma=Double.parseDouble(value(args,"--gamma",String.valueOf(config.gamma)));
        config.muO=Double.parseDouble(value(args,"--mu-o",String.valueOf(config.muO)));
        config.pseudoProportion=Double.parseDouble(value(args,"--mu-p",String.valueOf(config.pseudoProportion)));
        config.encoderLayers=Integer.parseInt(value(args,"--layers",String.valueOf(config.encoderLayers)));
        config.learningRate=Double.parseDouble(value(args,"--lr",String.valueOf(config.learningRate)));
        config.muE=Double.parseDouble(value(args,"--mu-e",String.valueOf(config.muE)));
    }

    private static String value(String[] args,String key,String fallback){
        for(int i=0;i<args.length-1;i++)if(args[i].equalsIgnoreCase(key))return args[i+1];
        return fallback;
    }

    private static boolean hasFlag(String[] args,String key){
        for(String argument:args)if(argument.equalsIgnoreCase(key))return true;
        return false;
    }

    private static void usage(){
        System.out.println("""
            DyMETER experiment runner (paper: Catching Every Ripple, TPAMI)

            java -cp out dymeter.DyMeterExperiment --data <csv> [options]

              --data <csv>        label,class,target,anomaly columns are used for evaluation only
              --generate <csv>    writes a labelled synthetic drifting dataset and exits
              --rows/--dimensions/--concepts/--anomaly-rate/--anomaly-shift/--concept-shift
              --variant <name>    DyMETER | all | comma separated list  (--list prints every variant)
              --runs <n>          independent runs averaged as mean±std (paper reports 5)
              --drift <type>      none | abrupt | gradual | recurrent | mixed
              --concepts <k>      number of nominal concepts used by the generator (default 4)
              --block <size>      rows per concept block (default 500)
              --magnitude <m>     size of the per-concept offset (default 2.0)
              --noise <ratio>     Gaussian noise on the training rows (robustness study)
              --coverage <frac>   fraction of training rows kept (concept-coverage study)
              --limit <rows>      use only the first rows of the CSV
              --epochs/--batch/--window/--layers/--lr/--tau/--lambda/--kappa/--gamma/--mu-o/--mu-p
              --seed <n>          random seed (default 42)
              --out <dir>         writes <dataset>-results.csv/.md (default docs/results, "none" disables)
              --list              prints the ablation variants and exits
              -h, --help          this message
            """);
        System.out.println("Ablation variants: "+Ablation.names());
    }
}
