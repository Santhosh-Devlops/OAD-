package dymeter;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Streaming data generator. The paper "develops a data generator to simulate the generation of
 * streaming data" with abrupt, gradual (incremental) and recurrent drift, and additionally studies
 * noisy initialisation and insufficient concept coverage. All four scenarios are provided here.
 */
final class StreamGenerator {
    enum Drift { NONE, ABRUPT, GRADUAL, RECURRENT, MIXED }

    private StreamGenerator(){}

    /**
     * Re-assembles the input table into a drifting stream: rows are grouped into nominal concepts by
     * a one-dimensional k-means on the per-row mean, each concept receives its own offset, and the
     * concept blocks are concatenated following the requested drift pattern.
     */
    static StreamData.Table generate(StreamData.Table base,Drift drift,int concepts,int blockSize,double magnitude,long seed){
        if(drift==Drift.NONE||concepts<2||base.rows()<concepts*10)return base;
        Random random=new Random(seed);
        int[] assignment=clusterMeans(base,concepts);
        double[][] offsets=new double[concepts][];
        double[] direction=direction(base);
        for(int c=0;c<concepts;c++){
            offsets[c]=new double[base.dimensions()];
            double factor=magnitude*(c/(double)(concepts-1)-0.5);
            for(int j=0;j<base.dimensions();j++)offsets[c][j]=factor*direction[j];
        }
        List<List<Integer>> members=new ArrayList<>();
        for(int c=0;c<concepts;c++)members.add(new ArrayList<Integer>());
        for(int i=0;i<base.rows();i++)members.get(Math.min(concepts-1,assignment[i])).add(i);
        List<double[]> features=new ArrayList<>();
        List<Boolean> labels=new ArrayList<>();
        if(drift==Drift.MIXED){
            appendPattern(features,labels,base,members,offsets,Drift.ABRUPT,blockSize,random);
            appendPattern(features,labels,base,members,offsets,Drift.GRADUAL,blockSize,random);
            appendPattern(features,labels,base,members,offsets,Drift.RECURRENT,blockSize,random);
        }else{
            appendPattern(features,labels,base,members,offsets,drift,blockSize,random);
        }
        boolean[] labelArray=new boolean[labels.size()];
        for(int i=0;i<labelArray.length;i++)labelArray[i]=labels.get(i);
        return new StreamData.Table(base.header,base.labelColumn,features,labelArray);
    }

    private static void appendPattern(List<double[]> features,List<Boolean> labels,StreamData.Table base,
            List<List<Integer>> members,double[][] offsets,Drift drift,int blockSize,Random random){
        int concepts=members.size();
        switch(drift){
            case ABRUPT -> {
                for(int c=0;c<concepts;c++)appendBlock(features,labels,base,members.get(c),offsets[c],blockSize,random);
            }
            case RECURRENT -> {
                for(int round=0;round<2;round++)
                    for(int c=0;c<concepts;c++)appendBlock(features,labels,base,members.get(c),offsets[c],blockSize,random);
            }
            case GRADUAL -> {
                for(int c=0;c<concepts;c++){
                    double[] from=offsets[c],to=offsets[(c+1)%concepts];
                    appendBlock(features,labels,base,members.get(c),from,blockSize,random);
                    int step=Math.max(1,blockSize/8);
                    for(int t=1;t<=8;t++){
                        double[] blended=new double[from.length];
                        for(int j=0;j<blended.length;j++)blended[j]=from[j]+(to[j]-from[j])*t/9.0;
                        appendBlock(features,labels,base,members.get(c),blended,step,random);
                    }
                }
            }
            default -> appendBlock(features,labels,base,members.get(0),offsets[0],blockSize,random);
        }
    }

    private static void appendBlock(List<double[]> features,List<Boolean> labels,StreamData.Table base,
            List<Integer> memberPool,double[] offset,int count,Random random){
        if(memberPool.isEmpty())return;
        for(int i=0;i<count;i++){
            int index=memberPool.get(random.nextInt(memberPool.size()));
            double[] x=base.features.get(index).clone();
            for(int j=0;j<x.length;j++)x[j]+=offset[j];
            features.add(x);
            labels.add(base.labels.length>index&&base.labels[index]);
        }
    }

    /** Noisy initialisation (Section IV-J): Gaussian noise injected into the training records. */
    static void addNoise(StreamData.Table table,int rows,double noiseRatio,long seed){
        if(noiseRatio<=0)return;
        Random random=new Random(seed);
        int limit=Math.min(rows,table.rows());
        for(int i=0;i<limit;i++){
            double[] x=table.features.get(i);
            for(int j=0;j<x.length;j++)x[j]+=random.nextGaussian()*noiseRatio;
        }
    }

    /** Insufficient concept coverage (Section IV-J): only a fraction of the records is retained. */
    static void subsampleTrainingRows(StreamData.Table table,int rows,double coverage,long seed){
        if(coverage>=1||coverage<=0)return;
        Random random=new Random(seed);
        int limit=Math.min(rows,table.rows());
        boolean[] keep=new boolean[limit];
        int retained=0;
        for(int i=0;i<limit;i++)if(random.nextDouble()<coverage){keep[i]=true;retained++;}
        if(retained==0)keep[0]=true;
        List<double[]> features=new ArrayList<>(table.rows());
        for(int i=0;i<table.rows();i++){
            int source=i;
            if(i<limit&&!keep[i]){
                do{source=random.nextInt(limit);}while(!keep[source]);
            }
            features.add(table.features.get(source));
        }
        table.features.clear();
        table.features.addAll(features);
    }

    private static int[] clusterMeans(StreamData.Table table,int concepts){
        int rows=table.rows();
        double[] values=new double[rows];
        for(int i=0;i<rows;i++){
            double sum=0;
            for(double value:table.features.get(i))sum+=value;
            values[i]=sum/table.dimensions();
        }
        double minimum=Double.MAX_VALUE,maximum=-Double.MAX_VALUE;
        for(double value:values){minimum=Math.min(minimum,value);maximum=Math.max(maximum,value);}
        double[] centroids=new double[concepts];
        for(int c=0;c<concepts;c++)centroids[c]=minimum+(maximum-minimum)*(c+0.5)/concepts;
        int[] assignment=new int[rows];
        for(int iteration=0;iteration<25;iteration++){
            double[] sums=new double[concepts];
            int[] counts=new int[concepts];
            for(int i=0;i<rows;i++){
                int best=0;
                for(int c=1;c<concepts;c++)if(Math.abs(values[i]-centroids[c])<Math.abs(values[i]-centroids[best]))best=c;
                assignment[i]=best;
                sums[best]+=values[i];
                counts[best]++;
            }
            for(int c=0;c<concepts;c++)if(counts[c]>0)centroids[c]=sums[c]/counts[c];
        }
        return assignment;
    }

    /** Dominant direction used to offset the concepts, taken from the table mean. */
    private static double[] direction(StreamData.Table table){
        int dimensions=table.dimensions();
        double[] mean=new double[dimensions];
        for(double[] x:table.features)for(int j=0;j<dimensions;j++)mean[j]+=x[j]/table.rows();
        double norm=0;
        for(double value:mean)norm+=value*value;
        norm=Math.sqrt(norm);
        if(norm<1e-9){mean[0]=1;norm=1;}
        for(int j=0;j<dimensions;j++)mean[j]/=norm;
        return mean;
    }

    /**
     * Labelled synthetic stream: 'concepts' equal blocks, each centred on its own nominal state
     * (conceptShift apart on every feature), with injected anomalies that sit anomalyShift away in an
     * independent, alternating direction. Keeping the anomaly direction separate from the concept
     * shift reproduces the paper's setting: drifted normal data is *unfamiliar* (high concept
     * uncertainty, so the IEC can escalate to the DSD) while still being nominal, whereas an anomaly
     * is a rare deviation inside the current concept.
     */
    static StreamData.Table synthetic(int rows,int dimensions,int concepts,double anomalyRate,
                                      double anomalyShift,double conceptShift,long seed){
        return synthetic(rows,dimensions,concepts,anomalyRate,anomalyShift,conceptShift,false,seed);
    }

    /**
     * @param correlatedDrift when true the anomalies and the concept shift share the same direction, so
     *   a drifted concept lands inside the historical anomaly mode: the evidential controller has no
     *   evidence about that region and reports the high concept uncertainty that escalates to the DSD.
     *   This is the "an anomaly under one concept is nominal under another" case of the paper's
     *   introduction; with false the drift is geometrically independent and the IEC stays confident.
     */
    static StreamData.Table synthetic(int rows,int dimensions,int concepts,double anomalyRate,
                                      double anomalyShift,double conceptShift,boolean correlatedDrift,long seed){
        Random random=new Random(seed);
        String[] header=new String[dimensions+1];
        for(int j=0;j<dimensions;j++)header[j]="f"+(j+1);
        header[dimensions]="label";
        List<double[]> features=new ArrayList<>(rows);
        boolean[] labels=new boolean[rows];
        int blockSize=Math.max(1,rows/Math.max(1,concepts));
        for(int i=0;i<rows;i++){
            int concept=Math.min(concepts-1,i/blockSize);
            double centre=concept*conceptShift;
            boolean anomaly=random.nextDouble()<anomalyRate;
            double[] x=new double[dimensions];
            for(int j=0;j<dimensions;j++){
                double anomalyDirection=correlatedDrift||(j%2==0)?1:-1;
                x[j]=centre+random.nextGaussian()+(anomaly?anomalyShift*anomalyDirection:0);
            }
            features.add(x);
            labels[i]=anomaly;
        }
        return new StreamData.Table(header,dimensions,features,labels);
    }

    /** Serialises a table back to CSV, including the label column when present. */
    static String toCsv(StreamData.Table table){
        StringBuilder builder=new StringBuilder();
        for(int j=0;j<table.header.length;j++){
            if(j>0)builder.append(',');
            builder.append(table.header[j]);
        }
        builder.append(System.lineSeparator());
        for(int i=0;i<table.rows();i++){
            double[] x=table.features.get(i);
            for(int j=0;j<x.length;j++){
                if(j>0)builder.append(',');
                builder.append(String.format(java.util.Locale.US,"%.6f",x[j]));
            }
            if(table.labelled()){
                builder.append(',');
                builder.append(table.labels[i]?1:0);
            }
            builder.append(System.lineSeparator());
        }
        return builder.toString();
    }
}
