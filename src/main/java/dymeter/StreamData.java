package dymeter;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * CSV/stream loading for the lab. The label column ("label", "class", "target", "anomaly", ...) is
 * never used for training - DyMETER is unsupervised - it is kept so that the evaluation layer can
 * compute AUCROC/AUCPR and so that the DyMETER-pl ablation can inject 1% of labelled anomalies.
 */
final class StreamData {
    private static final String[] LABEL_NAMES={"label","class","target","anomaly","is_anomaly","is_outlier","outlier",
            "anomaly_label","attack","attack_label","outcome","ground_truth","groundtruth","y","y_true"};
    private static final String[] ANOMALY_VALUES={"abnormal","anomaly","anomalous","attack","fraud","fraudulent","yes",
            "true","malicious","positive","outlier","outliers","intrusion","infected","malware"};

    static final class Table {
        final String[] header;
        final int labelColumn;
        final List<double[]> features;
        final boolean[] labels;
        double[] mean;
        double[] deviation;
        Table(String[] header,int labelColumn,List<double[]> features,boolean[] labels){
            this.header=header;this.labelColumn=labelColumn;this.features=features;this.labels=labels;
        }
        int rows(){return features.size();}
        int dimensions(){return features.isEmpty()?0:features.get(0).length;}
        boolean labelled(){return labels.length>0;}
        Table slice(int from,int to){
            List<double[]> part=new ArrayList<>(features.subList(from,to));
            boolean[] partLabels=labels.length==0?new boolean[0]:Arrays.copyOfRange(labels,from,to);
            return new Table(header,labelColumn,part,partLabels);
        }
    }

    private StreamData(){}

    static Table load(Path path)throws IOException{return parse(Files.readString(path,StandardCharsets.UTF_8));}

    static Table parse(String text){
        String[] lines=text.replace("\r\n","\n").replace('\r','\n').split("\n",-1);
        int firstRow=0;
        while(firstRow<lines.length&&lines[firstRow].isBlank())firstRow++;
        if(firstRow>=lines.length)throw new IllegalArgumentException("The CSV is empty.");
        String[] header=splitCsv(lines[firstRow].replace("\ufeff","").trim());
        int labelColumn=-1;
        for(int j=0;j<header.length;j++){
            String name=clean(header[j]);
            for(String candidate:LABEL_NAMES)if(name.equals(candidate)){labelColumn=j;break;}
            if(labelColumn>=0)break;
        }
        if(header.length-(labelColumn>=0?1:0)<1)
            throw new IllegalArgumentException("The CSV must contain at least one numeric feature column.");
        List<double[]> features=new ArrayList<>();
        List<Boolean> labels=new ArrayList<>();
        for(int i=firstRow+1;i<lines.length;i++){
            if(lines[i].isBlank())continue;
            String[] cells=splitCsv(lines[i]);
            if(cells.length!=header.length)
                throw new IllegalArgumentException("CSV row "+(i+1)+" has "+cells.length+" columns but the header has "+header.length+".");
            double[] x=new double[header.length-(labelColumn>=0?1:0)];
            int index=0;
            for(int j=0;j<cells.length;j++){
                if(j==labelColumn){
                    if(cells[j].isBlank())
                        throw new IllegalArgumentException("The label in CSV row "+(i+1)+" is empty; fill it or remove the label column.");
                    labels.add(isAnomaly(cells[j]));
                    continue;
                }
                String cell=cells[j].trim();
                double value;
                try{
                    value=Double.parseDouble(cell);
                }catch(NumberFormatException failure){
                    throw new IllegalArgumentException("Column "+clean(header[j])+" of row "+(i+1)+" is not numeric ('"+cell+"').");
                }
                if(!Double.isFinite(value))throw new IllegalArgumentException("Column "+clean(header[j])+" of row "+(i+1)+" is not finite.");
                x[index++]=value;
            }
            features.add(x);
        }
        if(features.isEmpty())throw new IllegalArgumentException("The CSV has no data rows.");
        boolean[] labelArray=new boolean[labels.size()];
        for(int i=0;i<labelArray.length;i++)labelArray[i]=labels.get(i);
        return new Table(header,labelColumn,features,labelArray);
    }

    /** Z-score standardisation with the statistics of the historical segment (mean 0, variance 1). */
    static void standardise(Table table,int historyRows){
        int dimensions=table.dimensions();
        table.mean=new double[dimensions];
        table.deviation=new double[dimensions];
        for(int i=0;i<historyRows;i++){
            double[] x=table.features.get(i);
            for(int j=0;j<dimensions;j++)table.mean[j]+=x[j]/historyRows;
        }
        for(int i=0;i<historyRows;i++){
            double[] x=table.features.get(i);
            for(int j=0;j<dimensions;j++){
                double d=x[j]-table.mean[j];
                table.deviation[j]+=d*d/historyRows;
            }
        }
        for(int j=0;j<dimensions;j++){
            table.deviation[j]=Math.sqrt(table.deviation[j]);
            if(!(table.deviation[j]>1e-12))table.deviation[j]=1;
        }
        List<double[]> scaled=new ArrayList<>(table.rows());
        for(double[] x:table.features){
            double[] out=new double[dimensions];
            for(int j=0;j<dimensions;j++)out[j]=(x[j]-table.mean[j])/table.deviation[j];
            scaled.add(out);
        }
        table.features.clear();
        table.features.addAll(scaled);
    }

    /** Sequential equal subsets, the paper's p1..pN split of Table VII. */
    static List<Table> partition(Table table,int parts){
        List<Table> result=new ArrayList<>();
        int size=table.rows(),step=Math.max(1,size/parts);
        for(int k=0;k<parts;k++){
            int from=Math.min(size,k*step);
            int to=(k==parts-1)?size:Math.min(size,(k+1)*step);
            if(from<to)result.add(table.slice(from,to));
        }
        return result;
    }

    private static String[] splitCsv(String line){
        List<String> cells=new ArrayList<>();
        StringBuilder cell=new StringBuilder();
        boolean quoted=false;
        for(int i=0;i<line.length();i++){
            char ch=line.charAt(i);
            if(ch=='"'){quoted=!quoted;continue;}
            if(ch==','&&!quoted){cells.add(cell.toString());cell.setLength(0);continue;}
            cell.append(ch);
        }
        cells.add(cell.toString());
        return cells.toArray(new String[0]);
    }

    private static String clean(String value){return value.trim().replaceAll("[\"']","").toLowerCase(Locale.ROOT);}

    /** Non-zero numeric labels and the usual anomaly keywords count as anomalous. */
    static boolean isAnomaly(String cell){
        String raw=cell.trim().replaceAll("[\"']","");
        if(raw.isEmpty())return false;
        if(raw.equalsIgnoreCase("nan")||raw.equalsIgnoreCase("infinity")
                ||raw.equalsIgnoreCase("+infinity")||raw.equalsIgnoreCase("-infinity"))
            throw new IllegalArgumentException("Ground-truth labels must be finite numbers or recognized class names.");
        try{
            double numeric=Double.parseDouble(raw);
            if(!Double.isFinite(numeric))
                throw new IllegalArgumentException("Ground-truth labels must be finite numbers or recognized class names.");
            return numeric!=0;
        }catch(NumberFormatException failure){
            String value=raw.toLowerCase(Locale.ROOT);
            for(String candidate:ANOMALY_VALUES)if(value.equals(candidate))return true;
            return false;
        }
    }
}
