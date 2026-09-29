package dymeter;

import java.util.Locale;

/**
 * Live progress of the job that is currently running. The training loops publish the stage and the
 * epoch they are executing, the streaming loop publishes how many records have been processed, and the
 * dashboard polls {@code /api/progress} to show the user what is happening in the background.
 *
 * Only the thread that called {@link #begin} may update the epoch counters, so a background offline
 * refresh never overwrites the display of the run the user is watching.
 */
final class RunProgress {
    static final int STAGE_COUNT=4;

    private static volatile String dataset="",variant="",stage="idle",message="",job="";
    private static volatile int stageIndex,epoch,epochs,rows,totalRows,jobIndex,jobCount;
    private static volatile long startedAt;
    private static volatile Thread owner;
    private static volatile boolean running,offlineUpdate;
    private static int lastConsolePercent=-1;
    private static String lastConsoleLabel="";

    private RunProgress(){}

    static synchronized void begin(String datasetName,String variantName){
        dataset=datasetName;
        variant=variantName;
        job=variantName;
        jobIndex=0;
        jobCount=0;
        stage="starting";
        message="allocating the SCD, IEC and DSD modules";
        stageIndex=0;
        epoch=0;
        epochs=0;
        rows=0;
        totalRows=0;
        startedAt=System.currentTimeMillis();
        running=true;
        offlineUpdate=false;
        owner=Thread.currentThread();
        lastConsolePercent=-1;
        lastConsoleLabel="";
        System.out.println("Starting "+dataset+" - "+variant);
        consoleUpdate(true);
    }

    /** Marks the variant that the ablation loop is currently training. */
    static void job(String variantName,int index,int total){
        variant=variantName;
        job=variantName+" ("+index+"/"+total+")";
        jobIndex=index;
        jobCount=total;
        consoleUpdate(true);
    }

    /** Announces the stage that is about to execute, e.g. "SCD training" with 900 epochs. */
    static void stage(int index,String name,String text,int totalEpochs){
        if(!mine())return;
        stage=name;
        message=text;
        stageIndex=index;
        epochs=totalEpochs;
        epoch=0;
        rows=0;
        consoleUpdate(true);
    }

    static void epoch(int value){
        if(!mine())return;
        epoch=value;
        consoleUpdate(false);
    }

    /** Streaming stage: records processed out of the streamed segment. */
    static void stream(int value,int total){
        if(!mine())return;
        stage="streaming";
        message="processing the stream record by record (Algorithm 2)";
        stageIndex=STAGE_COUNT;
        epoch=0;
        epochs=0;
        rows=value;
        totalRows=total;
        consoleUpdate(false);
    }

    static void offline(boolean value){
        offlineUpdate=value;
        if(running)consoleUpdate(false);
    }

    static synchronized void finish(){
        consoleUpdate(true,1.0,"Complete");
        System.out.println();
        running=false;
        stage="idle";
        message="";
    }

    static synchronized void fail(String reason){
        double progress=fraction();
        running=false;
        String detail=reason==null||reason.isBlank()?"pipeline failed":reason.replace('\r',' ').replace('\n',' ');
        consoleUpdate(true,progress,"Failed: "+detail);
        System.out.println();
        stage="failed";
        message=detail;
    }

    private static boolean mine(){return Thread.currentThread()==owner;}

    private static void consoleUpdate(boolean force){
        consoleUpdate(force,fraction(),null);
    }

    private static synchronized void consoleUpdate(boolean force,double progress,String finalLabel){
        if(finalLabel==null&&!running)return;
        int percent=(int)Math.round(100*Math.max(0,Math.min(1,progress)));
        String label=finalLabel==null
                ?(jobCount>0?job+" | ":"")+stage+(offlineUpdate?" | offline fine-tuning":"")
                :finalLabel;
        if(!force&&percent==lastConsolePercent&&label.equals(lastConsoleLabel))return;
        int width=24,filled=(int)Math.round(width*percent/100.0);
        StringBuilder bar=new StringBuilder(width);
        for(int i=0;i<width;i++)bar.append(i<filled?'#':'-');
        String detail=label;
        if(finalLabel==null&&epochs>0)detail+=" | epoch "+epoch+"/"+epochs;
        else if(finalLabel==null&&stageIndex>=STAGE_COUNT)detail+=" | record "+rows+"/"+totalRows;
        System.out.printf(Locale.ROOT,"\rProgress [%s] %3d%% | %s",bar,percent,detail);
        System.out.flush();
        lastConsolePercent=percent;
        lastConsoleLabel=label;
    }

    /** Fraction of the work completed, combining the stage index with the current epoch or record. */
    static double fraction(){
        if(!running)return 0;
        int index=Math.max(1,stageIndex);
        double within;
        if(stageIndex>=STAGE_COUNT)within=totalRows>0?rows/(double)totalRows:0;
        else within=epochs>0?epoch/(double)epochs:0;
        return Math.min(1,(index-1+within)/STAGE_COUNT);
    }

    static String toJson(){
        StringBuilder json=new StringBuilder("{\"running\":").append(running);
        json.append(",\"dataset\":\"").append(escape(dataset)).append('"');
        json.append(",\"variant\":\"").append(escape(variant)).append('"');
        json.append(",\"job\":\"").append(escape(job)).append('"');
        json.append(",\"stage\":\"").append(escape(stage)).append('"');
        json.append(",\"message\":\"").append(escape(message)).append('"');
        json.append(String.format(Locale.US,",\"stageIndex\":%d,\"stageCount\":%d",stageIndex,STAGE_COUNT));
        json.append(String.format(Locale.US,",\"epoch\":%d,\"epochs\":%d",epoch,epochs));
        json.append(String.format(Locale.US,",\"rows\":%d,\"totalRows\":%d",rows,totalRows));
        json.append(String.format(Locale.US,",\"jobIndex\":%d,\"jobCount\":%d",jobIndex,jobCount));
        json.append(",\"offlineUpdate\":").append(offlineUpdate);
        json.append(String.format(Locale.US,",\"elapsedMillis\":%d,\"fraction\":%.6f",
                running?System.currentTimeMillis()-startedAt:0,fraction()));
        return json.append('}').toString();
    }

    private static String escape(String value){
        return value==null?"":value.replace("\\","\\\\").replace("\"","\\\"");
    }
}
