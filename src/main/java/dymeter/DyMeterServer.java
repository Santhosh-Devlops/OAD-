package dymeter;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Executors;

/**
 * Local browser lab for the Java DyMETER port. It exposes the streaming API of the paper:
 *   POST /api/run       full four-stage DyMETER pipeline with per-row diagnostics, metrics and curves
 *   POST /api/ablation  the paper's ablation variants (SCD/DSD/DTO/IEC, focal loss, prior labels, ...)
 *   GET  /              the single page dashboard
 */
public final class DyMeterServer {
    private static final int PORT=8080;
    private static final int MAX_RESULT_ROWS=1500;

    public static void main(String[] args)throws Exception{
        HttpServer server=HttpServer.create(new InetSocketAddress(PORT),0);
        server.createContext("/api/run",DyMeterServer::run);
        server.createContext("/api/ablation",DyMeterServer::ablation);
        server.createContext("/api/progress",DyMeterServer::progress);
        server.createContext("/data/",DyMeterServer::dataset);
        server.createContext("/",DyMeterServer::page);
        server.setExecutor(Executors.newFixedThreadPool(4));
        server.start();
        System.out.println("DyMETER lab is ready at http://localhost:"+PORT);
        System.out.println("Modules: SCD (Eq. 3-4), IEC (Eq. 2, 5-8), DSD (Eq. 9-14), DTO (Eq. 15-20), Algorithms 1-2.");
    }

    private static void page(HttpExchange exchange)throws IOException{
        byte[] body=Files.readAllBytes(Path.of("src/main/resources/index.html"));
        exchange.getResponseHeaders().set("Content-Type","text/html; charset=utf-8");
        exchange.sendResponseHeaders(200,body.length);
        exchange.getResponseBody().write(body);
        exchange.close();
    }

    /** Live progress of the running job: stage, epoch, streamed records, elapsed time. */
    private static void progress(HttpExchange exchange)throws IOException{
        respond(exchange,200,RunProgress.toJson());
    }

    /** Serves the bundled datasets so that the lab can load the sample stream in the browser. */
    private static void dataset(HttpExchange exchange)throws IOException{
        String name=exchange.getRequestURI().getPath().substring("/data/".length());
        if(name.contains("..")||name.contains("/")||name.contains("\\")||!name.toLowerCase(Locale.ROOT).endsWith(".csv")){
            respond(exchange,400,error("Only CSV files inside the data directory can be served."));
            return;
        }
        Path path=Path.of("data",name);
        if(!Files.exists(path)){
            respond(exchange,404,error("data/"+name+" does not exist; generate it with --generate."));
            return;
        }
        byte[] body=Files.readAllBytes(path);
        exchange.getResponseHeaders().set("Content-Type","text/csv; charset=utf-8");
        exchange.sendResponseHeaders(200,body.length);
        exchange.getResponseBody().write(body);
        exchange.close();
    }

    private static void run(HttpExchange exchange)throws IOException{
        try{
            Map<String,String> parameters=body(exchange);
            StreamData.Table table=prepare(parameters);
            DyMeterEngine.Config base=DyMeterEngine.configure(table.features);
            override(base,parameters);
            Ablation.Variant variant=Ablation.byName(parameters.getOrDefault("variant","DyMETER"));
            String dataset=parameters.getOrDefault("dataset","");
            String filename=parameters.getOrDefault("filename",dataset.isBlank()?"uploaded-stream":dataset);
            RunProgress.begin(filename,variant.name);
            Ablation.Outcome outcome;
            try{
                outcome=Ablation.run(table,base,variant,seed(parameters));
            }finally{
                RunProgress.finish();
            }
            respond(exchange,200,result(table,base,variant,outcome,filename));
        }catch(Exception failure){
            respond(exchange,400,error(failure.getMessage()));
        }
    }

    private static void ablation(HttpExchange exchange)throws IOException{
        try{
            Map<String,String> parameters=body(exchange);
            StreamData.Table table=prepare(parameters);
            DyMeterEngine.Config base=DyMeterEngine.configure(table.features);
            override(base,parameters);
            long seed=seed(parameters);
            StringBuilder json=new StringBuilder("{\"data\":").append(data(table,parameters.getOrDefault("filename","uploaded-stream"))).append(",\"rows\":[");
            List<Ablation.Variant> variants=Ablation.variants();
            RunProgress.begin(parameters.getOrDefault("filename","uploaded-stream"),"ablation of "+variants.size()+" variants");
            boolean first=true;
            try{
                for(int index=0;index<variants.size();index++){
                    Ablation.Variant variant=variants.get(index);
                    RunProgress.job(variant.name,index+1,variants.size());
                    Ablation.Outcome outcome=Ablation.run(table,base,variant,seed);
                    if(!first)json.append(',');
                    first=false;
                    json.append(String.format(Locale.US,
                            "{\"name\":%s,\"description\":%s,\"aucroc\":%s,\"aucpr\":%s,\"fpr\":%s,\"recall\":%s,"
                                    + "\"driftEpisodes\":%d,\"thresholdResets\":%d,\"offlineUpdates\":%d}",
                            quote(variant.name),quote(variant.description),number(outcome.aucRoc()),number(outcome.aucPr()),
                            number(outcome.summary.falsePositiveRate),number(outcome.summary.recall),outcome.driftEpisodes,
                            outcome.thresholdResets,outcome.offlineUpdates));
                }
            }finally{
                RunProgress.finish();
            }
            json.append("]}");
            respond(exchange,200,json.toString());
        }catch(Exception failure){
            respond(exchange,400,error(failure.getMessage()));
        }
    }

    /** CSV text from the request, or a bundled dataset from data/, plus the generator scenario. */
    private static StreamData.Table prepare(Map<String,String> parameters){
        String dataset=parameters.get("dataset");
        String csv=parameters.get("csv");
        StreamData.Table table;
        if(csv!=null&&!csv.isBlank()){
            table=StreamData.parse(csv);
        }else if(dataset!=null&&!dataset.isBlank()){
            String name=dataset.trim();
            if(name.contains("..")||name.contains("/")||name.contains("\\"))
                throw new IllegalArgumentException("Invalid dataset name.");
            Path path=Path.of("data",name.endsWith(".csv")?name:name+".csv");
            if(!Files.exists(path))
                throw new IllegalArgumentException("data/"+name+" does not exist; generate it with --generate.");
            try{
                table=StreamData.load(path);
            }catch(IOException failure){
                throw new IllegalArgumentException("Could not read "+path+": "+failure.getMessage());
            }
        }else{
            throw new IllegalArgumentException("Upload a CSV with a header row and at least ten data rows.");
        }
        StreamGenerator.Drift drift=StreamGenerator.Drift.valueOf(parameters.getOrDefault("drift","none").toUpperCase(Locale.ROOT));
        long seed=seed(parameters);
        if(drift!=StreamGenerator.Drift.NONE){
            int concepts=Integer.parseInt(parameters.getOrDefault("concepts","4"));
            int block=Integer.parseInt(parameters.getOrDefault("block","300"));
            double magnitude=Double.parseDouble(parameters.getOrDefault("magnitude","2.0"));
            table=StreamGenerator.generate(table,drift,concepts,block,magnitude,seed);
        }else if(table.rows()>12000){
            table=table.slice(0,12000);
        }
        int history=Math.max(2,(int)Math.ceil(table.rows()*0.2));
        StreamData.standardise(table,history);
        double noise=Double.parseDouble(parameters.getOrDefault("noise","0"));
        if(noise>0)StreamGenerator.addNoise(table,history,noise,seed);
        return table;
    }

    private static void override(DyMeterEngine.Config config,Map<String,String> parameters){
        config.epochs=Integer.parseInt(parameters.getOrDefault("epochs",String.valueOf(config.epochs)));
        config.batchSize=Integer.parseInt(parameters.getOrDefault("batch",String.valueOf(config.batchSize)));
        config.tau=Double.parseDouble(parameters.getOrDefault("tau",String.valueOf(config.tau)));
        config.lambda=Double.parseDouble(parameters.getOrDefault("lambda",String.valueOf(config.lambda)));
        config.kappa=Double.parseDouble(parameters.getOrDefault("kappa",String.valueOf(config.kappa)));
        String muE=parameters.get("mue");
        if(muE!=null&&!muE.isBlank())config.muE=Double.parseDouble(muE.replace(',','.').trim());
    }

    private static long seed(Map<String,String> parameters){
        return Long.parseLong(parameters.getOrDefault("seed","42"));
    }

    private static String data(StreamData.Table table,String filename){
        StringBuilder json=new StringBuilder(String.format(Locale.US,
                "{\"filename\":%s,\"rows\":%d,\"dimensions\":%d,\"labelled\":%b,\"anomalies\":%d,\"labelColumn\":%d,\"columns\":[",
                quote(filename),table.rows(),table.dimensions(),table.labelled(),
                table.labelled()?countAnomalies(table):0,table.labelColumn));
        for(int j=0;j<table.header.length;j++){
            if(j>0)json.append(',');
            json.append(quote(table.header[j].trim()));
        }
        json.append("],\"means\":[");
        if(table.mean!=null)for(int j=0;j<table.mean.length;j++){
            if(j>0)json.append(',');
            json.append(number(table.mean[j]));
        }
        json.append("],\"deviations\":[");
        if(table.deviation!=null)for(int j=0;j<table.deviation.length;j++){
            if(j>0)json.append(',');
            json.append(number(table.deviation[j]));
        }
        return json.append("]}").toString();
    }

    private static int countAnomalies(StreamData.Table table){
        int total=0;
        for(boolean label:table.labels)if(label)total++;
        return total;
    }

    /** Human-readable explanation of why a concept-drift episode was declared at this record. */
    private static String driftReason(Ablation.Outcome outcome,int index,DyMeterEngine.Config base){
        StringBuilder reason=new StringBuilder();
        reason.append("Concept ").append(outcome.concepts[index]).append(" declared at stream row ").append(index+1)
                .append(": the smoothed concept uncertainty U_ctr = ")
                .append(String.format(Locale.US,"%.5f",outcome.uncertainties[index]))
                .append(" stayed above the learned threshold mu_e = ").append(String.format(Locale.US,"%.5f",outcome.muE))
                .append(" for 3 consecutive instances (Algorithm 2, line 3), so the IEC judged the static detector"
                        + " unreliable. Reconstruction error R_e = ")
                .append(String.format(Locale.US,"%.4f",outcome.errors[index]))
                .append(" against the historical reference r_e; the DTO re-initialised W_N and W_C (Eq. 17-19) and"
                        + " the instance was escalated to the dynamic detector Theta_d = Theta_s + Delta_Theta_d"
                        + " (Eq. 12). Instance flagged anomalous: ").append(outcome.decisions[index]).append('.');
        return reason.toString();
    }

    private static String result(StreamData.Table table,DyMeterEngine.Config base,Ablation.Variant variant,
                                 Ablation.Outcome outcome,String filename){
        StringBuilder json=new StringBuilder("{\"variant\":").append(quote(variant.name));
        json.append(",\"description\":").append(quote(variant.description));
        json.append(",\"dataset\":").append(data(table,filename));
        json.append(String.format(Locale.US,",\"epochs\":%d,\"timestamp\":%d",base.epochs,System.currentTimeMillis()));
        json.append(String.format(Locale.US,",\"config\":{\"rows\":%d,\"historical\":%d,\"latentWidth\":%d,\"encoderLayers\":%d,"
                        + "\"epochs\":%d,\"batch\":%d,\"window\":%d,\"tau\":%.4f,\"lambda\":%.4f,\"kappa\":%.4f,\"gamma\":%.2f,"
                        + "\"muO\":%.2f,\"learningRate\":%.0e,\"layers\":[",
                table.rows(),Math.min(base.historicalRows,table.rows()-1),base.latentWidth,base.encoderLayers,base.epochs,
                base.batchSize,base.windowSize,base.tau,base.lambda,base.kappa,base.gamma,base.muO,base.learningRate));
        int[] widths=AutoEncoder.symmetricWidths(table.dimensions(),base.latentWidth,base.encoderLayers);
        for(int i=0;i<widths.length;i++){
            if(i>0)json.append(',');
            json.append(widths[i]);
        }
        json.append("]}");
        json.append(String.format(Locale.US,",\"metrics\":{\"aucroc\":%s,\"aucpr\":%s,\"fpr\":%s,\"recall\":%s,"
                        + "\"precision\":%s,\"flagged\":%d,\"positives\":%d,\"negatives\":%d}",
                number(outcome.aucRoc()),number(outcome.aucPr()),number(outcome.summary.falsePositiveRate),
                number(outcome.summary.recall),number(outcome.summary.precision),outcome.summary.flagged,
                outcome.summary.positives,outcome.summary.negatives));
        json.append(String.format(Locale.US,",\"learned\":{\"muE\":%s,\"muP\":%s,\"muT\":%s}",
                number(outcome.muE),number(outcome.muP),number(outcome.muT)));
        json.append(",\"driftEvents\":[");
        boolean firstEvent=true;
        for(int i=0;i<outcome.streamRows;i++){
            if(!outcome.resets[i])continue;
            if(!firstEvent)json.append(',');
            firstEvent=false;
            json.append(String.format(Locale.US,
                    "{\"row\":%d,\"concept\":%d,\"uncertainty\":%s,\"error\":%s,\"threshold\":%s,\"anomaly\":%b,\"reason\":%s}",
                    i+1,outcome.concepts[i],number(outcome.uncertainties[i]),number(outcome.errors[i]),
                    number(outcome.thresholds[i]),outcome.decisions[i],
                    quote(driftReason(outcome,i,base))));
        }
        json.append(']');
        json.append(String.format(Locale.US,",\"summary\":{\"rows\":%d,\"driftRows\":%d,\"driftEpisodes\":%d,"
                        + "\"thresholdResets\":%d,\"offlineUpdates\":%d,\"trainingMillis\":%d,\"inferenceMillis\":%d}",
                outcome.streamRows,outcome.driftRows,outcome.driftEpisodes,outcome.thresholdResets,
                outcome.offlineUpdates,outcome.trainingMillis,outcome.inferenceMillis));
        json.append(",\"curves\":{\"score\":").append(numbers(outcome.scores))
                .append(",\"uncertainty\":").append(numbers(outcome.uncertainties))
                .append(",\"threshold\":").append(numbers(outcome.thresholds))
                .append(",\"error\":").append(numbers(outcome.errors))
                .append(",\"concept\":").append(integers(outcome.concepts))
                .append(",\"decision\":").append(booleans(outcome.decisions))
                .append(",\"drift\":").append(booleans(outcome.drift))
                .append(",\"reset\":").append(booleans(outcome.resets))
                .append(",\"updated\":").append(booleans(outcome.updates))
                .append(",\"label\":");
        json.append(table.labelled()?booleans(outcome.labels):"null");
        json.append('}').append(",\"results\":[");
        int limit=Math.min(outcome.streamRows,MAX_RESULT_ROWS);
        for(int i=0;i<limit;i++){
            if(i>0)json.append(',');
            json.append(String.format(Locale.US,"{\"row\":%d,\"concept\":%d,\"uncertainty\":%s,\"error\":%s,\"score\":%s,"
                            + "\"threshold\":%s,\"drift\":%b,\"reset\":%b,\"updated\":%b,\"anomaly\":%b}",
                    i+1,outcome.concepts[i],number(outcome.uncertainties[i]),number(outcome.errors[i]),
                    number(outcome.scores[i]),number(outcome.thresholds[i]),outcome.drift[i],outcome.resets[i],
                    outcome.updates[i],outcome.decisions[i]));
        }
        json.append("],\"truncated\":").append(outcome.streamRows>limit).append('}');
        return json.toString();
    }

    private static String numbers(double[] values){
        StringBuilder builder=new StringBuilder("[");
        for(int i=0;i<values.length;i++){
            if(i>0)builder.append(',');
            builder.append(number(values[i]));
        }
        return builder.append(']').toString();
    }

    private static String integers(int[] values){
        StringBuilder builder=new StringBuilder("[");
        for(int i=0;i<values.length;i++){
            if(i>0)builder.append(',');
            builder.append(values[i]);
        }
        return builder.append(']').toString();
    }

    private static String booleans(boolean[] values){
        StringBuilder builder=new StringBuilder("[");
        for(int i=0;i<values.length;i++){
            if(i>0)builder.append(',');
            builder.append(values[i]?1:0);
        }
        return builder.append(']').toString();
    }

    private static String number(double value){
        return Double.isNaN(value)||Double.isInfinite(value)?"null":String.format(Locale.US,"%.8g",value);
    }

    private static String quote(String value){
        return "\""+value.replace("\\","\\\\").replace("\"","\\\"").replace("\n"," ")+"\"";
    }

    private static String error(String message){
        return "{\"error\":"+quote(message==null?"Invalid input":message)+"}";
    }

    private static Map<String,String> body(HttpExchange exchange)throws IOException{
        String raw=new String(exchange.getRequestBody().readAllBytes(),StandardCharsets.UTF_8);
        Map<String,String> parameters=new LinkedHashMap<>();
        for(String pair:raw.split("&")){
            int separator=pair.indexOf('=');
            if(separator<0)continue;
            String key=java.net.URLDecoder.decode(pair.substring(0,separator),StandardCharsets.UTF_8);
            String value=java.net.URLDecoder.decode(pair.substring(separator+1),StandardCharsets.UTF_8);
            parameters.put(key,value);
        }
        return parameters;
    }

    private static void respond(HttpExchange exchange,int status,String payload)throws IOException{
        byte[] body=payload.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type","application/json; charset=utf-8");
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin","*");
        exchange.sendResponseHeaders(status,body.length);
        exchange.getResponseBody().write(body);
        exchange.close();
    }
}
