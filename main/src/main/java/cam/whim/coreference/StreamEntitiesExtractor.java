package cam.whim.coreference;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.impl.Arguments;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import opennlp.tools.coref.DiscourseEntity;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;

import java.io.*;
import java.lang.reflect.Field;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;

/**
 * Extract all entity mention sets from many documents in sequence, to avoid having to reload
 * the models for each document. Directories where input files are found are specified as arguments
 * and document names are given on stdin
 */
public class StreamEntitiesExtractor {
    public static void main(String[] args) {
        if (args.length > 0) {
            for (String arg : args) {
                System.out.println(arg);
            }
        } else {
            System.out.println("No arguments provided.");
        }

        ArgumentParser parser = ArgumentParsers.newArgumentParser("StreamChainsExtractor");
        parser.description("Extract all entity mention sets from many documents in sequence, to avoid having to reload " +
                "the models for each document. Directories where input files are found are specified as arguments " +
                "and document names are given on stdin");
        parser.addArgument("models-dir").help("Directory containing OpenNLP coreference models");
        parser.addArgument("tag-dir").help("Directory containing tag (supertag, etc) files from C&C");
        parser.addArgument("dependency-dir").help("Directory containing GRs from C&C");
        parser.addArgument("parse-dir").help("Directory containing parse trees from OpenNLP");
        parser.addArgument("output-dir").help("Directory to output the entities files to");
        parser.addArgument("--silent", "-s")
                .action(Arguments.storeTrue())
                .help("Silent mode: don't output the protagonist names and number of events");
        parser.addArgument("--progress").help("Output this string after each completed document");
        parser.addArgument("--single-mentions")
                .action(Arguments.storeTrue())
                .help("Include all detected entities, including those mentioned only once. By default, only output " +
                        "entities with multiple mentions");

        Namespace opts = null;
        try {
            opts = parser.parseArgs(args);
        } catch (ArgumentParserException e) {
            System.err.println("Error in command-line arguments: " + e);
            System.exit(1);
        }


        System.out.println("*** Traversing keys and values in opts ***" );
        for (Field field : opts.getClass().getFields()) {
            String key = field.getName();
            try {
                Object value = field.get(opts);
                System.out.println(key + ": " + value);
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            }
        }
        System.out.println("*** Traversing done ***" );

//        String modelsDirName = opts.getString("models-dir");
//        String tagDirName = opts.getString("tag-dir");
//        String dependencyDirName = opts.getString("dependency-dir");
//        String parseDirName = opts.getString("parse-dir");
//        String outputDirName = opts.getString("output-dir");
//        boolean singleMentions = opts.getBoolean("single_mentions");
//        boolean silent = opts.getBoolean("silent");
//        String progressString = opts.getString("progress");

        String projectPath = "/root/eventchains";
        String baseFileName="LDC2003T05";

        String modelsDirName =      "../../models/opennlp";
        String tagDirName =         projectPath + "/main/chains/gigaword-nyt/tmp/coref/pos/" + baseFileName;
        String dependencyDirName =  projectPath + "/main/chains/gigaword-nyt/tmp/coref/deps/" + baseFileName;
        String parseDirName =       projectPath + "/main/chains/gigaword-nyt/tmp/coref/parse/" + baseFileName;
        String outputDirName =      projectPath + "/main/chains/gigaword-nyt/tmp/coref/output/" + baseFileName;

        boolean singleMentions = false;
        boolean silent = true;
        String progressString = ".";

        // Load models
        System.out.println("models-dir is " + modelsDirName );
        File modelsDir = new File(modelsDirName);
        // Load a coref model
        Coreference coreference = null;
        try {
            coreference = new Coreference(new File(modelsDir.getAbsolutePath() + "/en-parser-chunking.bin"), modelsDir);
        } catch (Coreference.ModelLoadError modelLoadError) {
            modelLoadError.printStackTrace();
            System.exit(1);
        }

        // Get documents from stdin
        ObjectStream<String> lineStream = new PlainTextByLineStream(new InputStreamReader(System.in));

        String docname;
        try {
            while ((docname = lineStream.read()) != null) {
                File tagFile = new File(tagDirName + "/" + docname);
                File dependencyFile = new File(dependencyDirName + "/" + docname);
                File parseFile = new File(parseDirName + "/" + docname);
                File outputFile = new File(outputDirName + "/" + docname);

                try {
                    // Load parse data
                    List<ParsedSentence> sentences = ParsedSentence.readFiles(tagFile, dependencyFile, parseFile);

                    if (!silent)
                        System.out.println("Loaded " + sentences.size() + " sentences");

                    // Coref resolution
                    DiscourseEntity[] entities = coreference.resolveCoreferenceParsed(sentences);

                    if (!silent)
                        System.out.println("Found " + entities.length + " entities");

                    // Convert entities into a simpler data structure that we can store
                    List<cam.whim.coreference.simple.DiscourseEntity> simpleEntities =
                            cam.whim.coreference.simple.DiscourseEntity.fromDocumentEntities(entities);

                    // Unless outputting all entities, remove those with only one mention
                    if (!singleMentions) {
                        for (ListIterator<cam.whim.coreference.simple.DiscourseEntity> entityIterator = simpleEntities.listIterator(); entityIterator.hasNext(); ) {
                            if (entityIterator.next().mentions.size() < 2)
                                entityIterator.remove();
                        }
                    }

                    if (!silent)
                        System.out.println(simpleEntities.size() + " event chains found");

                    // Check the output dir exists
                    if (!outputFile.getParentFile().exists()) {
                        outputFile.getParentFile().mkdirs();
                    }
                    // Output entity sets
                    BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));
                    for (cam.whim.coreference.simple.DiscourseEntity entity : simpleEntities) {
                        // Write each entity to a line of the file
                        writer.write(entity.toString());
                        writer.newLine();
                    }
                    writer.close();
                } catch (ParsedSentence.SentenceReadError sentenceReadError) {
                    System.err.println("Error reading parse data for " + docname + ": " + sentenceReadError.getMessage());
                    // Continue to next input
                } catch (Exception e) {
                    try {
                        // Catch any exception and continue processing the next file
                        System.err.println("ERROR processing " + docname + ": " + e.getMessage());
                    } catch (AbstractMethodError error) {
                        System.err.println("ERROR processing " + docname + ": " + e.getClass().toString());
                    }
                    e.printStackTrace();
                }
                if (progressString != null)
                    System.err.print(progressString);
            }
        } catch (IOException e) {
            System.err.println("Error reading/writing: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
