package cam.whim.coreference;

import cam.whim.coreference.simple.DiscourseEntity;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.impl.Arguments;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;

import java.io.*;
import java.util.List;

/**
 * Extract all event chains from many documents in sequence, to avoid having to reload
 * the models for each document. Directories where input files are found are specified as arguments
 * and document names are given on stdin
 */
public class StreamChainsExtractor {
    public static void main(String[] args) {
        ArgumentParser parser = ArgumentParsers.newArgumentParser("StreamChainsExtractor");
        parser.description("Extract all event chains from many documents in sequence, to avoid having to reload " +
                "the models for each document. Directories where input files are found are specified as arguments " +
                "and document names are given on stdin");
        parser.addArgument("models-dir").help("Directory containing OpenNLP coreference models");
        parser.addArgument("tag-dir").help("Directory containing tag (supertag, etc) files from C&C");
        parser.addArgument("dependency-dir").help("Directory containing GRs from C&C");
        parser.addArgument("coref-dir").help("Directory containing coreference resolution output");
        parser.addArgument("output-dir").help("Directory to output the chains files to");
        parser.addArgument("--silent", "-s")
                .action(Arguments.storeTrue())
                .help("Silent mode: don't output the protagonist name and number of events");
        parser.addArgument("--ignore-errors")
                .action(Arguments.storeTrue())
                .help("Output an empty file when there's an error reading the input data");
        parser.addArgument("--progress").help("Output this string after each completed document");
        parser.addArgument("--states")
                .action(Arguments.storeTrue())
                .help("Extract state information from adjectives as well as events from verbs");

        Namespace opts = null;
        try {
            opts = parser.parseArgs(args);
        } catch (ArgumentParserException e) {
            System.err.println("Error in command-line arguments: " + e);
            System.exit(1);
        }

        String modelsDirName = opts.getString("models-dir");
        String tagDirName = opts.getString("tag-dir");
        String dependencyDirName = opts.getString("dependency-dir");
        String corefDirName = opts.getString("coref-dir");
        String outputDirName = opts.getString("output-dir");
        boolean silent = opts.getBoolean("silent");
        boolean ignoreErrors = opts.getBoolean("ignore_errors");
        String progressString = opts.getString("progress");
        boolean extractStates = opts.getBoolean("states");

        // Load models
        ChainsExtractor extractor = new ChainsExtractor();

        // Get documents from stdin
        ObjectStream<String> lineStream = new PlainTextByLineStream(new InputStreamReader(System.in));

        String docname;
        List<List<ChainsExtractor.ChainElement>> chains;
        try {
            while ((docname = lineStream.read()) != null) {
                File tagFile = new File(tagDirName + "/" + docname);
                File dependencyFile = new File(dependencyDirName + "/" + docname);
                File corefFile = new File(corefDirName + "/" + docname);
                File outputFile = new File(outputDirName + "/" + docname);

                String eventString = "";

                try {
                    CoreferenceResolvedDocument document = CoreferenceResolvedDocument.fromFiles(
                            tagFile, dependencyFile, corefFile);

                    // Coref resolution
                    chains = extractor.extractChains(document, extractStates);
                    if (!silent)
                        System.out.println(chains.size() + " event chains found");

                    if (chains != null)
                        eventString = ChainsExtractor.Event.toChainsString(chains, docname);
                    
                } catch (ParsedSentence.SentenceReadError sentenceReadError) {
                    System.err.println("Error reading parse data: " + sentenceReadError.getMessage());
                    if (!ignoreErrors) continue;
                } catch (DiscourseEntity.StringFormatError stringFormatError) {
                    System.err.println("Error reading entity data: " + stringFormatError.getMessage());
                    if (!ignoreErrors) continue;
                } catch (Exception e) {
                    // Catch any exception and continue processing the next file
                    System.err.println("ERROR processing " + docname + ": " + e.getMessage());
                    e.printStackTrace();
                    if (!ignoreErrors) continue;
                }

                // Check the output dir exists
                if (!outputFile.getParentFile().exists()) {
                    outputFile.getParentFile().mkdirs();
                }
                // Output chains
                BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));
                writer.write(eventString);
                writer.close();

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
