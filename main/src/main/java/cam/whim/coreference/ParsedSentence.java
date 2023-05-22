package cam.whim.coreference;

import cam.whim.narrative.chambersJurafsky.DependencyGraph;
import cam.whim.narrative.chambersJurafsky.DependencyReader;
import com.google.common.base.Joiner;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.impl.Arguments;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import opennlp.tools.parser.AbstractBottomUpParser;
import opennlp.tools.parser.Parse;
import opennlp.tools.util.Span;

import java.io.*;
import java.util.*;

/**
 * Wrapper for the output from C&C and OpenNLP's parser for a single sentence.
 */
public class ParsedSentence {
    public final String[] words;
    public final Parse parse;
    public final String[] lemmas;
    public final DependencyGraph dependencyGraph;
    public final Map<String, List<Span>> namedEntities;

    public ParsedSentence(String[] words, Parse parse, String[] lemmas, DependencyGraph dependencyGraph,
                          Map<String, List<Span>> namedEntities) {
        this.words = words;
        this.parse = parse;
        this.lemmas = lemmas;
        this.dependencyGraph = dependencyGraph;
        this.namedEntities = namedEntities;

        assert lemmas.length == parse.getTagNodes().length;
        assert lemmas.length == words.length;
    }

    public int size() {
        return words.length;
    }

    /**
     * Read in several files with different parser output for a set of sentences.
     *
     * @param tagFile      C&C's tag output
     * @param depFile      C&C's GRs output
     * @param parseFile    OpenNLP's parser output
     * @return a ParsedSentence for each sentence found in the file set
     * @throws IOException
     * @throws SentenceReadError
     */
    public static List<ParsedSentence> readFiles(File tagFile, File depFile, File parseFile) throws IOException, SentenceReadError {
        // Load parse tree data
        List<Parse> parses = buildOpenNLPParses(parseFile);
        // Read in tag lines
        List<String> tagLines = readTagLines(tagFile);
        // Split up the lines
        List<CandcTagLine> candcTagLines = new ArrayList<CandcTagLine>();
        try {
            for (String line : tagLines)
                candcTagLines.add(CandcTagLine.fromLine(line));
        } catch (CandcTagLine.LineFormatError lineFormatError) {
            throw new SentenceReadError("error reading tag line: " + lineFormatError.getMessage());
        }

        // Load dependency data
        List<DependencyGraph> dependencyGraphs = DependencyReader.getDependencyGraphs(depFile);
        // Sometimes blank lines at the end get missed for some reason
        if (dependencyGraphs.size() == tagLines.size() - 1) {
            // Fill out with an extra empty dep graph
            dependencyGraphs.add(new DependencyGraph());
        }

        // Check we've got the same number of everything
        if (candcTagLines.size() != dependencyGraphs.size()) {
            if (candcTagLines.size() < 2 && dependencyGraphs.size() < 2)
                // Sometime this happens: not worth worrying about
                return new ArrayList<ParsedSentence>();
            else
                throw new SentenceReadError("Differing numbers of tagged sentences (" + candcTagLines.size() + ") and " +
                        "dependency graphs (" + dependencyGraphs.size() + ") in " + tagFile.getAbsolutePath() + " and " +
                        depFile.getAbsolutePath());
        }
        if (parses.size() != candcTagLines.size()) {
            if (parses.size() < 2 && candcTagLines.size() < 2)
                // Sometime this happens: not worth worrying about
                return new ArrayList<ParsedSentence>();
            else
                throw new SentenceReadError("Differing numbers of tagged sentences (" + candcTagLines.size() + ") and " +
                        "parse trees (" + parses.size() + ") in " + tagFile.getAbsolutePath() + " and " +
                        parseFile.getAbsolutePath());
        }

        // Wrap each sentence up
        List<ParsedSentence> sentences = new ArrayList<ParsedSentence>();
        for (int sentence = 0; sentence < parses.size(); sentence++) {
            try {
                sentences.add(
                        new ParsedSentence(
                                candcTagLines.get(sentence).words,
                                parses.get(sentence),
                                candcTagLines.get(sentence).lemmas,
                                dependencyGraphs.get(sentence),
                                candcTagLines.get(sentence).getNamedEntities()
                        ));
            } catch (CandcTagLine.LineFormatError lineFormatError) {
                throw new SentenceReadError("error processing named entities (sentence " + sentence + "): " +
                        lineFormatError.getMessage() + "; tags: " + Joiner.on(',').join(candcTagLines.get(sentence).namedEntityTags));
            }
        }

        return sentences;
    }

    public static List<Parse> buildOpenNLPParses(List<String> inputLines) {
        List<Parse> parsedText = new ArrayList<Parse>();

        for (String line : inputLines) {
            try {
                Parse p = Parse.parseParse(line);
                postProcessParse(p);
                parsedText.add(p);
            } catch (Exception e) {
                parsedText.add(null);
            }
        }
        return parsedText;
    }

    private static void postProcessParse(Parse parse) {
        if (parse.getType().equals("#")) {
            // These occur in parser output, but confuse the coreference system
            // Change them to something else
            parse.setType("HASH");
        }
        // Recurse to fix up the children too
        for (Parse child : parse.getChildren()) {
            postProcessParse(child);
        }
    }

    public static List<Parse> buildOpenNLPParses(File inputFile) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(inputFile));
        List<String> inputLines = new ArrayList<String>();

        String line;
        while ((line = reader.readLine()) != null)
            inputLines.add(line);

        // Remove blank lines from the end
        ListIterator<String> it = inputLines.listIterator(inputLines.size());
        while (it.hasPrevious() && (line = it.previous().trim()).isEmpty())
            it.remove();

        return buildOpenNLPParses(inputLines);
    }

    /**
     * Read in lines from C&C output
     *
     * @param inputFile    input file
     * @return list of lines
     * @throws IOException
     */
    protected static List<String> readTagLines(File inputFile) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(inputFile));
        String line;
        // Read in tagged text
        List<String> tagLines = new ArrayList<String>();
        while ((line = br.readLine()) != null)
            if (!line.trim().isEmpty()) tagLines.add(line.trim());
        return tagLines;
    }

    public static class SentenceReadError extends Exception {
        public SentenceReadError(String message) {
            super(message);
        }
    }

    public static void main(String[] args) {
        ArgumentParser parser = ArgumentParsers.newArgumentParser("ParsedSentence");
        parser.description("Test reading in a full ParsedSentence data structure from the various files it comes from");
        parser.addArgument("tag-file").help("Tag (supertag, etc) file from C&C");
        parser.addArgument("dependency-file").help("File containing GRs from C&C");
        parser.addArgument("parse-file").help("OpenNLP parse output");

        Namespace opts = null;
        try {
            opts = parser.parseArgs(args);
        } catch (ArgumentParserException e) {
            System.err.println("Error in command-line arguments: " + e);
            System.exit(1);
        }

        File tagFile = new File(opts.getString("tag-file"));
        File depFile = new File(opts.getString("dependency-file"));
        File parseFile = new File(opts.getString("parse-file"));

        try {
            List<ParsedSentence> sentences = ParsedSentence.readFiles(tagFile, depFile, parseFile);
            System.out.println("Successfully read " + sentences.size() + " sentences");
        } catch (IOException e) {
            e.printStackTrace();
        } catch (SentenceReadError sentenceReadError) {
            sentenceReadError.printStackTrace();
        }
    }
}
