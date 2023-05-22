package cam.whim.narrative.chambersJurafsky;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.impl.Arguments;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import opennlp.tools.coref.*;
import opennlp.tools.coref.mention.DefaultParse;
import opennlp.tools.coref.mention.Mention;
import opennlp.tools.coref.mention.MentionContext;
import opennlp.tools.parser.Parse;

import java.io.*;
import java.util.*;

/**
 * Wrapper for OpenNLP's coreference resolution tool.
 *
 */
public class ProtagonistExtractor {
    private Linker linker;

    /**
     * Load a coreference model stored in a directory. Can be used, for example, with OpenNLP's pre-trained
     * models.
     *
     * @param modelsDir    directory where model files live
     * @throws IOException if file can't be loaded
     */
    public ProtagonistExtractor(File modelsDir) throws IOException {
        // Load a coref linker suitable for shallow-parsed (chunked) input
        linker = new DefaultLinker(modelsDir.getAbsolutePath(), LinkerMode.TEST);
    }

    public ProtagonistExtractor(Linker linker) {
        this.linker = linker;
    }

    /**
     * Run coreference resolution on pre-parsed data.
     * @param parsedText    list of parsed sentences. May be shallow-parsed (i.e. chunked)
     * @return list of entities found in the passage
     */
    public DiscourseEntity[] resolveCoreference(List<Parse> parsedText) {
        List<opennlp.tools.coref.mention.Parse> wrappedParses = new ArrayList<opennlp.tools.coref.mention.Parse>();
        int num = 0;
        for (Parse parse : parsedText) {
            wrappedParses.add(new DefaultParse(parse, num));
            num++;
        }

        return resolveCoreferenceWrapped(wrappedParses);
    }

    /**
     * Run coreference resolution on pre-parsed data.
     * @param parsedText    list of parsed sentences. May be shallow-parsed (i.e. chunked)
     * @return list of entities found in the passage
     */
    public DiscourseEntity[] resolveCoreferenceWrapped(List<opennlp.tools.coref.mention.Parse> parsedText) {
        List<Mention> mentions = new ArrayList<Mention>();

        for (opennlp.tools.coref.mention.Parse parse : parsedText) {
            // Get entity mentions for coref resolution
            Mention[] extents = linker.getMentionFinder().getMentions(parse);

            // Leave out mentions that don't correspond to a constituent in the parse tree -- they'll be no use to us
            List<Mention> treeMentions = new ArrayList<Mention>();
            for (Mention extent : extents)
                if (extent.getParse() != null) treeMentions.add(extent);
            mentions.addAll(treeMentions);
        }

        if (!mentions.isEmpty())
            // Run coreference resolution
            return linker.getEntities(mentions.toArray(new Mention[mentions.size()]));
        else
            // No mentions, so nothing for coreference resolution to do
            return new DiscourseEntity[0];
    }

    /**
     * Run coreference resolution and return the most commonly mentioned entity in the document.
     *
     * @param parsedText    list of parsed sentences. Each should be a parse tree in PTB format
     * @return single most common entity
     */
    public DiscourseEntity extractProtagonist(List<Parse> parsedText) {
        DiscourseEntity[] entities = resolveCoreference(parsedText);

        // Find the most commonly mentioned entity
        int maxMentions = 0;
        DiscourseEntity protagonist = null;
        for (DiscourseEntity e : entities) {
            if (e.getNumMentions() > maxMentions) {
                maxMentions = e.getNumMentions();
                protagonist = e;
            }
        }
        return protagonist;
    }

    /**
     *
     * @param protagonist
     * @param parses shallow parse, constructed from C&C output. Should include POS tags.
     * @param dependencyGraphs
     * @return
     */
    public List<Event> extractVerbalDependencies(DiscourseEntity protagonist,
                                                 List<Parse> parses,
                                                 List<DependencyGraph> dependencyGraphs,
                                                 List<String[]> lemmas) {
        // Mention spans are stored as character indices, but we need to know the word numbers they start
        //  and end on to link them to dependency graphs
        List<Map<Integer, Integer>> leftEdgeToWordMaps = new ArrayList<Map<Integer, Integer>>();
        for (Parse p : parses)
            leftEdgeToWordMaps.add(ParseTreeUtils.getLeftEdgeToWordMap(p));
        List<Map<Integer, Integer>> rightEdgeToWordMaps = new ArrayList<Map<Integer, Integer>>();
        for (Parse p : parses)
            rightEdgeToWordMaps.add(ParseTreeUtils.getRightEdgeToWordMap(p));

        Iterator<MentionContext> mentions = protagonist.getMentions();
        List<Event> events = new ArrayList<Event>();
        while (mentions.hasNext()) {
            MentionContext mc = mentions.next();
            // Work out what parse it came from
            DependencyGraph dependencyGraph = dependencyGraphs.get(mc.getSentenceNumber());
            String[] sentenceLemmas = lemmas.get(mc.getSentenceNumber());
            // Pull out the POS tags
            List<String> sentencePosTags = new ArrayList<String>();
            for (Parse tagNode : parses.get(mc.getSentenceNumber()).getTagNodes())
                sentencePosTags.add(tagNode.getLabel());

            // Work out what word the mention begins and ends on
            int leftWord = leftEdgeToWordMaps.get(mc.getSentenceNumber()).get(mc.getSpan().getStart());
            int rightWord = rightEdgeToWordMaps.get(mc.getSentenceNumber()).get(mc.getSpan().getEnd());

            // Find verbal dependency heads governing words in this mention phrase
            String verb, verbLemma;
            String depType = "none";
            int verbIndex;
            EventType type;

            for (DependencyGraph.Dependency dep : dependencyGraph.dependencies) {
                verbLemma = "";

                if (dep.type.equals("ncsubj") && dep.args.length > 1 &&
                        leftWord <= dep.args[1].index && dep.args[1].index <= rightWord &&
                        sentencePosTags.get(dep.args[0].index).startsWith("VB")) {
                    // Verb subject
                    // Note that this includes passive subjects, which have args[2] == "obj" (previously nsubjpass)
                    verb = dep.args[0].word;
                    verbIndex = dep.args[0].index;

                    verbLemma = sentenceLemmas[verbIndex].toLowerCase();
                    if (verbLemma.equals("'m")) verbLemma = "be";   // Doesn't get lemmatized properly

                    depType = "nsubj";
                    type = EventType.NORMAL;

                    // If the lemma of the verb is "be", we might be able to get a predicative out of this
                    if ((verbLemma.equals("be") || verbLemma.equals("become"))) {
                        // Check whether there's an adjectival complement to the verb
                        List<DependencyGraph.Dependency> complements =
                                dependencyGraph.findDependencies("xcomp", 1, dep.args[0].index);
                        for (DependencyGraph.Dependency comp : complements) {
                            // Check whether this complement's an adjective
                            if (sentencePosTags.get(comp.args[2].index).startsWith("JJ")) {
                                // Found an adjectival verb complement: convert this extraction to a predicative
                                type = EventType.PREDICATIVE;
                                depType = verbLemma;
                                // Put the adjective in place of the verb
                                verb = comp.args[2].word.toLowerCase();
                                verbIndex = comp.args[2].index;
                                verbLemma = sentenceLemmas[verbIndex].toLowerCase();
                                // Don't look for any more
                                break;
                            }
                        }
                        // Exclude these lemmas if we didn't find a complement -- they don't tell us anything
                        if (depType.equals("nsubj")) {
                            continue;
                        }
                    }
                } else if (dep.type.equals("dobj") && dep.args.length > 1 &&
                        leftWord <= dep.args[1].index && dep.args[1].index <= rightWord &&
                        sentencePosTags.get(dep.args[0].index).startsWith("VB")) {
                    // Verb direct object
                    // I think this includes indirect objects (args[2] == "iobj"?) -- previously iobj relation
                    verb = dep.args[0].word;
                    verbIndex = dep.args[0].index;
                    depType = "dobj";
                    type = EventType.NORMAL;

                    verbLemma = sentenceLemmas[verbIndex].toLowerCase();
                    if (verbLemma.equals("'m")) verbLemma = "be";   // Doesn't get lemmatized properly

                    // be X is not interesting
                    if ((verbLemma.equals("be") || verbLemma.equals("become"))) continue;
                } else if (dep.type.equals("dobj") && dep.args.length > 1 &&
                        leftWord <= dep.args[1].index && dep.args[1].index <= rightWord &&
                        sentencePosTags.get(dep.args[0].index).equals("IN")) {
                    type = EventType.NORMAL;
                    // Dependent of a preposition
                    // Only include this if the preposition is attached to a verb
                    List<DependencyGraph.Dependency> prepAttachments =
                            dependencyGraph.findDependencies("iobj", 1, dep.args[0].index);
                    if (prepAttachments.size() == 0) {
                        continue;
                    }

                    // Look for an attachment to a verb
                    verb = null;
                    verbIndex = -1;
                    for (DependencyGraph.Dependency verbDep : prepAttachments) {
                        if (sentencePosTags.get(verbDep.args[0].index).startsWith("VB")) {
                            // Include the preposition with the dependency type
                            verb = verbDep.args[0].word;
                            verbIndex = verbDep.args[0].index;
                            depType = "prep_" + dep.args[0].word.toLowerCase();

                            verbLemma = sentenceLemmas[verbIndex].toLowerCase();
                            if (verbLemma.equals("'m")) verbLemma = "be";   // Doesn't get lemmatized properly
                            break;
                        }
                    }
                    if (verb == null)
                        // No verb attachment found
                        continue;
                } else {
                    continue;
                }
                verb = verb.toLowerCase();

                // Add particles to the verbs to get their event representation
                List<DependencyGraph.Dependency> verbParticles = dependencyGraph.findDependencies("ncmod", 1, verbIndex);
                for (DependencyGraph.Dependency verbParticle : verbParticles) {
                    if (verbParticle.args[2] != null &&
                            sentencePosTags.get(verbParticle.args[2].index).equals("RP")) {
                        // Found a particle associated with the verb -- add it on
                        verb += "+" + verbParticle.args[2].word.toLowerCase();
                        verbLemma += "+" + verbParticle.args[2].word.toLowerCase();
                        // Only add one particle per verb
                        break;
                    }
                }

                events.add(new Event(verb, verbLemma, depType, mc.getSentenceNumber(), leftWord, rightWord, type));
            }
        }

        return events;
    }

    /**
     * Simple data structure for events represented as verb dependencies.
     */
    public static class Event {
        public final String verb;
        public final String verbLemma;
        public final String dependencyType;
        public final int sentence;
        public final int leftWord;
        public final int rightWord;
        public final EventType type;

        public Event(String verb, String verbLemma, String dependencyType, int sentence, int leftWord, int rightWord, EventType type) {
            this.verb = verb;
            this.verbLemma = verbLemma;
            this.dependencyType = dependencyType;
            this.sentence = sentence;
            this.leftWord = leftWord;
            this.rightWord = rightWord;
            this.type = type;
        }

        @Override
        public String toString() {
            return "Event{" + verb + '-' + dependencyType + '}';
        }

        public List<String> getCsvColumns() {
            List<String> cols = new ArrayList<String>();
            cols.add(verb);
            cols.add(verbLemma);
            cols.add(dependencyType);
            cols.add("" + sentence);
            cols.add("" + leftWord);
            cols.add("" + rightWord);
            cols.add(type.name);
            return cols;
        }

        public List<String> getCsvColumns(String filename) {
            List<String> cols = new ArrayList<String>();
            cols.add(verb);
            cols.add(verbLemma);
            cols.add(dependencyType);
            cols.add("" + sentence);
            cols.add("" + leftWord);
            cols.add("" + rightWord);
            cols.add(filename);
            cols.add(type.name);
            return cols;
        }

        public static List<String> getEmptyLineCsvColumns(String filename) {
            List<String> cols = new ArrayList<String>();
            cols.add("NO ENTITIES/EVENTS");
            cols.add("");
            cols.add("");
            cols.add("");
            cols.add("");
            cols.add("");
            cols.add(filename);
            cols.add("");
            return cols;
        }
    }

    public static enum EventType {
        NORMAL ("norm"),
        PREDICATIVE ("pred");

        public final String name;
        EventType(String name) { this.name = name; }
    }

    public static void main(String[] args) {
        ArgumentParser parser = ArgumentParsers.newArgumentParser("ProtagonistExtractor");
        parser.description("Do coreference resolution to find the protagonist of a document and extract all the " +
                "events the protagonist is involved in");
        parser.addArgument("models-dir");
        parser.addArgument("pos-tag-file");
        parser.addArgument("dependency-parse-file");
        parser.addArgument("output.csv");
        parser.addArgument("--silent", "-s")
                .action(Arguments.storeTrue())
                .help("Silent mode: don't output the protagonist name and number of events");

        Namespace opts = null;
        try {
            opts = parser.parseArgs(args);
        } catch (ArgumentParserException e) {
            System.err.println("Error in command-line arguments: " + e);
            System.exit(1);
        }

        String modelsDirName = opts.getString("models-dir");
        String tagFileName = opts.getString("pos-tag-file");
        String dependencyFileName = opts.getString("dependency-parse-file");
        String outputFileName = opts.getString("output.csv");
        boolean silent = opts.getBoolean("silent");

        // Load models
        File modelsDir = new File(modelsDirName);
        ProtagonistExtractor coref = null;
        try {
            coref = new ProtagonistExtractor(modelsDir);
        } catch (IOException e) {
            System.err.println("Could not load coreference model");
            e.printStackTrace();
            System.exit(1);
        }

        File outputFile = new File(outputFileName);

        // TODO Fix this
        /*
        try {
            // Load parse tree data
            List<ParsedSentence> sentences = ParsedSentence.readFiles(new File(tagFileName), new File(dependencyFileName));

            DiscourseEntity protagonist = null;
            // Coref resolution
            // TODO Fix this
            protagonist = coref.extractProtagonist(sentences);

            BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile, true));
            CsvListWriter csvWriter = new CsvListWriter(writer, CsvPreference.EXCEL_PREFERENCE);

            if (protagonist != null) {
                // Got a protagonist: output it
                if (!silent)
                    System.out.println("Protagonist: " + protagonist.getMentions().next().toString().trim() + ", with " + protagonist.getNumMentions() + " mentions");

                // Look for verbal dependencies involving this protagonist
                // FIXME Update this for sentence data structure
                List<Event> events =
                        coref.extractVerbalDependencies(protagonist, parses, dependencyGraphs, lemmas);

                if (!silent)
                    System.out.println("Extracted " + events.size() + " events");

                // Output the events to a CSV file
                if (events.isEmpty()) {
                    csvWriter.write(Event.getEmptyLineCsvColumns(depGraphFile.getName()));
                } else {
                    for (Event event : events) {
                        csvWriter.write(event.getCsvColumns(depGraphFile.getName()));
                    }
                }
            } else {
                if (!silent)
                    System.out.println("No entities found");
                // Write a line to the CSV so we know the file's been processed and that nothing was found
                csvWriter.write(Event.getEmptyLineCsvColumns(depGraphFile.getName()));
            }

            csvWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        } catch (ParsedSentence.SentenceReadError sentenceReadError) {
            System.err.println("Error reading C&C output: " + sentenceReadError.getMessage());
            System.exit(1);
        }
        */
    }
}
