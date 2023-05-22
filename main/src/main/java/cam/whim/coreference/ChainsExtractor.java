package cam.whim.coreference;

import cam.whim.narrative.chambersJurafsky.DependencyGraph;
import com.google.common.base.Joiner;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.impl.Arguments;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import opennlp.tools.coref.sim.GenderEnum;
import opennlp.tools.util.Span;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Extract all the coreference chains from a document.
 * Wrapper for OpenNLP's coreference resolution tool, using the output from C&C.
 * Based on ProtagonistExtractor, but looks for *all* coreference chains. I think this is what we should have
 * done all along, so will probably remove ProtagonistExtractor later.
 *
 */
public class ChainsExtractor {
    /**
     * Extract all coreference chains from C&C parsed text (read in as a shallow parse).
     * Coreference resolution also needs to have been done previously.
     *
     * @param document    parsed sentences with coreference data
     * @return all coreference chains
     */
    public List<List<ChainElement>> extractChains(CoreferenceResolvedDocument document, boolean extractStates) {
        List<List<ChainElement>> eventChains = new ArrayList<List<ChainElement>>();

        // Don't bother looking at any entities that don't have 2 mentions -- we can't possibly get a chain
        for (cam.whim.coreference.simple.DiscourseEntity protagonist : document.entities)
            if (protagonist.mentions.size() > 1) {
                // Get an event chain for each protagonist
                try {
                    List<Event> events = extractVerbalDependencies(protagonist, document);
                    // Ignore this chain if it's not got two or more events in it
                    if (events.size() > 1) {
                        List<ChainElement> elements = new ArrayList<ChainElement>(events);
                        eventChains.add(elements);

                        // Also get states from adjectives
                        if (extractStates) {
                            List<State> states = extractAdjectivalDependencies(protagonist, document, events.get(0).neType);
                            // Add them into the same chain
                            eventChains.get(eventChains.size() - 1).addAll(states);
                        }
                    }
                } catch (CoreferenceResolvedDocument.SpanRangeError spanRangeError) {
                    // This almost certainly won't happen
                }
            }

        return eventChains;
    }

    /**
     * Some words don't get properly lemmatized: fix them in a hacky way.
     *
     * @param lemmas input lemma array
     * @return
     */
    private static void correctLemmas(String[] lemmas) {
        String lemma;
        for (int i = 0; i < lemmas.length; i++) {
            lemma = lemmas[i].toLowerCase();

            if (lemma.equals("'m")) lemmas[i] = "be";   // Don't get lemmatized properly
            if (lemma.equals("'s")) lemmas[i] = "be";
            if (lemma.equals("'re")) lemmas[i] = "be";

            // Get rid of any punctuation that's got through
            lemmas[i] = lemmas[i].replaceAll("\\W", "");
        }
    }

    /**
     *
     * @param protagonist
     * @param document all the sentences, with parse data read in from various files and coreference data
     * @return
     */
    public List<Event> extractVerbalDependencies(cam.whim.coreference.simple.DiscourseEntity protagonist,
                                                 CoreferenceResolvedDocument document) throws CoreferenceResolvedDocument.SpanRangeError {
        List<cam.whim.coreference.simple.MentionContext> mentions = protagonist.mentions;

        // Try to infer a NE type for the protagonist
        Span mentionWordSpan, headWordSpan;
        String entityType = "unknown";
        if ((protagonist.gender == GenderEnum.MALE || protagonist.gender == GenderEnum.FEMALE) &&
                protagonist.genderProb > 0.9) {
            // This entity has gender, so is presumably a person (this is English)
            // Only use this if it's really confident
            entityType = "person";
        } else {
            // Try to set the entity type by looking at NEs in the mentions
            List<String> neMentionTypes = new ArrayList<String>();

            String[] neTags;
            for (cam.whim.coreference.simple.MentionContext mention : mentions) {
                mentionWordSpan = document.getWordSpan(mention.sentenceNumber, mention.indexSpan);
                headWordSpan = document.getWordSpan(mention.sentenceNumber, mention.headSpan);
                // Look at the NE tags of the head NP
                neTags = Arrays.copyOfRange(document.neTags.get(mention.sentenceNumber), headWordSpan.getStart(), headWordSpan.getEnd());
                // If they're all the same (and not O), we have a NE type for this mention
                String neType = null;
                for (String neTag : neTags)
                    if (!neTag.equals("O") && ((neType == null || neTag.equals(neType))))
                        neType = neTag;
                // Collect up all the NE types from the mentions
                if (neType != null)
                    neMentionTypes.add(neType);
            }

            if (neMentionTypes.size() > 0) {
                // Take the most common NE type
                Multiset<String> neTypeCounter = HashMultiset.create(neMentionTypes);
                int maxCount = 0;
                String bestType = null;
                for (Multiset.Entry<String> entry : neTypeCounter.entrySet())
                    if (entry.getCount() > maxCount) {
                        bestType = entry.getElement();
                        maxCount = entry.getCount();
                    }
                // The tags are of the form I-*
                // Drop the I and look up the corresponding entity type
                if (bestType != null)
                    entityType = CandcTagLine.NE_MAP.get(bestType.substring(2));
            }
        }

        List<Event> events = new ArrayList<Event>();
        for (cam.whim.coreference.simple.MentionContext mention : mentions) {
            int sentenceNumber = mention.sentenceNumber;

            // Work out what parse it came from
            DependencyGraph dependencyGraph = document.dependencyGraphs.get(sentenceNumber);
            String[] sentenceLemmas = document.lemmas.get(sentenceNumber);
            correctLemmas(sentenceLemmas);
            String[] sentencePosTags = document.posTags.get(sentenceNumber);

            // Work out what word the mention begins and ends on
            // Only look at the head of the mention -- otherwise we get into trouble with complex NPs
            Span wordSpan = document.getWordSpan(sentenceNumber, mention.headSpan);
            int leftWord = wordSpan.getStart();
            int rightWord = wordSpan.getEnd() - 1;

            // Find verbal dependency heads governing words in this mention phrase
            String verb, verbLemma;
            String depType = "none";
            int verbIndex;
            EventType type;

            for (DependencyGraph.Dependency dep : dependencyGraph.dependencies) {
                verbLemma = "";

                if (dep.type.equals("ncsubj") && dep.args.length > 1 &&
                        leftWord <= dep.args[1].index && dep.args[1].index <= rightWord &&
                        sentencePosTags[dep.args[0].index].startsWith("VB")) {
                    // Verb subject
                    // Note that this includes passive subjects, which have args[2] == "obj" (previously nsubjpass)
                    verb = dep.args[0].word;
                    verbIndex = dep.args[0].index;

                    verbLemma = sentenceLemmas[verbIndex].toLowerCase();

                    depType = "nsubj";
                    type = EventType.NORMAL;

                    // If the lemma of the verb is "be", we might be able to get a predicative out of this
                    if ((verbLemma.equals("be") || verbLemma.equals("become"))) {
                        // Check whether there's an adjectival complement to the verb
                        List<DependencyGraph.Dependency> complements =
                                dependencyGraph.findDependencies("xcomp", 1, dep.args[0].index);
                        for (DependencyGraph.Dependency comp : complements) {
                            // Check whether this complement's an adjective
                            if (sentencePosTags[comp.args[2].index].startsWith("JJ")) {
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
                        sentencePosTags[dep.args[0].index].startsWith("VB")) {
                    // Verb direct object
                    // I think this includes indirect objects (args[2] == "iobj"?) -- previously iobj relation
                    verb = dep.args[0].word;
                    verbIndex = dep.args[0].index;
                    depType = "dobj";
                    type = EventType.NORMAL;

                    verbLemma = sentenceLemmas[verbIndex].toLowerCase();

                    // be X is not interesting
                    if ((verbLemma.equals("be") || verbLemma.equals("become"))) continue;
                } else if (dep.type.equals("dobj") && dep.args.length > 1 &&
                        leftWord <= dep.args[1].index && dep.args[1].index <= rightWord &&
                        // Preposition POS tags: IN and TO
                        (sentencePosTags[dep.args[0].index].equals("IN") ||
                         sentencePosTags[dep.args[0].index].equals("TO"))) {
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
                        if (sentencePosTags[verbDep.args[0].index].startsWith("VB")) {
                            // Include the preposition with the dependency type
                            verb = verbDep.args[0].word;
                            verbIndex = verbDep.args[0].index;
                            verbLemma = sentenceLemmas[verbIndex].toLowerCase();

                            String preposition = sentenceLemmas[dep.args[0].index].toLowerCase();
                            if (preposition.contains(",")) {
                                // This happens in very few cases and probably as a result of a tokenization/parsing error
                                continue;
                            }
                            depType = "prep_" + preposition;
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
                            sentencePosTags[verbParticle.args[2].index].equals("RP")) {
                        // Found a particle associated with the verb -- add it on
                        verb += "+" + verbParticle.args[2].word.toLowerCase();
                        verbLemma += "+" + verbParticle.args[2].word.toLowerCase();
                        // Only add one particle per verb
                        break;
                    }
                }

                // Filter the verb a bit - some funny characters get in
                verb = verb.replaceAll("[/}{,\\s]", "").trim();
                // Only use this event if there's something left of the verb
                if (verb.length() > 0) {
                    // Look at all verb arguments
                    Set<String> verbArgs = new HashSet<String>();
                    for (DependencyGraph.Dependency verbDep : dependencyGraph.findDependencies(0, verbIndex)) {
                        if (verbDep.type.equals("ncsubj"))
                            verbArgs.add("subj");
                        else if (verbDep.type.equals("dobj"))
                            verbArgs.add("obj");
                        else if (verbDep.type.equals("obj2"))
                            verbArgs.add("obj2");
                        else if (verbDep.type.equals("iobj"))
                            verbArgs.add("iobj-" + verbDep.args[1].word);
                    }

                    events.add(new Event(verb, verbLemma, depType, sentenceNumber, leftWord, rightWord, type,
                            verbIndex, entityType, verbArgs));
                }
            }
        }

        return events;
    }

    public List<State> extractAdjectivalDependencies(cam.whim.coreference.simple.DiscourseEntity protagonist,
                                                     CoreferenceResolvedDocument document, String neType)
            throws CoreferenceResolvedDocument.SpanRangeError {
        List<cam.whim.coreference.simple.MentionContext> mentions = protagonist.mentions;

        List<State> states = new ArrayList<State>();
        for (cam.whim.coreference.simple.MentionContext mention : mentions) {
            int sentenceNumber = mention.sentenceNumber;

            // Work out what parse it came from
            DependencyGraph dependencyGraph = document.dependencyGraphs.get(sentenceNumber);
            String[] sentenceLemmas = document.lemmas.get(sentenceNumber);
            correctLemmas(sentenceLemmas);
            String[] sentencePosTags = document.posTags.get(sentenceNumber);

            // Work out what word the mention begins and ends on
            // Only look at the head of the mention -- otherwise we get into trouble with complex NPs
            Span wordSpan = document.getWordSpan(sentenceNumber, mention.headSpan);
            int leftWord = wordSpan.getStart();
            int rightWord = wordSpan.getEnd() - 1;

            // Find adjectival dependencies of the mention
            String adjective, adjectiveLemma, depType;
            int adjectiveIndex;

            for (DependencyGraph.Dependency dep : dependencyGraph.dependencies) {
                if (dep.type.equals("ncmod") && dep.args.length > 2 &&
                        leftWord <= dep.args[1].index && dep.args[1].index <= rightWord &&
                        sentencePosTags[dep.args[2].index].startsWith("J")) {
                    adjective = dep.args[2].word.toLowerCase();
                    adjectiveIndex = dep.args[2].index;
                    adjectiveLemma = sentenceLemmas[adjectiveIndex].toLowerCase();
                    depType = "ncmod";

                    states.add(new State(adjective, adjectiveLemma, depType, sentenceNumber, leftWord, rightWord,
                            adjectiveIndex, neType));
                }
            }
        }

        return states;
    }

    public static abstract class ChainElement {
        public abstract String toChainString();
    }

    /**
     * Simple data structure for events represented as verb dependencies.
     */
    public static class Event extends ChainElement {
        public final String verb;
        public final String verbLemma;
        public final String dependencyType;
        public final int sentence;
        public final int leftWord;
        public final int rightWord;
        public final EventType type;
        public final int verbIndex;
        public final String neType;
        public final Set<String> verbArgTypes;

        public Event(String verb, String verbLemma, String dependencyType, int sentence, int leftWord, int rightWord,
                     EventType type, int verbIndex, String neType, Set<String> verbArgTypes) {
            this.verb = verb;
            this.verbLemma = verbLemma;
            this.dependencyType = dependencyType;
            this.sentence = sentence;
            this.leftWord = leftWord;
            this.rightWord = rightWord;
            this.type = type;
            this.verbIndex = verbIndex;
            this.neType = neType;
            this.verbArgTypes = verbArgTypes;
        }

        @Override
        public String toString() {
            return "Event{" + verb + '-' + dependencyType + '}';
        }

        public String toChainString() {
            return "{ " + verb + " / " + verbLemma + " / " + dependencyType + " / " + sentence + " ( " + leftWord +
                    "-" + rightWord + " ) / " + type + " / " + verbIndex + " / " + neType + " / " +
                    Joiner.on(',').join(verbArgTypes) + " }";
        }

        public static String toChainString(List<ChainElement> events, String docname) {
            List<String> eventStrings = new ArrayList<String>();
            for (ChainElement e : events)
                eventStrings.add(e.toChainString());
                
            String prefix = "";
            if (docname != null)
                prefix = docname + " ";
            
            return prefix + Joiner.on("").join(eventStrings);
        }

        public static String toChainsString(List<List<ChainElement>> chains, String docname) {
            StringBuilder sb = new StringBuilder();
            for (List<ChainElement> chain : chains)
                sb.append(Event.toChainString(chain, docname)).append("\n");
            return sb.toString();
        }
    }

    /**
     * Like an event, but represents a pairing of an adjective with the entity.
     */
    public static class State extends ChainElement {
        public final String adjective;
        public final String adjectiveLemma;
        public final String dependencyType;
        public final int sentence;
        public final int leftWord;
        public final int rightWord;
        public final int adjectiveIndex;
        public final String neType;

        public State(String verb, String verbLemma, String dependencyType, int sentence, int leftWord, int rightWord,
                     int verbIndex, String neType) {
            this.adjective = verb;
            this.adjectiveLemma = verbLemma;
            this.dependencyType = dependencyType;
            this.sentence = sentence;
            this.leftWord = leftWord;
            this.rightWord = rightWord;
            this.adjectiveIndex = verbIndex;
            this.neType = neType;
        }

        @Override
        public String toString() {
            return "State{" + adjective + '-' + dependencyType + '}';
        }

        public String toChainString() {
            return "{State " + adjective + " / " + adjectiveLemma + " / " + dependencyType + " / " + sentence + " ( " + leftWord +
                    "-" + rightWord + " ) / " + adjectiveIndex + " / " + neType + " }";
        }

        public static String toChainString(List<State> states, String docname) {
            List<String> stateStrings = new ArrayList<String>();
            for (State s : states)
                stateStrings.add(s.toChainString());

            String prefix = "";
            if (docname != null)
                prefix = docname + " ";

            return prefix + Joiner.on("").join(stateStrings);
        }

        public static String toChainsString(List<List<State>> chains, String docname) {
            StringBuilder sb = new StringBuilder();
            for (List<State> chain : chains)
                sb.append(State.toChainString(chain, docname)).append("\n");
            return sb.toString();
        }
    }

    /**
     * Distinguish different classes of extracted events.
     */
    public static enum EventType {
        NORMAL ("norm"),
        PREDICATIVE ("pred");

        public final String name;
        EventType(String name) { this.name = name; }
    }

    public static void main(String[] args) {
        ArgumentParser parser = ArgumentParsers.newArgumentParser("ChainsExtractor");
        parser.description("Extract all event chains from a document");
        parser.addArgument("models-dir").help("Directory containing OpenNLP coreference models");
        parser.addArgument("pos-tag-file").help("Tag (supertag, etc) file from C&C");
        parser.addArgument("dependency-parse-file").help("GRs from C&C");
        parser.addArgument("parse-file").help("Parse trees from OpenNLP");
        parser.addArgument("entities-file").help("Entities for the document from the output of coreference resolution");
        parser.addArgument("output").help("File to output the chains to");
        parser.addArgument("--silent", "-s")
                .action(Arguments.storeTrue())
                .help("Silent mode: don't output the protagonist name and number of events");
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
        String tagFileName = opts.getString("pos-tag-file");
        String dependencyFileName = opts.getString("dependency-parse-file");
        String parseFileName = opts.getString("parse-file");
        String entitiesFileName = opts.getString("entities-file");
        String outputFileName = opts.getString("output");
        File outputFile = new File(outputFileName);
        boolean silent = opts.getBoolean("silent");
        boolean extractStates = opts.getBoolean("states");

        // Load models
        ChainsExtractor extractor = new ChainsExtractor();

        List<List<ChainElement>> chains = null;
        try {
            // Load parse tree data
            try {
                CoreferenceResolvedDocument document = CoreferenceResolvedDocument.fromFiles(
                        new File(tagFileName), new File(dependencyFileName), new File(entitiesFileName));

                // Pull out all event chains
                chains = extractor.extractChains(document, extractStates);
                if (!silent)
                    System.out.println(chains.size() + " event chains found");

                // Output chains
                BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));
                if (chains != null)
                    writer.write(Event.toChainsString(chains, null));
                writer.close();
            } catch (ParsedSentence.SentenceReadError sentenceReadError) {
                System.err.println("Error reading parse data: " + sentenceReadError.getMessage());
                System.exit(1);
            } catch (cam.whim.coreference.simple.DiscourseEntity.StringFormatError stringFormatError) {
                System.err.println("Error reading coreference data: " + stringFormatError.getMessage());
                System.exit(1);
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
    }
}
