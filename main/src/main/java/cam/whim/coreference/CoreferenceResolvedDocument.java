package cam.whim.coreference;

import cam.whim.coreference.simple.DiscourseEntity;
import cam.whim.narrative.chambersJurafsky.DependencyGraph;
import cam.whim.narrative.chambersJurafsky.DependencyReader;
import com.google.common.base.Joiner;
import opennlp.tools.util.Span;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Like a ParsedSentence, but with coreference resolution thrown in too. To save time, we only
 * read in certain bits of the parser output that are needed for post-coreference processing. It
 * may be necessary to add more in later if it turns out we need, e.g., OpenNLP parses.
 *
 */
public class CoreferenceResolvedDocument {
    public final List<String[]> words;
    public final List<String[]> lemmas;
    public final List<String[]> posTags;
    public final List<String[]> neTags;
    public final List<DependencyGraph> dependencyGraphs;
    public final List<DiscourseEntity> entities;

    public CoreferenceResolvedDocument(List<String[]> words,
                                       List<String[]> lemmas,
                                       List<String[]> posTags,
                                       List<String[]> neTags,
                                       List<DependencyGraph> dependencyGraphs,
                                       List<DiscourseEntity> entities) {
        this.words = words;
        this.lemmas = lemmas;
        this.posTags = posTags;
        this.dependencyGraphs = dependencyGraphs;
        this.entities = entities;
        this.neTags = neTags;
    }

    private CoreferenceResolvedDocument() {
        this.words = new ArrayList<String[]>();
        this.lemmas = new ArrayList<String[]>();
        this.posTags = new ArrayList<String[]>();
        this.dependencyGraphs = new ArrayList<DependencyGraph>();
        this.entities = new ArrayList<DiscourseEntity>();
        this.neTags = new ArrayList<String[]>();
    }

    public static CoreferenceResolvedDocument fromFiles(File tagFile, File depFile, File entitiesFile)
            throws IOException, ParsedSentence.SentenceReadError, DiscourseEntity.StringFormatError {
        // Read in tag lines
        List<String> tagLines = ParsedSentence.readTagLines(tagFile);
        // Split up the lines
        List<String[]> words = new ArrayList<String[]>();
        List<String[]> lemmas = new ArrayList<String[]>();
        List<String[]> posTags = new ArrayList<String[]>();
        List<String[]> neTags = new ArrayList<String[]>();
        CandcTagLine tagLine;
        try {
            for (String line : tagLines) {
                tagLine = CandcTagLine.fromLine(line);
                words.add(tagLine.words);
                lemmas.add(tagLine.lemmas);
                posTags.add(tagLine.posTags);
                neTags.add(tagLine.namedEntityTags);
            }
        } catch (CandcTagLine.LineFormatError lineFormatError) {
            throw new ParsedSentence.SentenceReadError("error reading tag line: " + lineFormatError.getMessage());
        }

        // Load dependency data
        List<DependencyGraph> dependencyGraphs = DependencyReader.getDependencyGraphs(depFile);

        // Read in the coreference resolution output
        List<DiscourseEntity> entities = DiscourseEntity.fromFile(entitiesFile);

        // Check we've got the same number of everything
        if (words.size() != dependencyGraphs.size()) {
            if (words.size() < 2 && dependencyGraphs.size() < 2)
                // Sometimes this happens: not worth worrying about
                return new CoreferenceResolvedDocument();

            throw new ParsedSentence.SentenceReadError("Differing numbers of tagged sentences (" + words.size() + ") and " +
                    "dependency graphs (" + dependencyGraphs.size() + ") in " + tagFile.getAbsolutePath() + " and " +
                    depFile.getAbsolutePath());
        }

        return new CoreferenceResolvedDocument(words, lemmas, posTags, neTags, dependencyGraphs, entities);
    }

    public Span getWordSpan(int sentenceNumber, Span characterSpan) throws SpanRangeError {
        int start = -1, end = -1;
        int charStart = characterSpan.getStart();
        int charEnd = characterSpan.getEnd();

        // How many chars we've consumed
        int cursor = 0;
        // The word that we're at the end of
        int wordNum = 0;
        for (String word : words.get(sentenceNumber)) {
            // Consume the first word (plus a space)
            cursor += word.length() + 1;

            // If we've not started yet and we've now passed the start index, the first word was the one we just saw
            if (start == -1 && charStart < cursor)
                start = wordNum;
            // Allow starting and ending on the same word
            // If we've passed the end index (probably just before the space), the previous word was the last in the range
            if (charEnd < cursor) {
                // Range end is non-inclusive
                end = wordNum + 1;
                break;
            }

            wordNum++;
        }

        if (end == -1)
            throw new SpanRangeError("character span (" + charStart + "," + charEnd + ") is not a valid word span " +
                    "for the sentence: " + Joiner.on(" ").join(words.get(sentenceNumber)));

        return new Span(start, end);
    }

    public static class SpanRangeError extends Exception {
        public SpanRangeError(String message) {
            super(message);
        }
    }
}
