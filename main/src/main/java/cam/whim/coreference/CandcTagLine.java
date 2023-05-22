package cam.whim.coreference;

import opennlp.tools.util.Span;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Simple wrapper around the different lists of elements contained in a tag line from C&C's output.
 *
 */
public class CandcTagLine {
    public final String[] words;
    public final String[] lemmas;
    public final String[] namedEntityTags;
    public final String[] posTags;
    private Map<String, List<Span>> namedEntities;

    public CandcTagLine(String[] words, String[] lemmas, String[] namedEntityTags, String[] posTags) {
        this.words = words;
        this.lemmas = lemmas;
        this.namedEntityTags = namedEntityTags;
        this.posTags = posTags;
    }

    /**
     * Get the map of named entity types to spans. This is computed from the tags read in the first time
     * it's needed, so if it's never used we don't waste time on it.
     *
     * @return NEs, grouped by type
     */
    public Map<String, List<Span>> getNamedEntities() throws LineFormatError {
        if (namedEntities == null)
            // Interpret the NE tags to get NE spans of different types
            namedEntities = buildNESpanMap(namedEntityTags);
        return namedEntities;
    }

    public static CandcTagLine fromLine(String line) throws LineFormatError {
        // Split up the tags
        String[] tokens = line.split("\\s+");
        String[] lemmas = new String[tokens.length];
        String[] words = new String[tokens.length];
        String[] neTags = new String[tokens.length];
        String[] posTags = new String[tokens.length];

        // Pull the lemmas out for each word
        for (int child = 0; child < tokens.length; child++) {
            // We need a special case for where the word itself is '|'
            // No idea why anyone would use this in a text, but it comes up
            if (tokens[child].startsWith("||||"))
                tokens[child] = tokens[child].replaceAll("\\|\\|", "/|");
            // Split up this token into its tags
            String[] tags = tokens[child].split("\\|");

            // Occasionally there are |s in the words, giving us too many parts
            if (tags.length > 6) {
                // Let's see what we can do
                // Split from the end
                String buff = tokens[child];

                // CCG supertag
                int div = buff.lastIndexOf('|');
                buff = buff.substring(0, div);

                // NE tag
                div = buff.lastIndexOf('|');
                neTags[child] = buff.substring(div + 1);
                buff = buff.substring(0, div);

                // Chunk tag
                div = buff.lastIndexOf('|');
                buff = buff.substring(0, div);

                // POS tag
                div = buff.lastIndexOf('|');
                posTags[child] = buff.substring(div + 1);
                buff = buff.substring(0, div);

                // The rest is the word + lemma
                // Assume they're the same length (can't split on |, because they include it), middle char is divider
                words[child] = buff.substring(0, (buff.length() / 2));
                lemmas[child] = buff.substring((buff.length() / 2) + 1);
            } else if (tags.length < 6) {
                throw new LineFormatError("not enough tags in '" + tokens[child] + "' in line: " + line);
            } else {
                words[child] = tags[0];
                lemmas[child] = tags[1];
                posTags[child] = tags[2];
                neTags[child] = tags[4];
            }
        }

        return new CandcTagLine(words, lemmas, neTags, posTags);
    }

    /* Constants for chunk state labels, read in from C&C */
    public static final String INSIDE = "I";
    public static final String BEGIN = "B";
    public static final String OUTSIDE = "O";

    /**
     * Mapping from the NER types output by C&C to the names used by OpenNLP.
     */
    public static final Map<String, String> NE_MAP = new HashMap<String, String>();
    static {
        NE_MAP.put("PER", "person");
        NE_MAP.put("LOC", "location");
        NE_MAP.put("ORG", "organization");
        NE_MAP.put("DAT", "date");
        NE_MAP.put("PCT", "percentage");
        NE_MAP.put("TIM", "time");
        NE_MAP.put("MON", "money");
    }

    /**
     * Extract all the named entity spans from C&C output tag data. Spans are grouped by their type.
     * The types used are mapped from C&C's vocabulary to OpenNLP's.
     *
     * @param neTags    input from C&C
     * @return  map from NE types to sentence spans
     */
    private static Map<String, List<Span>> buildNESpanMap(String[] neTags) throws LineFormatError {
        ///////////////
        // Interpret the C&C NER tags
        String neIob;
        String neType;
        String currentNeType = null;
        int neStart = -1;
        // Store a list of spans for each NE type
        Map<String, List<Span>> neTokenSpans = new HashMap<String, List<Span>>();
        for (String t : NE_MAP.keySet())
            neTokenSpans.put(t, new ArrayList<Span>());

        for (int w = 0; w <= neTags.length; w++) {
            // Split up the tag into its components
            if (w == neTags.length) {
                // End of the sequence: clean up if a NE is open
                neIob = OUTSIDE;
                neType = "";
            } else {
                String[] parts = neTags[w].split("-");
                neIob = parts[0];
                if (parts.length > 1)
                    neType = parts[1];
                else
                    neType = "";
            }

            if ((currentNeType != null) && (
                    // B- or O- markers after non-outside markers
                    (neIob.equals(BEGIN) || neIob.equals(OUTSIDE) ||
                            // I- followed by I-, but with different NE types
                            (neIob.equals(INSIDE) && !currentNeType.equals(neType))))) {
                // End of the current NE
                if (!neTokenSpans.containsKey(currentNeType)) {
                    throw new LineFormatError("unknown NE type: " + currentNeType);
                }
                neTokenSpans.get(currentNeType).add(new Span(neStart, w));
                currentNeType = null;
            }

            if (w < neTags.length && (
                    (neIob.equals(INSIDE) && currentNeType == null) ||
                            neIob.equals(BEGIN) ||
                            (neIob.equals(INSIDE) && !neType.equals(currentNeType)))) {
                // Start of a new NE
                neStart = w;
                currentNeType = neType;
            }
        }

        return neTokenSpans;
    }

    public static class LineFormatError extends Exception {
        public LineFormatError(String message) {
            super(message);
        }
    }
}
