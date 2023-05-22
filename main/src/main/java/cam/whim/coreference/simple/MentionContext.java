package cam.whim.coreference.simple;

import opennlp.tools.util.Span;

/**
 * Simplifed MentionContext to store just the information we need after coref resolution is complete.
 *
 */
public class MentionContext {
    /**
     * Sentence-character-based span that the mention covers
     */
    public final Span indexSpan;

    /**
     * Position of the NP in the sentence.
     */
    public final int nounLocation;

    /**
     * Position of the NP in the document.
     */
    public final  int nounNumber;

    /**
     * Number of noun phrases in the sentence which contains this mention.
     */
    public final int maxNounLocation;

    /**
     * Index of the sentence in the document which contains this mention.
     */
    public final int sentenceNumber;

    public final String text;

    public final Span headSpan;

    public MentionContext(Span indexSpan, int nounLocation, int nounNumber, int maxNounLocation, int sentenceNumber,
                          String text, Span headSpan) {
        this.indexSpan = indexSpan;
        this.nounLocation = nounLocation;
        this.nounNumber = nounNumber;
        this.maxNounLocation = maxNounLocation;
        this.sentenceNumber = sentenceNumber;
        this.text = text;
        this.headSpan = headSpan;
    }

    public static MentionContext fromMentionContext(opennlp.tools.coref.mention.MentionContext mentionContext) {
        return new MentionContext(
                mentionContext.getIndexSpan(),
                mentionContext.getNounPhraseSentenceIndex(),
                mentionContext.getNounPhraseDocumentIndex(),
                mentionContext.getMaxNounPhraseSentenceIndex(),
                mentionContext.getSentenceNumber(),
                mentionContext.toText(),
                mentionContext.getHead().getSpan()
        );
    }

    public String toString() {
        StringBuffer sb = new StringBuffer();
        sb.append("(" + indexSpan.getStart() + "," + indexSpan.getEnd() + ")");
        sb.append(";");

        // The text might contain some of our dividers
        String storeText = text;
        storeText = storeText.replaceAll(";", "@semicolon@");
        storeText = storeText.replaceAll(" / ", "@slash@");
        storeText = storeText.replaceAll(" // ", "@slashes@");
        storeText = storeText.replaceAll(",", "@comma@");
        sb.append(storeText);
        sb.append(";");

        sb.append(nounLocation);
        sb.append(";");
        sb.append(nounNumber);
        sb.append(";");
        sb.append(maxNounLocation);
        sb.append(";");
        sb.append(sentenceNumber);
        sb.append(";");
        sb.append("(" + headSpan.getStart() + "," + headSpan.getEnd() + ")");

        return sb.toString();
    }

    public static MentionContext fromString(String mentionString) throws StringFormatError {
        String[] parts = mentionString.split(";");
        if (parts.length < 7)
            throw new StringFormatError("not enough information in mention: " + mentionString);

        // Extract the span
        String spanString = parts[0].substring(1, parts[0].length() - 1);
        String[] spanParts = spanString.split(",");
        Span span = new Span(Integer.parseInt(spanParts[0]), Integer.parseInt(spanParts[1]), null);

        // Extract the other integer values
        int nounLocation, nounNumber, maxNounLocation, sentenceNumber;
        try {
            nounLocation = Integer.parseInt(parts[2]);
            nounNumber = Integer.parseInt(parts[3]);
            maxNounLocation = Integer.parseInt(parts[4]);
            sentenceNumber = Integer.parseInt(parts[5]);
        } catch (NumberFormatException numberFormatException) {
            throw new StringFormatError("error parsing integers in fields 2-5 of '" + mentionString + "'");
        }

        // Get the head span
        String headSpanString = parts[6].substring(1, parts[6].length() - 1);
        String[] headSpanParts = headSpanString.split(",");
        Span headSpan = new Span(Integer.parseInt(headSpanParts[0]), Integer.parseInt(headSpanParts[1]), null);

        // Get the mention text
        String mentionText = parts[1];
        mentionText = mentionText.replaceAll("@semicolon@", ";");
        mentionText = mentionText.replaceAll("@slash@", " / ");
        mentionText = mentionText.replaceAll("@slashes@", " // ");
        mentionText = mentionText.replaceAll("@comma@", ",");

        return new MentionContext(span, nounLocation, nounNumber, maxNounLocation, sentenceNumber, mentionText, headSpan);
    }

    public static class StringFormatError extends Exception {
        public StringFormatError(String message) {
            super(message);
        }
    }
}
