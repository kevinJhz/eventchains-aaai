package cam.whim.narrative.chambersJurafsky;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Read in POS tags from the output of the Stanford POS tagger.
 * Should be sentence per line, words separated by spaces, words separated from their POS tags by _s.
 * Blank lines will be ignored.
 *
 */
public class PosTagReader {
    public static List<List<String>> readPosTags(File posTagFile) throws IOException {
        List<List<String>> posTags = new ArrayList<List<String>>();
        BufferedReader reader = new BufferedReader(new FileReader(posTagFile));

        StringBuffer sb = new StringBuffer();
        String line;
        while ((line = reader.readLine()) != null) {
            line = line.trim();
            // Ignore blank lines and (()) lines (empty input)
            if (!line.isEmpty() && !line.equals("(())")) {
                // Words should be separated by spaces
                String[] tokens = line.split(" ");

                List<String> sentencePosTags = new ArrayList<String>();
                for (String token : tokens) {
                    int divider = token.indexOf("/");
                    if (divider == -1) {
                        // No divider found: presumably no POS tag
                        // Put a null in the list for this word
                        sentencePosTags.add(null);
                    } else {
                        sentencePosTags.add(token.substring(divider + 1));
                    }
                }

                posTags.add(sentencePosTags);
            }
        }

        reader.close();
        return posTags;

    }
}
