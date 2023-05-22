package cam.whim.narrative.chambersJurafsky;

import opennlp.tools.parser.Parse;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Load parse data from a file in PTB format with many parse trees in it.
 *
 */
public class ParseTreeReader {
    public static List<Parse> getParseTrees(File treeFile) throws IOException {
        // Read in parse data
        // Should be in PTB format
        List<String> parseData = new ArrayList<String>();
        BufferedReader reader = new BufferedReader(new FileReader(treeFile));

        StringBuffer sb = new StringBuffer();
        String line;
        while ((line = reader.readLine()) != null) {
            // These odd lines are empty parses, presumably a blank line in the input. Skip them
            if (line.equals("(())"))
                continue;
            // There's a blank line between each tree
            if (line.trim().isEmpty()) {
                // End of a parse tree
                parseData.add(sb.toString());
                sb = new StringBuffer();
            } else sb.append(line);
        }
        if (!sb.toString().isEmpty()) parseData.add(sb.toString());

        reader.close();

        List<Parse> parses = new ArrayList<Parse>();
        for (String parseStr : parseData)
            // ParseParse the parse data into a Parse parse...
            parses.add(Parse.parseParse(parseStr));

        return parses;
    }
}
