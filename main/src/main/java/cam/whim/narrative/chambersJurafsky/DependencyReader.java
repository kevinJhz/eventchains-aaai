package cam.whim.narrative.chambersJurafsky;

import com.google.common.base.Joiner;
import com.google.common.base.Splitter;
import com.google.common.io.Files;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Read dependencies from a file. Sentences should be separated by a blank line.
 * This assumes the dependency format output by C&C.
 *
 */
public class DependencyReader {
    public static List<DependencyGraph> getDependencyGraphs(File treeFile) throws IOException {
        // Read in dependency data
        List<String> depData = new ArrayList<String>();
        BufferedReader reader = new BufferedReader(new FileReader(treeFile));

        List<DependencyGraph> graphs = new ArrayList<DependencyGraph>();
        Map<Integer, String> words = new HashMap<Integer, String>();
        List<DependencyGraph.Dependency> dependencies = new ArrayList<DependencyGraph.Dependency>();
        int blankLines = 0;

        // Read in the full file
        // We don't use readline() here, because it won't recognise a blank line at the end of a file, important if
        //  there's an empty parse at the end
        String data = Files.toString(treeFile, Charset.defaultCharset());

        // Split lines using a splitter, not split(), so we don't lose multiple blank lines
        for (String line : Splitter.on('\n').split(data)) {
            line = line.trim();
            // There's a blank line between each sentence
            if (line.isEmpty()) {
                blankLines++;
                // A blank line after a dependency graph indicates it's ended
                // If we have 2 blank lines in a row, there's an empty dependency graph
                if (dependencies.size() > 0 || (blankLines % 2 == 0)) {
                    // End of a graph
                    graphs.add(new DependencyGraph(dependencies));
                    // Reset accumulators
                    words = new HashMap<Integer, String>();
                    dependencies = new ArrayList<DependencyGraph.Dependency>();
                }
            } else {
                blankLines = 0;

                // Normal dependencies look like:
                // (type word0_id0 word1_id1 [word2_id2])
                int bracketStart = line.indexOf('(');
                int bracketEnd = line.lastIndexOf(')');
                String[] tokens = line.substring(bracketStart + 1, bracketEnd).split(" ");
                String type = tokens[0];
                // Get each argument
                List<String> args = new ArrayList<String>();
                // Sometimes weird things happen and we have no arguments!
                if (tokens.length > 1)
                    args.add(tokens[1]);
                if (tokens.length > 2)
                    args.add(tokens[2]);
                if (tokens.length > 3)
                    args.add(tokens[3]);

                List<DependencyGraph.WordIndex> splitArgs = new ArrayList<DependencyGraph.WordIndex>();
                for (String arg : args)
                    splitArgs.add(splitLex(arg));

                dependencies.add(new DependencyGraph.Dependency(type, splitArgs));

                // Make sure these words are in the word map
                for (DependencyGraph.WordIndex arg : splitArgs)
                    if (arg != null)
                        words.put(arg.index, arg.word);
            }
        }
        if (!dependencies.isEmpty()) graphs.add(new DependencyGraph(dependencies));

        reader.close();

        return graphs;
    }

    public static DependencyGraph.WordIndex splitLex(String lex) {
        if (lex.equals("_"))
            return null;
        int split = lex.lastIndexOf('_');

        if (split == -1)
            // This is a non-word marker, e.g. poss or obj
            return new DependencyGraph.NonWord(lex);

        String word = lex.substring(0, split);
        int index = Integer.parseInt(lex.substring(split + 1));
        return new DependencyGraph.WordIndex(word, index);
    }
}
