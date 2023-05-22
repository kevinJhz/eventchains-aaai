package cam.whim.narrative.chambersJurafsky;

import opennlp.tools.parser.Parse;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by mtw29 on 31/01/14.
 */
public class ParseTreeUtils {
    /**
     * Build a map from the character indices of the left edges of words to the (1-indexed) word number.
     *
     * @param parse
     * @return
     */
    public static Map<Integer, Integer> getLeftEdgeToWordMap(Parse parse) {
        return getLeftEdgeToWordMap(parse, 1);
    }

    private static Map<Integer, Integer> getLeftEdgeToWordMap(Parse parse, int baseIndex) {
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        if (parse.getChildren().length > 0) {
            // Internal node -- recurse
            int wordsDone = baseIndex;
            for (Parse child : parse.getChildren()) {
                Map<Integer, Integer> childMap = getLeftEdgeToWordMap(child, wordsDone);
                wordsDone += childMap.size();
                map.putAll(childMap);
            }
        } else {
            // Leaf
            map.put(parse.getSpan().getStart(), baseIndex);
        }
        return map;
    }

    /**
     * Build a map from the character indices of the right edges of words (exclusive) to the (1-indexed) word number.
     *
     * @param parse
     * @return
     */
    public static Map<Integer, Integer> getRightEdgeToWordMap(Parse parse) {
        return getRightEdgeToWordMap(parse, 1);
    }

    private static Map<Integer, Integer> getRightEdgeToWordMap(Parse parse, int baseIndex) {
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        if (parse.getChildren().length > 0) {
            // Internal node -- recurse
            int wordsDone = baseIndex;
            for (Parse child : parse.getChildren()) {
                Map<Integer, Integer> childMap = getRightEdgeToWordMap(child, wordsDone);
                wordsDone += childMap.size();
                map.putAll(childMap);
            }
        } else {
            // Leaf
            map.put(parse.getSpan().getEnd(), baseIndex);
        }
        return map;
    }
}
