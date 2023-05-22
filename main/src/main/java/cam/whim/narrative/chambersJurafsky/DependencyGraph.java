package cam.whim.narrative.chambersJurafsky;

import com.google.common.base.Joiner;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;

import java.util.*;

/**
 * Data structure for dependency graphs.
 */
public class DependencyGraph {
    public final Map<Integer, String> words;
    public final List<Dependency> dependencies;

    public DependencyGraph(List<Dependency> dependencies) {
        this.dependencies = dependencies;

        words = new HashMap<Integer, String>();
    }

    public DependencyGraph() {
        this(new ArrayList<Dependency>());
    }

    public String toString() {
        return Joiner.on('\n').join(dependencies);
    }

    /**
     * Find dependences of the given type whose C{argNum}th argument is word C{wordNum}.
     *
     * @param type       required dep type
     * @param argNum     arg num to check
     * @param wordNum    word num required
     * @return list of dependencies matching
     */
    public List<Dependency> findDependencies(String type, int argNum, int wordNum) {
        List<Dependency> found = new ArrayList<Dependency>();
        for (DependencyGraph.Dependency dep : dependencies) {
            if (dep.type.equals(type) && dep.args.length > argNum &&
                    dep.args[argNum].index == wordNum) {
                found.add(dep);
            }
        }
        return found;
    }

    /**
     * Find dependencies whose C{argNum}th argument is word C{wordNum}.
     *
     * @param argNum     arg num to check
     * @param wordNum    word num required
     * @return list of dependencies matching
     */
    public List<Dependency> findDependencies(int argNum, int wordNum) {
        List<Dependency> found = new ArrayList<Dependency>();
        for (DependencyGraph.Dependency dep : dependencies)
            if (dep.args.length > argNum && dep.args[argNum] != null && dep.args[argNum].index == wordNum)
                found.add(dep);
        return found;
    }

    public static class WordIndex {
        public final String word;
        public final int index;

        public WordIndex(String word, int index) {
            this.word = word;
            this.index = index;
        }

        public String toString() {
            return this.word + "_" + this.index;
        }
    }

    public static class NonWord extends WordIndex {
        public final String type;

        public NonWord(String type) {
            super(null, -1);
            this.type = type;
        }

        public String toString() {
            return type;
        }
    }

    public static class Dependency {
        public final String type;
        public final WordIndex[] args;

        public Dependency(String type, WordIndex[] args) {
            this.type = type;
            this.args = args;
        }

        public Dependency(String type, List<WordIndex> args) {
            this(type, args.toArray(new WordIndex[args.size()]));
        }

        public String toString() {
            List<String> argFormat = new ArrayList<String>();
            for (WordIndex w : args) {
                if (w == null)
                    argFormat.add("_");
                else
                    argFormat.add(w.toString());
            }
            return "(" + type + " " + Joiner.on(' ').join(argFormat) + ")";
        }
    }
}
