package cam.whim.narrative.chambersJurafsky;

import java.io.File;
import java.io.IOException;

/**
 * Like ProtagonistExtractor, but extracts all possible entities, not just the most common one.
 *
 */
public class EntitiesExtractor {

    public static void main(String[] args) {
        // Check we've got enough args
        if (args.length < 4) {
            System.err.println("Usage: EntitiesExtractor <models-dir> <pos-tag-file> <parse-tree-file> <dependency-parse-file>");
            System.exit(1);
        }

        String modelsDirName = args[0];
        String posTagFileName = args[1];
        String parseTreeFileName = args[2];
        String dependencyFileName = args[3];

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

        // TODO Fix this
        /*
        try {
            // Load parse tree data
            List<ParsedSentence> sentences = ParsedSentence.readFiles(new File(parseTreeFileName), new File(dependencyFileName));

            // Coref resolution
            // TODO Fix this
            DiscourseEntity[] entities = coref.resolveCoreference(sentences);
            for (DiscourseEntity entity : entities) {
                // Pull all of the mention texts into the string name
                List<String> nameParts = new ArrayList<String>();
                Iterator<MentionContext> mentions = entity.getMentions();
                while (mentions.hasNext())
                    nameParts.add(mentions.next().toString().trim().replace('"', '\''));
                String entityName = Joiner.on('/').join(nameParts);

                System.out.println("Entity=\"" + entityName + "\", mentions=" + entity.getNumMentions());
                // Look for verbal dependencies involving this entity
                List<ProtagonistExtractor.Event> events =
                        coref.extractVerbalDependencies(entity, parses, dependencyGraphs, lemmas);

                // Output the events
                for (ProtagonistExtractor.Event event : events) {
                    System.out.println(Joiner.on(',').join(event.getCsvColumns()));
                }
                System.out.println();
            }
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
