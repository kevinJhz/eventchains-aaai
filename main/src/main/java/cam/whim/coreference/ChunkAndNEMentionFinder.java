package cam.whim.coreference;

import opennlp.tools.coref.Linker;
import opennlp.tools.coref.mention.*;
import opennlp.tools.coref.resolver.ResolverUtils;
import opennlp.tools.util.Span;

import java.util.*;

/**
 * Stupid behaviour in the AbstractMentionFinder means I need to subclass it here.
 */
public class ChunkAndNEMentionFinder extends AbstractMentionFinder {
    private static ChunkAndNEMentionFinder instance;

    private ChunkAndNEMentionFinder(HeadFinder hf) {
        headFinder = hf;
        // For our chunked data it makes no sense to try breaking down compound NPs
        collectPrenominalNamedEntities = false;
        collectCoordinatedNounPhrases = false;
    }

    /**
     * Retrieves the one and only existing instance.
     *
     * @param hf
     * @return one and only existing instance
     */
    public static ChunkAndNEMentionFinder getInstance(HeadFinder hf) {
        if (instance == null) {
            instance = new ChunkAndNEMentionFinder(hf);
        }
        else if (instance.headFinder != hf) {
            instance = new ChunkAndNEMentionFinder(hf);
        }
        return instance;
    }

    public Mention[] getMentions(Parse p, Map<String, List<Span>> namedEntities) {
        List<Parse> nps = p.getNounPhrases();
        Collections.sort(nps);

        // Turn around the NE map
        Map<Span, String> neMap = new HashMap<Span, String>();
        for (String neType : namedEntities.keySet()) {
            for (Span span : namedEntities.get(neType)) {
                neMap.put(span, neType);
            }
        }

        Mention[] mentions = collectMentions(nps, neMap);
        return mentions;
    }

    private Mention[] collectMentions(List<Parse> nps, Map<Span, String> neMap) {
        // Largely follow what AbstractMentionFinder does
        List<Mention> mentions = new ArrayList<Mention>(nps.size());
        Set<Parse> recentMentions = new HashSet<Parse>();
        for (Parse np : nps) {
            // This is a shallow parse, so we don't need to check for embeddings of mentions
            clearMentions(recentMentions, np);
            Parse head = headFinder.getLastHead(np);

            // We need to set the NE type when creating a Mention (idiotic!)
            String neType = null;
            if (neMap.containsKey(np.getSpan())) {
                // This is a named entity
                neType = neMap.get(np.getSpan());
            }

            Mention extent = new Mention(np.getSpan(), head.getSpan(), head.getEntityId(), np, null, neType);
            System.out.println("Adding extent: " + extent);
            mentions.add(extent);
            recentMentions.add(np);
        }
        Collections.sort(mentions);
        removeDuplicates(mentions);
        return mentions.toArray(new Mention[mentions.size()]);
    }

    private void clearMentions(Set<Parse> mentions, Parse np) {
        Span npSpan =np.getSpan();
        for(Iterator<Parse> mi=mentions.iterator();mi.hasNext();) {
            Parse mention = mi.next();
            if (!mention.getSpan().contains(npSpan)) {
                //System.err.println("clearing "+mention+" for "+np);
                mi.remove();
            }
        }
    }


    private void removeDuplicates(List<Mention> extents) {
        Mention lastExtent = null;
        for (Iterator<Mention> ei = extents.iterator(); ei.hasNext();) {
            Mention e = ei.next();
            if (lastExtent != null && e.getSpan().equals(lastExtent.getSpan())) {
                ei.remove();
            }
            else {
                lastExtent = e;
            }
        }
    }
}