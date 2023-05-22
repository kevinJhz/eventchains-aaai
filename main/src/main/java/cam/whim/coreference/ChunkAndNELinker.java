package cam.whim.coreference;

import opennlp.tools.coref.DefaultLinker;
import opennlp.tools.coref.LinkerMode;

import java.io.IOException;

/**
 * Have to override this too just to set the overridden mention finder.
 */
public class ChunkAndNELinker extends DefaultLinker {
    public ChunkAndNELinker(String modelDirectory, LinkerMode mode) throws IOException {
        this(modelDirectory, mode, true);
    }

    public ChunkAndNELinker(String modelDirectory, LinkerMode mode, boolean useDiscourseModel) throws IOException {
        this(modelDirectory, mode, useDiscourseModel, -1);
    }

    public ChunkAndNELinker(String modelDirectory, LinkerMode mode, boolean useDiscourseModel, double fixedNonReferentialProbability) throws IOException {
        super(modelDirectory, mode, useDiscourseModel, fixedNonReferentialProbability);
        removeUnresolvedMentions = false;
    }

    /**
     * This is the bit we override to use the modified mention finder.
     */
    protected void initMentionFinder() {
        mentionFinder = ChunkAndNEMentionFinder.getInstance(headFinder);
    }
}
