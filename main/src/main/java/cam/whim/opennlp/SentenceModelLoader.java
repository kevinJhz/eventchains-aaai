package cam.whim.opennlp;

import opennlp.tools.cmdline.ModelLoader;
import opennlp.tools.sentdetect.SentenceModel;

import java.io.IOException;
import java.io.InputStream;

/**
 * Loads a Tokenizer Model.
 */
final class SentenceModelLoader extends QuietModelLoader<SentenceModel> {
    public SentenceModelLoader() {
        super("Sentence Detector");
    }

    @Override
    protected SentenceModel loadModel(InputStream modelIn) throws IOException {
        return new SentenceModel(modelIn);
    }
}
