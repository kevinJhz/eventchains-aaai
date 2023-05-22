package cam.whim.opennlp;

import opennlp.tools.cmdline.ModelLoader;
import opennlp.tools.tokenize.TokenizerModel;

import java.io.IOException;
import java.io.InputStream;

/**
 * Loads a Tokenizer Model quietly.
 */
public final class TokenizerModelLoader extends QuietModelLoader<TokenizerModel> {
    public TokenizerModelLoader() {
        super("Tokenizer");
    }

    @Override
    protected TokenizerModel loadModel(InputStream modelIn) throws IOException {
        return new TokenizerModel(modelIn);
    }
}
