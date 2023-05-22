package cam.whim.coreference.simple;

import com.google.common.base.Joiner;
import com.sun.org.apache.bcel.internal.generic.NEW;
import opennlp.tools.coref.sim.GenderEnum;
import opennlp.tools.coref.sim.NumberEnum;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Like DiscourseEntity, but simpler, just storing the information we need after coref resolution is
 * finished.
 *
 */
public class DiscourseEntity {
    public final String category;
    public final GenderEnum gender;
    public final double genderProb;
    public final NumberEnum number;
    public final double numberProb;
    public final List<MentionContext> mentions;

    public DiscourseEntity(String category, GenderEnum gender, double genderProb, NumberEnum number, double numberProb, List<MentionContext> mentions) {
        this.category = category;
        this.gender = gender;
        this.genderProb = genderProb;
        this.number = number;
        this.numberProb = numberProb;
        this.mentions = mentions;
    }

    public static DiscourseEntity fromDiscourseEntity(opennlp.tools.coref.DiscourseEntity entity) {
        // Get the simplified representation of all the mentions
        List<MentionContext> mentions = new ArrayList<MentionContext>();
        for (Iterator<opennlp.tools.coref.mention.MentionContext> inMentions = entity.getMentions(); inMentions.hasNext();)
            mentions.add(MentionContext.fromMentionContext(inMentions.next()));

        return new DiscourseEntity(entity.getCategory(),
                entity.getGender(),
                entity.getGenderProbability(),
                entity.getNumber(),
                entity.getNumberProbability(),
                mentions);
    }

    public String toString() {
        StringBuffer sb = new StringBuffer();

        sb.append("category=");
        sb.append(category);
        sb.append(" // ");

        sb.append("gender=");
        sb.append(gender.toString());
        sb.append(" // ");

        sb.append("genderProb=" + genderProb);
        sb.append(" // ");

        sb.append("number=");
        sb.append(number.toString());
        sb.append(" // ");

        sb.append("numberProb=" + numberProb);
        sb.append(" // ");

        sb.append("mentions=");
        List<String> mentionStrings = new ArrayList<String>();
        for (MentionContext mention : mentions)
            mentionStrings.add(mention.toString());
        sb.append(Joiner.on(" / ").join(mentionStrings));

        return sb.toString();
    }

    /**
     * Inverse of toString().
     *
     * @param entityString
     * @return
     */
    public static DiscourseEntity fromString(String entityString) throws StringFormatError {
        String[] parts = entityString.split(" // ");
        if (parts.length < 6)
            throw new StringFormatError("not enough information in entity string: " + entityString);

        // Extract category
        if (!parts[0].startsWith("category="))
            throw new StringFormatError("category not found in entity: " + entityString);
        String category = parts[0].substring(9);

        // Extract gender
        if (!parts[1].startsWith("gender="))
            throw new StringFormatError("gender not found in entity: " + entityString);
        String genderString = parts[1].substring(7);
        GenderEnum gender = GenderEnum.UNKNOWN;
        if (genderString.equals("male"))
            gender = GenderEnum.MALE;
        else if (genderString.equals("female"))
            gender = GenderEnum.FEMALE;
        else if (genderString.equals("neuter"))
            gender = GenderEnum.NEUTER;

        // Extract gender probability
        if (!parts[2].startsWith("genderProb="))
            throw new StringFormatError("gender prob not found in entity: " + entityString);
        double genderProb = Double.parseDouble(parts[2].substring(11));

        // Extract number
        if (!parts[3].startsWith("number="))
            throw new StringFormatError("number not found in entity: " + entityString);
        String numberString = parts[3].substring(7);
        NumberEnum number = NumberEnum.UNKNOWN;
        if (numberString.equals("singular"))
            number = NumberEnum.SINGULAR;
        else if (numberString.equals("plural"))
            number = NumberEnum.PLURAL;

        // Extract number prob
        if (!parts[4].startsWith("numberProb="))
            throw new StringFormatError("number prob not found in entity: " + entityString);
        double numberProb = Double.parseDouble(parts[4].substring(11));

        // Extract all the mentions
        if (!parts[5].startsWith("mentions="))
            throw new StringFormatError("mentions not found in entity: " + entityString);
        String[] mentionStrings = parts[5].substring(9).split(" / ");
        List<MentionContext> mentions = new ArrayList<MentionContext>();
        for (String mentionString : mentionStrings) {
            try {
                mentions.add(MentionContext.fromString(mentionString));
            } catch (MentionContext.StringFormatError stringFormatError) {
                throw new StringFormatError("error in mention formatting: " + stringFormatError.getMessage());
            }
        }

        return new DiscourseEntity(category, gender, genderProb, number, numberProb, mentions);
    }

    /**
     * Convert the DiscourseEntitys from a document's coreference resolution to the simple representation.
     *
     * @param documentEntities
     * @return
     */
    public static List<DiscourseEntity> fromDocumentEntities(opennlp.tools.coref.DiscourseEntity[] documentEntities) {
        List<DiscourseEntity> entities = new ArrayList<DiscourseEntity>();
        // Convert each entity to the simple format
        for (opennlp.tools.coref.DiscourseEntity documentEntity : documentEntities)
            entities.add(DiscourseEntity.fromDiscourseEntity(documentEntity));
        return entities;
    }

    public static List<DiscourseEntity> fromFile(File file) throws IOException, StringFormatError {
        List<DiscourseEntity> entities = new ArrayList<DiscourseEntity>();
        BufferedReader reader = new BufferedReader(new FileReader(file));

        try {
            String line;
            while ((line = reader.readLine()) != null) {
                // Read one entity from each line of the file
                entities.add(DiscourseEntity.fromString(line));
            }
        } finally {
            reader.close();
        }

        return entities;
    }

    public static class StringFormatError extends Exception {
        public StringFormatError(String message) {
            super(message);
        }
    }
}
