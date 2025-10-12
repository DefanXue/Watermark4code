import java.io.IOException;
import java.io.OutputStream;
import java.io.DataOutputStream;
import java.util.ArrayList;
import java.util.List;

public class Annotation {
    public byte[] annotation;
    public Annotation previousAnnotation;
    public Annotation nextAnnotation;

    public void computeVisitEnd() {
        // This method's implementation is not provided in the Java snippet.
        // Its behavior must be mirrored exactly in C#.
        // For translation purposes, we assume it modifies 'this' or related state.
        // If it throws exceptions, those must be propagated.
    }

    public void computePutAnnotations(short val_99, OutputStream obj_60) throws IOException {
        DataOutputStream dataOutputStream = new DataOutputStream(obj_60);

        List<Annotation> annotations = new ArrayList<>();
        int totalLength = 2; // Start with 2 for the initial short value
        Annotation currentAnnotation = this;

        // Traverse forwards to collect annotations and compute total length
        while (currentAnnotation != null) {
            currentAnnotation.computeVisitEnd(); // Process the current annotation
            totalLength += currentAnnotation.annotation.length; // Accumulate annotation data length
            annotations.add(currentAnnotation); // Add to list for forward iteration
            currentAnnotation = currentAnnotation.nextAnnotation; // Move to the next annotation
        }

        // Write header information
        dataOutputStream.writeShort(val_99); // Write the initial short value
        dataOutputStream.writeInt(totalLength); // Write the total computed length
        dataOutputStream.writeShort(annotations.size()); // Write the total number of annotations

        // Write annotation data in forward order
        for (Annotation annotation : annotations) {
            dataOutputStream.write(annotation.annotation, 0, annotation.annotation.length); // Write annotation data
        }
    }
}