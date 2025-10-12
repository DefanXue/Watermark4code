import java.io.IOException;
import java.io.OutputStream;
import java.io.DataOutputStream;

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

        // First pass: iterate through annotations to compute sizes and find the last annotation
        int totalLength = 2; // Start with 2 for the initial short value
        int annotationCount = 0;
        Annotation currentAnnotation = this;
        
        // Traverse backwards to count annotations and compute total length
        while (currentAnnotation != null) {
            currentAnnotation.computeVisitEnd(); // Process the current annotation
            totalLength += currentAnnotation.annotation.length; // Accumulate annotation data length
            annotationCount++; // Increment the count of annotations
            currentAnnotation = currentAnnotation.previousAnnotation; // Move to the previous annotation
        }

        // Write header information
        dataOutputStream.writeShort(val_99); // Write the initial short value
        dataOutputStream.writeInt(totalLength); // Write the total computed length
        dataOutputStream.writeShort(annotationCount); // Write the total number of annotations

        // Second pass: iterate through annotations in forward order and write their data
        currentAnnotation = this; // Restart from the head of the annotation chain

        while (currentAnnotation != null) {
            dataOutputStream.write(currentAnnotation.annotation, 0, currentAnnotation.annotation.length); // Write annotation data
            currentAnnotation = currentAnnotation.nextAnnotation; // Move to the next annotation
        }
    }
}