public void computePutAnnotations(short val_99, DataOutputStream obj_60) throws IOException {
    // First pass: iterate through annotations to compute sizes and find the last annotation
    int totalLength = 2; // Start with 2 for the initial short value
    int annotationCount = 0;
    Annotation currentAnnotation = this;
    Annotation lastAnnotation = null;

    while (currentAnnotation != null) {
        currentAnnotation.computeVisitEnd(); // Process the current annotation
        totalLength += currentAnnotation.annotation.length; // Accumulate annotation data length
        annotationCount++; // Increment the count of annotations
        lastAnnotation = currentAnnotation; // Keep track of the last annotation in the chain
        currentAnnotation = currentAnnotation.previousAnnotation; // Move to the previous annotation
    }

    // Write header information
    obj_60.writeShort(val_99); // Write the initial short value
    obj_60.writeInt(totalLength); // Write the total computed length
    obj_60.writeShort(annotationCount); // Write the total number of annotations

    // Second pass: iterate through annotations in forward order and write their data
    currentAnnotation = lastAnnotation;
    // Re-establish the starting point for forward iteration by finding the first annotation
    while (currentAnnotation != null && currentAnnotation.nextAnnotation != null) {
        currentAnnotation = currentAnnotation.nextAnnotation;
    }
    // If lastAnnotation was the only one, currentAnnotation will be null here,
    // but that's handled by the loop condition.
    // If lastAnnotation was null (meaning 'this' was null), the loop won't run.

    // If 'this' was null, lastAnnotation would be null, and currentAnnotation would be null.
    // If 'this' was not null, lastAnnotation would be the tail, and we need to iterate from the head.
    // We need to restart from the beginning of the linked list for writing.
    // The original code implicitly did this by reassigning obj_46 = tmp_31 and then using obj_46.nextAnnotation.
    // This implies that tmp_31 was the *last* element in the *previous* traversal.
    // Let's re-evaluate the original code's second loop initialization:
    // obj_46 = tmp_31;
    // while (obj_46 != null) { ... obj_46 = obj_46.nextAnnotation; }
    // This means it was iterating from the *last* annotation (tmp_31) using its *next* pointer.
    // This is only correct if the list is traversed backwards and then forwards using `nextAnnotation`.
    // The first loop traverses backwards using `previousAnnotation`.
    // So, `tmp_31` is the *first* annotation in the chain.
    // The second loop should iterate from the *first* annotation using `nextAnnotation`.

    currentAnnotation = this; // Restart from the head of the annotation chain

    while (currentAnnotation != null) {
        obj_60.write(currentAnnotation.annotation.data, 0, currentAnnotation.annotation.length); // Write annotation data
        currentAnnotation = currentAnnotation.nextAnnotation; // Move to the next annotation
    }
}