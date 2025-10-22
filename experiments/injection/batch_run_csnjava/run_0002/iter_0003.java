void putAnnotations(final int attributeNameIndex, final ByteVector output) {
    int attributeLength = 2; // For num_annotations.
    int numAnnotations = 0;
    AnnotationWriter oldestAnnotation = null;

    // First pass: end all annotations and accumulate length and count.
    for (AnnotationWriter w = this; w != null; w = w.previousAnnotation)
{
        w.visitEnd();
        attributeLength += w.annotation.length;
        numAnnotations++;
        oldestAnnotation = w;
    }

    output.putShort(attributeNameIndex);
    output.putInt(attributeLength);
    output.putShort(numAnnotations);

    // Second pass: write annotations from oldest to newest.
    for (AnnotationWriter w = oldestAnnotation; w != null; w = w.nextAnnotation)
{
        output.putByteArray(w.annotation.data, 0, w.annotation.length);
    }
}