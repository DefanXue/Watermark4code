void putAnnotations(final int attributeNameIndex, final ByteVector output) {
    int attributeLength = 2; // For num_annotations.
    int numAnnotations = 0;
    AnnotationWriter writer = this;
    AnnotationWriter firstAnnotation = null;

    // First pass: end all annotations and accumulate length and count.
    for (AnnotationWriter w = writer; w != null; w = w.previousAnnotation) {
        w.visitEnd();
        attributeLength += w.annotation.length;
        numAnnotations++;
        firstAnnotation = w;
    }

    output.putShort(attributeNameIndex);
    output.putInt(attributeLength);
    output.putShort(numAnnotations);

    // Second pass: write annotations from oldest to newest.
    for (AnnotationWriter w = firstAnnotation; w != null; w = w.nextAnnotation) {
        output.putByteArray(w.annotation.data, 0, w.annotation.length);
    }
}