void putAnnotations(final int attributeNameIndex, final ByteVector output) {
    int totalLength = 2; // For num_annotations.
    int count = 0;
    AnnotationWriter oldest = null;

    for (AnnotationWriter w = this; w != null; w = w.previousAnnotation) {
        w.visitEnd();
        totalLength += w.annotation.length;
        count++;
        oldest = w;
    }

    output.putShort(attributeNameIndex);
    output.putInt(totalLength);
    output.putShort(count);

    for (AnnotationWriter w = oldest; w != null; w = w.nextAnnotation) {
        output.putByteArray(w.annotation.data, 0, w.annotation.length);
    }
}