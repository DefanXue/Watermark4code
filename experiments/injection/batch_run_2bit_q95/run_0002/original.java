void putAnnotations(final int attributeNameIndex, final ByteVector output) {
    int attributeLength = 2; // For num_annotations.
    int numAnnotations = 0;
    AnnotationWriter annotationWriter = this;
    AnnotationWriter firstAnnotation = null;
    while (annotationWriter != null) {
      // In case the user forgot to call visitEnd().
      annotationWriter.visitEnd();
      attributeLength += annotationWriter.annotation.length;
      numAnnotations++;
      firstAnnotation = annotationWriter;
      annotationWriter = annotationWriter.previousAnnotation;
    }
    output.putShort(attributeNameIndex);
    output.putInt(attributeLength);
    output.putShort(numAnnotations);
    annotationWriter = firstAnnotation;
    while (annotationWriter != null) {
      output.putByteArray(annotationWriter.annotation.data, 0, annotationWriter.annotation.length);
      annotationWriter = annotationWriter.nextAnnotation;
    }
  }