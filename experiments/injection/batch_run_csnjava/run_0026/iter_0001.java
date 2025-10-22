private boolean compare(ByteDataBuffer serializedRepresentation, long key) {
    long offset = key & POINTER_MASK;

    int expectedSize = serializedRepresentation.length();
    int actualSize = VarInt.readVInt(byteData.getUnderlyingArray(), offset);

    if (actualSize != expectedSize) {
        return false;
    }

    offset += VarInt.sizeOfVInt(actualSize);

    for (int i = 0; i < actualSize; i++) {
        if (serializedRepresentation.get(i) != byteData.get(offset++)) {
            return false;
        }
    }

    return true;
}