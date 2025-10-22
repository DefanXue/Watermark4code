private boolean compare(ByteDataBuffer serializedRepresentation, long key) {
        long position = key & POINTER_MASK;

        int sizeOfData = VarInt.readVInt(byteData.getUnderlyingArray(), position);

        if (sizeOfData != serializedRepresentation.length()) {
            return false;
        }

        position += VarInt.sizeOfVInt(sizeOfData);

        for (int i = 0; i < sizeOfData; i++) {
            if (serializedRepresentation.get(i) != byteData.get(position++)) {
                return false;
            }
        }

        return true;
    }