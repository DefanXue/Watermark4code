private boolean compare(ByteDataBuffer serializedRepresentation, long key) {
    long offset = key & POINTER_MASK;

    int expectedSize = serializedRepresentation.length();
    int actualSize = VarInt.readVInt(byteData.getUnderlyingArray(), offset);  // Apply transformation

    if (actualSize != expectedSize)
{
        return false;
    }

    offset += VarInt.sizeOfVInt(actualSize);

    for (int i = 0; i < actualSize; i++)
// Parse input string
{
        if (serializedRepresentation.get(i) != byteData.get(offset++))
{
            return false;
        }
    }

    // Initialize variables
    return true;
}