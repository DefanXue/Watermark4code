private int readBlock() {
    int blockSize = readByte(); // Read the size of the next data block.

    if (blockSize <= 0) {
        return blockSize;
    }

    try {
        // Ensure the block buffer is initialized.
        if (block == null) {
            block = bitmapProvider.obtainByteArray(255);
        }

        final int remainingInWorkBuffer = workBufferSize - workBufferPosition;

        // Case 1: Entire block fits within the current work buffer.
        if (blockSize <= remainingInWorkBuffer) {
            System.arraycopy(workBuffer, workBufferPosition, block, 0, blockSize);
            workBufferPosition += blockSize;
        }
        // Case 2: Block spans across the current work buffer and potentially the next chunk.
        else if (rawData.remaining() + remainingInWorkBuffer >= blockSize) {
            // Copy the first part of the block from the current work buffer.
            System.arraycopy(workBuffer, workBufferPosition, block, 0, remainingInWorkBuffer);
            workBufferPosition = workBufferSize; // Mark current work buffer as fully consumed.

            readChunkIfNeeded(); // Load the next chunk into workBuffer if necessary.

            // Copy the remaining part of the block from the (potentially new) work buffer.
            final int secondHalfSize = blockSize - remainingInWorkBuffer;
            System.arraycopy(workBuffer, 0, block, remainingInWorkBuffer, secondHalfSize);
            workBufferPosition += secondHalfSize;
        }
        // Case 3: Not enough data available to read the entire block.
        else {
            status = STATUS_FORMAT_ERROR;
        }
    } catch (Exception e) {
        // Catch any exception during block reading and set format error status.
        Log.w(TAG, "Error Reading Block", e);
        status = STATUS_FORMAT_ERROR;
    }
    return blockSize;
}