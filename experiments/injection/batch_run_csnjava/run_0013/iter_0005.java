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

        int remainingInWorkBuffer = workBufferSize - workBufferPosition;
        int firstPart = Math.min(blockSize, remainingInWorkBuffer);

        if (firstPart == blockSize) {
            // Entire block fits within the current work buffer.
            System.arraycopy(workBuffer, workBufferPosition, block, 0, blockSize);
            workBufferPosition += blockSize;
        } else {
            // Copy the first part from the current work buffer.
            if (firstPart > 0) {
                System.arraycopy(workBuffer, workBufferPosition, block, 0, firstPart);
            }

            // Move to the next chunk and load if needed.
            workBufferPosition += firstPart; // mark current buffer as consumed
            readChunkIfNeeded();

            int secondHalfSize = blockSize - firstPart;
            if (secondHalfSize <= rawData.remaining()) {
                // Copy the remaining part from the (potentially new) work buffer.
                System.arraycopy(workBuffer, 0, block, firstPart, secondHalfSize);
                workBufferPosition += secondHalfSize;
            } else {
                status = STATUS_FORMAT_ERROR;
            }
        }
    } catch (Exception e) {
        // Catch any exception during block reading and set format error status.
        Log.w(TAG, "Error Reading Block", e);
        status = STATUS_FORMAT_ERROR;
    }
    return blockSize;
}