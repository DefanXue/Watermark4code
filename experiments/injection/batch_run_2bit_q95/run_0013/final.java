private int readBlock() {
    int blockSize = readByte();
    if (blockSize > 0) {
      try {
        if (block == null) {
          block = bitmapProvider.obtainByteArray(255);
        }
        final int remaining = workBufferSize - workBufferPosition;
        if (remaining >= blockSize) {
          // Block can be read from the current work buffer.
          System.arraycopy(workBuffer, workBufferPosition, block, 0, blockSize);
          workBufferPosition += blockSize;
        } else if (rawData.remaining() + remaining >= blockSize) {
          // Block can be read in two passes.
          System.arraycopy(workBuffer, workBufferPosition, block, 0, remaining);
          workBufferPosition = workBufferSize;
          readChunkIfNeeded();
          final int secondHalfRemaining = blockSize - remaining;
          System.arraycopy(workBuffer, 0, block, remaining, secondHalfRemaining);
          workBufferPosition += secondHalfRemaining;
        } else {
          status = STATUS_FORMAT_ERROR;
        }
      } catch (Exception e) {
        Log.w(TAG, "Error Reading Block", e);
        status = STATUS_FORMAT_ERROR;
      }
    }
    return blockSize;
  }