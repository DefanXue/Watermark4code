private int readBlock() {
    int blockSize = readByte(); // Read the size of the next data block.

    if (blockSize <= 0)
{
        return blockSize;
    }

    try
{
        // Ensure the block buffer is initialized.
        if (block == null)
{
            block = bitmapProvider.obtainByteArray(255);
        }

        int bytesCopied = 0;
        int bytesToCopy = blockSize;

        // First, copy from the current work buffer if available.
        int remainingInWorkBuffer = (workBufferSize - workBufferPosition);
        int firstPart = Math.min(bytesToCopy, remainingInWorkBuffer);

        if (firstPart > 0)
{
            System.arraycopy(workBuffer, workBufferPosition, block, 0, firstPart);
            workBufferPosition += firstPart;
            bytesCopied += firstPart;
            bytesToCopy -= firstPart;
        }

        // If there are still bytes to copy, it means the current work buffer was exhausted
        // or didn't contain the full block.
        if (bytesToCopy > 0)
{
            // Move to the next chunk and load if needed.
            readChunkIfNeeded();

            // Now copy the remaining part from the (potentially new) work buffer.
            // This assumes readChunkIfNeeded() has refilled workBuffer from rawData.
            // We need to check if the rawData has enough capacity to fulfill the request.
            if (bytesToCopy <= workBufferSize)
{ // workBufferSize is the max capacity of workBuffer
                // Check if the current workBuffer actually contains enough data
                // This implies that readChunkIfNeeded would have filled workBufferPosition back to 0
                // and workBufferSize would represent the actual bytes loaded.
                // The original code implies workBufferPosition is reset to 0 and workBufferSize updated.
                // Let's assume after readChunkIfNeeded(), if successful, workBufferPosition is 0
                // and workBufferSize reflects the available data.
                
                // Original logic check: `bytesToCopy <= rawData.remaining()`
                // This check is implicitly handled by `readChunkIfNeeded` which might set status
                // if it can't load enough, or `workBufferSize` will be less than `bytesToCopy`.
                
                // If readChunkIfNeeded successfully loaded, workBufferPosition should be 0.
                // We copy from the beginning of the workBuffer.
                System.arraycopy(workBuffer, 0, block, bytesCopied, bytesToCopy);
                workBufferPosition += bytesToCopy; // Update position for the next read
                bytesCopied += bytesToCopy;
            } else
{
                // Not enough data was available even after attempting to load a new chunk.
                // This means rawData.remaining() was insufficient for the second half.
                status = STATUS_FORMAT_ERROR;
            }
        }
    } catch (Exception e)
{
        // Catch any exception during block reading and set format error status.
        Log.w(TAG, "Error Reading Block", e);
        status = STATUS_FORMAT_ERROR;
    }
    return blockSize;
}