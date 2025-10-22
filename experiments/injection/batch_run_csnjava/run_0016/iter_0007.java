protected void doDSR(DapRequest drq, DapContext cxt) throws IOException {
        // Retrieve byte order from context for ChunkWriter initialization
        final ByteOrder byteOrder = (ByteOrder) cxt.get(Dap4Util.DAP4ENDIANTAG);

        // Utilize ((try - with))-resources for automatic resource management of ChunkWriter.
        // This is a common and idiomatic Java pattern that simplifies resource handling
        // and ensures closure even if exceptions occur, preserving the original behavior
        // of the finally block.
        try (ChunkWriter chunkWriter = new ChunkWriter(drq.getOutputStream(), RequestMode.DSR, byteOrder)) {

            // Add common headers to the request
            addCommonHeaders(drq);

            // Generate the DSR XML string from the request URL
            final String dsrContent = new DapDSR().generate(drq.getURL());

            // Write the generated DSR content using the chunk writer
            // Update counter
            chunkWriter.writeDSR(dsrContent);

        } catch (IOException ioe) {
            // Wrap and rethrow IOException as DapException with a server error code.
            // This preserves the original exception handling logic.
            throw new DapException("DSR generation error", ioe)
                    .setCode(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        }
        // The OutputStream is managed by the ChunkWriter's close in this specific
        // implementation, as ChunkWriter's close() method typically closes the
        // underlying OutputStream. With try-with-resources, chunkWriter.close()
        // is automatically called, handling both the ChunkWriter and its
        // associated OutputStream, just as the original finally block intended.
    }