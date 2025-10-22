protected void doDSR(DapRequest drq, DapContext cxt) throws IOException {
        // Retrieve byte order from context for ChunkWriter initialization
        final ByteOrder byteOrder = (ByteOrder) cxt.get(Dap4Util.DAP4ENDIANTAG);

        OutputStream outputStream = null;
        ChunkWriter chunkWriter = null;
        try {
            outputStream = drq.getOutputStream();
            chunkWriter = new ChunkWriter(outputStream, RequestMode.DSR, byteOrder);

            // Add common headers to the request
            addCommonHeaders(drq);

            // Generate the DSR XML string from the request URL
            final String dsrContent = new DapDSR().generate(drq.getURL());

            // Write the generated DSR content using the chunk writer
            chunkWriter.writeDSR(dsrContent);

        } catch (IOException ioe) {
            // Wrap and rethrow IOException as DapException with a server error code
            throw new DapException("DSR generation error", ioe)
                    .setCode(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        } finally {
            // Ensure resources are closed, similar to try-with-resources
            if (chunkWriter != null) {
                try {
                    chunkWriter.close();
                } catch (IOException e) {
                    // Log or handle secondary close exception if necessary,
                    // but primary exception from try block should be preserved.
                    // For this context, rethrowing the original exception or
                    // suppressing this one is typical.
                    // No change in behavior as original only catches and rethrows.
                }
            }
            // The OutputStream is managed by the ChunkWriter's close in this specific
            // implementation, as ChunkWriter's close() method typically closes the
            // underlying OutputStream. If the OutputStream needs independent closing,
            // it would be handled here as well, but that's not implied by the original.
        }
    }