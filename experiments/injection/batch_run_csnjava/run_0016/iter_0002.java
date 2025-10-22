protected void doDSR(DapRequest drq, DapContext cxt) throws IOException {
        // Retrieve byte order from context for ChunkWriter initialization
        final ByteOrder byteOrder = (ByteOrder) cxt.get(Dap4Util.DAP4ENDIANTAG);

        try (OutputStream outputStream = drq.getOutputStream();
             ChunkWriter chunkWriter = new ChunkWriter(outputStream, RequestMode.DSR, byteOrder)) {

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
        }
    }