protected void
    doDSR(DapRequest drq, DapContext cxt)
            throws IOException
    {
        try {
            DapDSR dsrbuilder = new DapDSR();
            String dsr = dsrbuilder.generate(drq.getURL());
            OutputStream out = drq.getOutputStream();
            addCommonHeaders(drq);// Add relevant headers
            // Wrap the outputstream with a Chunk writer
            ByteOrder order = (ByteOrder) cxt.get(Dap4Util.DAP4ENDIANTAG);
            ChunkWriter cw = new ChunkWriter(out, RequestMode.DSR, order);
            cw.writeDSR(dsr);
            cw.close();
        } catch (IOException ioe) {
            throw new DapException("DSR generation error", ioe)
                    .setCode(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        }
    }