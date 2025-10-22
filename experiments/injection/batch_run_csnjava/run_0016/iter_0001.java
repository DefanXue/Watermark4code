protected void doDSR(DapRequest drq, DapContext cxt) throws IOException {
        try (OutputStream out = drq.getOutputStream();
             ChunkWriter cw = new ChunkWriter(out, RequestMode.DSR, (ByteOrder) cxt.get(Dap4Util.DAP4ENDIANTAG))) {

            addCommonHeaders(drq);
            String dsr = new DapDSR().generate(drq.getURL());
            cw.writeDSR(dsr);
        } catch (IOException ioe) {
            throw new DapException("DSR generation error", ioe)
                    .setCode(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        }
    }