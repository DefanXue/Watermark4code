protected void doDSR(DapRequest drq, DapContext cxt) {
    try {
        DapDSR dsrbuilder = new DapDSR();
        String dsr = dsrbuilder.generate(drq.getURL());
        OutputStream outputStream = drq.getOutputStream();
        addCommonHeaders(drq);
        ByteOrder order = (ByteOrder) cxt.get(Dap4Util.DAP4ENDIANTAG);
        ChunkWriter cw = new ChunkWriter(outputStream, RequestMode.DSR, order);
        cw.writeDSR(dsr);
        cw.close();
    } catch (IOException ioe) {
        throw new DapException("DSR generation error", ioe).setCode(500);
    }
}