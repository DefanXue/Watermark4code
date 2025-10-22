protected Object
    readAtomic(List<Slice> slices)
            throws DapException
    {
        if(slices == null)
            throw new DapException("DataCursor.read: null set of slices");
        assert (this.scheme == scheme.ATOMIC);
        DapVariable atomvar = (DapVariable) getTemplate();
        int rank = atomvar.getRank();
        assert slices != null && ((rank == 0 && slices.size() == 1) || (slices.size() == rank));
        // Get VarNotes and TypeNotes
        Notes n = ((Nc4DSP) this.dsp).find(this.template);
        Object result = null;
        long count = DapUtil.sliceProduct(slices);
        VarNotes vn = (VarNotes) n;
        TypeNotes ti = vn.getBaseType();
        if(getContainer() == null) {
            if(rank == 0) { //scalar
                result = readAtomicScalar(vn, ti);
            } else {
                result = readAtomicVector(vn, ti, count, slices);
            }
        } else {// field of a structure instance or record
            long elemsize = ((DapType) ti.get()).getSize();
            assert (this.container != null);
            long trueoffset = computeTrueOffset(this);
            Nc4Pointer varmem = getMemory();
            Nc4Pointer mem = varmem.share(trueoffset, count * elemsize);
            result = getatomicdata(ti.getType(), count, elemsize, mem);
        }
        return result;
    }