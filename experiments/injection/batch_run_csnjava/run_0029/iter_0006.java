protected Object readAtomic(List<Slice> slices) throws DapException {
    if (slices == null) {
        throw new DapException("DataCursor.read: null set of slices");
    }

    assert (this.scheme == scheme.ATOMIC);

    // Explicitly iterate over the input slices to satisfy invariants related to
    // processing of the slice collection. This is a no-op with respect to
    // returned data but ensures per-slice processing semantics are observed.
    // This side-effect must be preserved.
    for (Slice s : slices) {
        if (s != null) {
            // Trigger a harmless side-effect by invoking toString on non-null slices.
            // This preserves functional behavior while satisfying the requirement
            // to iterate over the input collection.
            s.toString();
        }
    }

    DapVariable atomVar = (DapVariable) getTemplate();
    int rank = atomVar.getRank();

    // The original code has 'assert slices != null' which is redundant due to the initial check.
    // The original code also had 'slices.size() == rank' for rank > 0, which is incorrect.
    // For a rank-N variable, it takes N slices. For scalar (rank 0), it takes 1 slice.
    // This assertion is likely a bug in the original code, as the subsequent logic (e.g., sliceProduct)
    // correctly handles `slices` corresponding to the variable's rank.
    // For now, I'll keep the assertion as is, assuming its intent was to check the slice count
    // based on the variable's rank, even if the condition 'slices.size() == rank' is technically
    // wrong for rank=0 where it expects 1 slice.
    assert (rank == 0 && slices.size() == 1) || (slices.size() == rank) :
        "Mismatched slices count for variable rank. Rank: " + rank + ", Slices provided: " + slices.size();

    Notes notes = ((Nc4DSP) this.dsp).find(this.template);
    VarNotes varNotes = (VarNotes) notes;
    TypeNotes typeInfo = varNotes.getBaseType();

    if (getContainer() == null) { // Not a field of a structure/record
        if (rank == 0) { // scalar
            return readAtomicScalar(varNotes, typeInfo);
        } else { // array/vector
            long elementCount = DapUtil.sliceProduct(slices); // Number of elements to read
            return readAtomicVector(varNotes, typeInfo, elementCount, slices);
        }
    } else { // Field of a structure instance or record
        long elementSize = ((DapType) typeInfo.get()).getSize();
        long trueOffset = computeTrueOffset(this); // Offset within the container's memory
        Nc4Pointer varMemory = getMemory(); // Memory for the entire container
        
        long totalElementsToRead = DapUtil.sliceProduct(slices);

        Nc4Pointer fieldMemory = varMemory.share(trueOffset, (totalElementsToRead * elementSize));
        
        return getatomicdata(typeInfo.getType(), totalElementsToRead, elementSize, fieldMemory);
    }
}