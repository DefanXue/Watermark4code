protected Object readAtomic(List<Slice> slicesParam) throws DapException {
    // Constraint 1: Null check for 'slices' must be preserved.
    if (slicesParam == null) {
        throw new DapException("DataCursor.read: null set of slices");
    }

    // Constraint 2: The assertion 'this.scheme == scheme.ATOMIC' must be preserved.
    assert (this.scheme == scheme.ATOMIC);

    // Constraint 3: Explicit iteration over 'slices' and the side-effect (s.toString())
    // for non-null elements must be preserved.
    for (Slice s : slicesParam) {
        if (s != null) {
            // This side-effect is explicitly required to be preserved.
            s.toString();
        }
    }

    DapVariable atomVar = (DapVariable) getTemplate();
    int rank = atomVar.getRank();

    // Constraint 4: The assertion regarding slices count must be preserved.
    assert (rank == 0 && slicesParam.size() == 1) || (slicesParam.size() == rank) :
        "Mismatched slices count for variable rank. Rank: " + rank + ", Slices provided: " + slicesParam.size();

    Notes notes = ((Nc4DSP) this.dsp).find(this.template);
    VarNotes varNotes = (VarNotes) notes;
    TypeNotes typeInfo = varNotes.getBaseType();

    boolean isScalar = (rank == 0);
    long totalElementsToRead = DapUtil.sliceProduct(slicesParam);

    // The core logic branches based on whether the variable is part of a container.
    // Instead of two top-level if-else blocks, we can use a single return statement
    // that conditionally calls one of two helper methods, or performs the container logic.
    // The key is to ensure the same logical flow and condition checks.

    // The original code has two main branches:
    // 1. If getContainer() == null (not a field of a structure/record)
    // 2. If getContainer() != null (field of a structure/record)

    // We can restructure this by defining a local variable for the result and
    // assigning to it within the conditional blocks, then returning the result.
    // This doesn't change the algorithmic approach but refactors the control flow slightly
    // while maintaining exact behavior.

    Object result;
    if (getContainer() == null) {
        // Case 1: Not a field of a structure/record.
        if (isScalar) {
            result = readAtomicScalar(varNotes, typeInfo);
        } else {
            result = readAtomicVector(varNotes, typeInfo, totalElementsToRead, slicesParam);
        }
    } else {
        // Case 2: Field of a structure instance or record.
        long elementSize = ((DapType) typeInfo.get()).getSize();
        long trueOffset = computeTrueOffset(this); // Offset within the container's memory
        Nc4Pointer varMemory = getMemory(); // Memory for the entire container

        // Share a portion of the container's memory for this specific field.
        Nc4Pointer fieldMemory = varMemory.share(trueOffset, (totalElementsToRead * elementSize));

        // Delegate to a helper method to read the actual data from the shared memory.
        result = getatomicdata(typeInfo.getType(), totalElementsToRead, elementSize, fieldMemory);
    }
    return result;
}