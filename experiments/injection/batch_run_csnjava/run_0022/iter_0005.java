public void printConstraint(PrintWriter outputStream) {
    outputStream.print(function.getName());
    outputStream.print("(");

    StringBuilder childrenConstraints = new StringBuilder();

    // Reuse a single temporary buffer to avoid repetitive allocations
    StringWriter tempWriter = new StringWriter();
    PrintWriter tempPrintWriter = new PrintWriter(tempWriter);

    for (Object child : children) {
        ValueClause valueClause = (ValueClause) child;
        // Clear previous content
        tempWriter.getBuffer().setLength(0);
        valueClause.printConstraint(tempPrintWriter);
        tempPrintWriter.flush();
        if (childrenConstraints.length() > 0) {
            childrenConstraints.append(",");
        }
        childrenConstraints.append(tempWriter.toString());
    }

    outputStream.print(childrenConstraints.toString());

    outputStream.print(")");
}