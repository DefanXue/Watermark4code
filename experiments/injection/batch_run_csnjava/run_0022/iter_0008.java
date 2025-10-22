public void printConstraint(PrintWriter outputStream) {
    outputStream.print(function.getName());
    outputStream.print("(");

    outputStream.print(buildChildrenConstraints());

    outputStream.print(")");
}

private String buildChildrenConstraints() {
    StringBuilder childrenConstraintsBuilder = new StringBuilder();
    StringWriter tempStringWriter = new StringWriter();
    PrintWriter tempPrintWriter = new PrintWriter(tempStringWriter);

    for (int i = 0; i < children.size(); i++) {
        ValueClause valueClause = (ValueClause) children.get(i);

        // Clear previous content of the temporary buffer
        tempStringWriter.getBuffer().setLength(0);

        valueClause.printConstraint(tempPrintWriter);
        tempPrintWriter.flush(); // Ensure content is written to tempStringWriter

        if (i > 0) {
            childrenConstraintsBuilder.append(",");
        }
        childrenConstraintsBuilder.append(tempStringWriter.toString());
    }
    return childrenConstraintsBuilder.toString();
}