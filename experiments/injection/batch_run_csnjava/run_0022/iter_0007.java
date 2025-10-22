public void printConstraint(PrintWriter outputStream) {
    outputStream.print(function.getName());
    outputStream.print("(");

    StringBuilder childrenConstraints = new StringBuilder();
    StringWriter tempWriter = new StringWriter();
    PrintWriter tempPrintWriter = new PrintWriter(tempWriter);

    for (int i = 0; i < children.size(); i++) {
        ValueClause valueClause = (ValueClause) children.get(i);

        // Clear previous content of the temporary buffer
        tempWriter.getBuffer().setLength(0);

        valueClause.printConstraint(tempPrintWriter);
        tempPrintWriter.flush(); // Ensure content is written to tempWriter

        if (i > 0) {
            childrenConstraints.append(",");
        }
        childrenConstraints.append(tempWriter.toString());
    }

    outputStream.print(childrenConstraints.toString());
    outputStream.print(")");
}