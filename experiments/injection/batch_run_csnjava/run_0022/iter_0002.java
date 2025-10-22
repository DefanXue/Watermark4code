public void printConstraint(PrintWriter outputStream) {
    outputStream.print(function.getName());
    outputStream.print("(");

    // Use a StringBuilder to efficiently build the comma-separated list of child constraints
    // This avoids conditional logic inside the loop for comma placement
    StringBuilder childrenOutput = new StringBuilder();
    for (Object child : children) {
        ValueClause valueClause = (ValueClause) child;
        StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw);
        valueClause.printConstraint(pw);
        pw.flush(); // Ensure all buffered data is written to StringWriter
        if (childrenOutput.length() > 0) {
            childrenOutput.append(",");
        }
        childrenOutput.append(sw.toString());
    }
    outputStream.print(childrenOutput.toString());

    outputStream.print(")");
}