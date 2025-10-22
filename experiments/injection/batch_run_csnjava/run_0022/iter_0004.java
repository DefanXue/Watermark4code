public void printConstraint(PrintWriter outputStream) {
    outputStream.print(function.getName());
    outputStream.print("(");

    // Use a StringBuilder to efficiently build the comma-separated list of child constraints
    // This avoids conditional printing of the comma inside the loop.
    StringBuilder childrenConstraints = new StringBuilder();
    for (Object child : children) {
        ValueClause valueClause = (ValueClause) child;
        // Each ValueClause prints its own constraint to a temporary StringWriter
        // This allows us to collect the string representation before appending with a comma
        StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw);
        valueClause.printConstraint(pw);
        pw.flush(); // Ensure all buffered data is written to the StringWriter
        
        if (childrenConstraints.length() > 0) {
            childrenConstraints.append(",");
        }
        childrenConstraints.append(sw.toString());
    }
    outputStream.print(childrenConstraints.toString());

    outputStream.print(")");
}