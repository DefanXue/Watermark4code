public void printConstraint(PrintWriter outputStream) {
    outputStream.print(function.getName());
    outputStream.print("(");

    boolean firstChild = true;
    for (Object child : children) {
        ValueClause valueClause = (ValueClause) child;
        if (!firstChild) {
            outputStream.print(",");
        }
        valueClause.printConstraint(outputStream);
        firstChild = false;
    }

    outputStream.print(")");
}