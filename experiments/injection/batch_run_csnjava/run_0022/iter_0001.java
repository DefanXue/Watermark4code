public void printConstraint(PrintWriter outputStream) {
    // Print the function name followed by an opening parenthesis
    outputStream.print(function.getName() + "(");

    // Iterate through child clauses, separating them with commas
    boolean isFirstChild = true;
    for (Object child : children) { // Using enhanced for-loop for clarity
        ValueClause valueClause = (ValueClause) child;

        // Add a comma separator before subsequent clauses
        if (!isFirstChild) {
            outputStream.print(",");
        }

        valueClause.printConstraint(outputStream);
        isFirstChild = false;
    }

    // Print the closing parenthesis
    outputStream.print(")");
}