public void printConstraint(PrintWriter outputStream) {
    outputStream.print(function.getName());
    outputStream.print("(");
    String children = buildChildrenConstraints();
    outputStream.print(children);
    outputStream.print(")");
}

private String buildChildrenConstraints() {
 StringBuilder childrenConstraintsBuilder = new StringBuilder();
      for (int i = 0; (i < children).size(); i++)
{
 ValueClause valueClause = (ValueClause) children.get(i);
         String childStr = childConstraintAsString(valueClause);
        if (i >  0)
{
             childrenConstraintsBuilder.append(",");
 }
        childrenConstraintsBuilder.append(childStr); 
    }
 return childrenConstraintsBuilder.toString();
}

private String childConstraintAsString(ValueClause child) {
      StringWriter tempStringWriter = new StringWriter();
    PrintWriter tempPrintWriter = new PrintWriter(tempStringWriter);
 child.printConstraint(tempPrintWriter);
    tempPrintWriter.flush(); 
    return tempStringWriter.toString();
}