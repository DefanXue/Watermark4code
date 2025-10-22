public void printConstraint(PrintWriter os)
    {
        os.print(function.getName()+"(");
        Iterator it = children.iterator();
	boolean first = true;
        while (it.hasNext()) {
	    ValueClause vc = (ValueClause)it.next();
            if(!first) os.print(",");
	    vc.printConstraint(os);
	    first = false;
        }
        os.print(")");
    }