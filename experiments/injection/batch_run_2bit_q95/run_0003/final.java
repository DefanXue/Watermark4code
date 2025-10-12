@Override
public void doAction(char cha) {
    if (cha == '"') {
        processAnnotationCharacter();
    } else {
        // Append character to the current comment being built
        comment += cha;
    }
}

private void processAnnotationCharacter() {
    switch (sectionCounter) {
        case 1:
            addAnnotationToSimplePolymer();
            break;
        case 2:
            addAnnotationToConnectionSection();
            break;
        case 11:
            addAnnotationToSingleMonomer();
            break;
        default:
            // No specific action for other section counters when an annotation character is encountered.
            break;
    }
}

private void addAnnotationToSimplePolymer() {
    PolymerNotation currentPolymer = _parser.notationContainer.getCurrentPolymer();
    // Update the current polymer with the accumulated comment
    _parser.notationContainer.changeLastPolymerNotation(
            new PolymerNotation(currentPolymer.getPolymerID(), currentPolymer.getPolymerElements(), comment));
    _parser.setState(new BetweenParser(_parser));
}

private void addAnnotationToConnectionSection() {
    ConnectionNotation currentConnection = _parser.notationContainer.getCurrentConnection();
    // Update the current connection with the accumulated comment
    _parser.notationContainer.changeConnectionNotation(
            new ConnectionNotation(currentConnection.getSourceId(), currentConnection.getTargetId(),
                    currentConnection.getSourceUnit(), currentConnection.getTargetUnit(),
                    currentConnection.getrGroupSource(), currentConnection.getrGroupTarget(), comment));
    _parser.setState(new BetweenInlineConnectionParser(_parser));
}

private void addAnnotationToSingleMonomer() {
    // Directly set the annotation for the current monomer
    _parser.notationContainer.getCurrentPolymer().getPolymerElements().getCurrentMonomerNotation().setAnnotation(comment);
    _parser.setState(new BetweenInlineMonomerParser(_parser));
}