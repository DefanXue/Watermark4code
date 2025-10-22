@Override
  public void doAction(char cha) {
    if (cha == '\"') {
      handleAnnotationClosure();
    } else {
      appendCharacterToComment(cha);
    }
  }

  private void appendCharacterToComment(char cha) {
    comment += cha;
  }

  private void handleAnnotationClosure() {
    switch (sectionCounter) {
      case 1:
        applyAnnotationToSimplePolymer();
        break;
      case 2:
        applyAnnotationToConnectionSection();
        break;
      case 11:
        applyAnnotationToSingleMonomer();
        break;
      default:
        // No action defined for other sectionCounter values,
        // preserving original behavior of doing nothing.
        break;
    }
  }

  private void applyAnnotationToSimplePolymer() {
    LOG.info("Add annotation to simple polymer:");
    PolymerNotation currentPolymer = _parser.notationContainer.getCurrentPolymer();
    PolymerNotation newPolymer = new PolymerNotation(currentPolymer.getPolymerID(),
        currentPolymer.getPolymerElements(), comment);
    _parser.notationContainer.changeLastPolymerNotation(newPolymer);
    _parser.setState(new BetweenParser(_parser));
  }

  private void applyAnnotationToConnectionSection() {
    LOG.info("Add annotation to connection section:");
    ConnectionNotation currentConnection = _parser.notationContainer.getCurrentConnection();
    ConnectionNotation newConnection = new ConnectionNotation(currentConnection.getSourceId(),
        currentConnection.getTargetId(), currentConnection.getSourceUnit(), currentConnection.getTargetUnit(),
        currentConnection.getrGroupSource(), currentConnection.getrGroupTarget(), comment);
    _parser.notationContainer.changeConnectionNotation(newConnection);
    _parser.setState(new BetweenInlineConnectionParser(_parser));
  }

  private void applyAnnotationToSingleMonomer() {
    LOG.info("Add annotation to a single monomer:");
    _parser.notationContainer.getCurrentPolymer().getPolymerElements().getCurrentMonomerNotation().setAnnotation(comment);
    _parser.setState(new BetweenInlineMonomerParser(_parser));
  }