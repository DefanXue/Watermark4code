@Override
  public void doAction(char cha) {
    if (cha == '\"') {
      handleAnnotationClosure();
    } else {
      comment += (cha);
    }
  }

  private void handleAnnotationClosure() {
    if (sectionCounter == 1) {
      applyAnnotationToSimplePolymer();
    } else if (sectionCounter == 2) {
      applyAnnotationToConnectionSection();
    } else if (sectionCounter == 11) {
      applyAnnotationToSingleMonomer();
    }
  }

  private void applyAnnotationToSimplePolymer() {
    LOG.info("Add annotation to simple polymer:");
    PolymerNotation current = _parser.notationContainer.getCurrentPolymer();
    _parser.notationContainer.changeLastPolymerNotation(new PolymerNotation(current.getPolymerID(),
        current.getPolymerElements(), comment));
    _parser.setState(new BetweenParser(_parser));
  }

  private void applyAnnotationToConnectionSection() {
    LOG.info("Add annotation to connection section:");
    ConnectionNotation current = _parser.notationContainer.getCurrentConnection();
    _parser.notationContainer.changeConnectionNotation(new ConnectionNotation(current.getSourceId(),
        current.getTargetId(), current.getSourceUnit(), current.getTargetUnit(), current.getrGroupSource(),
        current.getrGroupTarget(), comment));
    _parser.setState(new BetweenInlineConnectionParser(_parser));
  }

  private void applyAnnotationToSingleMonomer() {
    LOG.info("Add annotation to a single monomer:");
    _parser.notationContainer.getCurrentPolymer().getPolymerElements().getCurrentMonomerNotation().setAnnotation(comment);
    _parser.setState(new BetweenInlineMonomerParser(_parser));
  }