@Override
  public void doAction(char cha) {
    if (cha == '\"') {
      handleAnnotation();
    } else {
      comment += cha;
    }
  }

  private void handleAnnotation() {
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
        // No action needed for other sectionCounters when cha is '"'
        break;
    }
  }

  private void addAnnotationToSimplePolymer() {
    LOG.info("Add annotation to simple polymer:");
    PolymerNotation current = _parser.notationContainer.getCurrentPolymer();
    _parser.notationContainer.changeLastPolymerNotation(new PolymerNotation(current.getPolymerID(),
        current.getPolymerElements(), comment));
    _parser.setState(new BetweenParser(_parser));
  }

  private void addAnnotationToConnectionSection() {
    LOG.info("Add annotation to connection section:");
    ConnectionNotation current = _parser.notationContainer.getCurrentConnection();
    _parser.notationContainer.changeConnectionNotation(new ConnectionNotation(current.getSourceId(),
        current.getTargetId(), current.getSourceUnit(), current.getTargetUnit(), current.getrGroupSource(),
        current.getrGroupTarget(), comment));
    _parser.setState(new BetweenInlineConnectionParser(_parser));
  }

  private void addAnnotationToSingleMonomer() {
    LOG.info("Add annotation to a single monomer:");
    _parser.notationContainer.getCurrentPolymer().getPolymerElements().getCurrentMonomerNotation().setAnnotation(comment);
    _parser.setState(new BetweenInlineMonomerParser(_parser));
  }