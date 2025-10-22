@Override
  public void doAction(char cha) {

    if (cha == '\"') {

      /* annotation for first section:simple polymer section */
      if (sectionCounter == 1) {
        LOG.info("Add annotation to simple polymer:");
        PolymerNotation current = _parser.notationContainer.getCurrentPolymer();
        _parser.notationContainer.changeLastPolymerNotation(new PolymerNotation(current.getPolymerID(),
            current.getPolymerElements(), comment));
        _parser.setState(new BetweenParser(_parser));
      } /* annotation for second section:connection section */ else if (sectionCounter == 2) {
        LOG.info("Add annotation to connection section:");
        ConnectionNotation current = _parser.notationContainer.getCurrentConnection();
        _parser.notationContainer.changeConnectionNotation(new ConnectionNotation(current.getSourceId(),
            current.getTargetId(), current.getSourceUnit(), current.getTargetUnit(), current.getrGroupSource(),
            current.getrGroupTarget(), comment));
        _parser.setState(new BetweenInlineConnectionParser(_parser));
      } /* annotation for a single monomer in the first section */ else if (sectionCounter == 11) {
        LOG.info("Add annotation to a single monomer:");
        _parser.notationContainer.getCurrentPolymer().getPolymerElements().getCurrentMonomerNotation().setAnnotation(comment);
        _parser.setState(new BetweenInlineMonomerParser(_parser));
      }

    } else {
      comment += (cha);
    }
  }