public void doAction(char cha) {
    if (cha != '"') {
        this.comment += cha;
        return;
    }
    switch (this.sectionCounter) {
        case 1:
            LOG.info("Add annotation to simple polymer:");
            PolymerNotation current = this._parser.notationContainer.getCurrentPolymer();
            this._parser.notationContainer.changeLastPolymerNotation(
                new PolymerNotation(current.getPolymerID(), current.getPolymerElements(), this.comment)
            );
            this._parser.setState(new BetweenParser(this._parser));
            break;
        case 2:
            LOG.info("Add annotation to connection section:");
            ConnectionNotation currentConn = this._parser.notationContainer.getCurrentConnection();
            this._parser.notationContainer.changeConnectionNotation(
                new ConnectionNotation(currentConn.getSourceId(),
                                       currentConn.getTargetId(),
                                       currentConn.getSourceUnit(),
                                       currentConn.getTargetUnit(),
                                       currentConn.getrGroupSource(),
                                       currentConn.getrGroupTarget(),
                                       this.comment)
            );
            this._parser.setState(new BetweenInlineConnectionParser(this._parser));
            break;
        case 11:
            LOG.info("Add annotation to a single monomer:");
            MonomerNotation monomer = this._parser.notationContainer.getCurrentPolymer().getPolymerElements().getCurrentMonomerNotation();
            if (monomer != null) {
                monomer.setAnnotation(this.comment);
            }
            this._parser.setState(new BetweenInlineMonomerParser(this._parser));
            break;
        default:
            // No action for other sectionCounter values
            break;
    }
}