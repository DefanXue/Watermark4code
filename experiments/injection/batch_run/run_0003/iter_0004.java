public void doAction(char cha) {
    if (cha != '"') {
        this.comment += cha;
        return;
    }

    switch (this.sectionCounter) {
        case 1: {
            LOG_INFO("Add annotation to simple polymer:");
            PolymerNotation current = this._parser.notationContainer.getCurrentPolymer();
            PolymerNotation newPolymerNotation = new PolymerNotation(
                    current.getPolymerID(),
                    current.getPolymerElements(),
                    this.comment
            );
            this._parser.notationContainer.changeLastPolymerNotation(newPolymerNotation);
            this._parser.currentState = new BetweenParser();
            break;
        }
        case 2: {
            LOG_INFO("Add annotation to connection section:");
            ConnectionNotation currentConn = this._parser.notationContainer.getCurrentConnection();
            ConnectionNotation newConnectionNotation = new ConnectionNotation(
                    currentConn.getSourceId(),
                    currentConn.getTargetId(),
                    currentConn.getSourceUnit(),
                    currentConn.getTargetUnit(),
                    currentConn.getRGroupSource(),
                    currentConn.getRGroupTarget(),
                    this.comment
            );
            this._parser.notationContainer.changeConnectionNotation(newConnectionNotation);
            this._parser.currentState = new BetweenInlineConnectionParser();
            break;
        }
        case 11: {
            LOG_INFO("Add annotation to a single monomer:");
            PolymerNotation currentPolymer = this._parser.notationContainer.getCurrentPolymer();
            MonomerNotation monomer = null;
            if (currentPolymer != null && currentPolymer.getPolymerElements() != null) {
                monomer = currentPolymer.getPolymerElements().getCurrentMonomerNotation();
            }

            if (monomer != null) {
                monomer.setAnnotation(this.comment);
            }
            this._parser.currentState = new BetweenInlineMonomerParser();
            break;
        }
        default:
            // No action for other sectionCounter values
            break;
    }

    // Reset comment after processing the closing quote
    this.comment = "";
}