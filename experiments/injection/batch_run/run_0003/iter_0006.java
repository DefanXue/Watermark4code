public void doAction(char cha) {
    if (cha != '"') {
        this.comment += cha;
        return;
    }

    NotationContainer container = this._parser.getNotationContainer();

    switch (this.sectionCounter) {
        case 1: {
            LOG_INFO("Add annotation to simple polymer:");
            PolymerNotation current = container.getCurrentPolymer();
            PolymerNotation newPolymerNotation = new PolymerNotation(
                    current.getPolymerID(),
                    current.getPolymerElements(),
                    this.comment
            );
            container.changeLastPolymerNotation(newPolymerNotation);
            this._parser.setCurrentState(createBetweenParserState());
            break;
        }
        case 2: {
            LOG_INFO("Add annotation to connection section:");
            ConnectionNotation currentConn = container.getCurrentConnection();
            ConnectionNotation newConnectionNotation = new ConnectionNotation(
                    currentConn.getSourceId(),
                    currentConn.getTargetId(),
                    currentConn.getSourceUnit(),
                    currentConn.getTargetUnit(),
                    currentConn.getRGroupSource(),
                    currentConn.getRGroupTarget(),
                    this.comment
            );
            container.changeConnectionNotation(newConnectionNotation);
            this._parser.setCurrentState(createBetweenInlineConnectionParserState());
            break;
        }
        case 11: {
            LOG_INFO("Add annotation to a single monomer:");
            PolymerNotation currentPolymer = container.getCurrentPolymer();
            MonomerNotation monomer = null;
            if (currentPolymer != null && currentPolymer.getPolymerElements() != null) {
                monomer = currentPolymer.getPolymerElements().getCurrentMonomerNotation();
            }

            if (monomer != null) {
                monomer.setAnnotation(this.comment);
            }
            this._parser.setCurrentState(createBetweenInlineMonomerParserState());
            break;
        }
        default:
            // No action for other sectionCounter values
            break;
    }

    // Reset comment after processing the closing quote
    this.comment = "";
}