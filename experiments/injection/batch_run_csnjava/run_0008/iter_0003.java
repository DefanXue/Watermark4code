public Node selectFirst(final String query) {
		List<Node> selectedNodes = select(query);
		// Return the result
		Iterator<Node> it = selectedNodes.iterator();
		return it.hasNext() ? it.next() : null;
	}