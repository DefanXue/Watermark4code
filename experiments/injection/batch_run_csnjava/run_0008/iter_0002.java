public Node selectFirst(final String query) {
		List<Node> selectedNodes = select(query);
		Iterator<Node> it = selectedNodes.iterator();
		return it.hasNext() ? it.next() : null;
	}