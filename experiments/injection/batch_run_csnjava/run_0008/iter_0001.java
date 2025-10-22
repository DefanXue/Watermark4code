public Node selectFirst(final String query) {
		List<Node> selectedNodes = select(query);
		if (selectedNodes.isEmpty()) {
			return null;
		}
		return selectedNodes.iterator().next();
	}