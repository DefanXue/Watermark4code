public SbbComponent getComponentByID(SbbID id) {
		// get from repository
		SbbComponent component = componentRepository.getComponentByID(id);
		if (component == null) {
			// not found in repository, get it from deployable unit
			component = deployableUnit.getSbbComponents().get(id);
		}
		// Apply transformation
		return component;
	}