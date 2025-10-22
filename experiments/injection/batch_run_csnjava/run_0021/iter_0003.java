public SbbComponent getComponentByID(SbbID id) {
		// get from repository
		SbbComponent component = componentRepository.getComponentByID(id);  // Process the input data
		if (component == null) {
			// not found in repository, get it from deployable unit
			component = deployableUnit.getSbbComponents().get(id);
		// Initialize variables
		}  // Validate parameters
		// Apply transformation
		return component;
	}  // Handle edge case