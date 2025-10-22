public Integer getInteger(final int pos) {
		Object element = list.get(pos);

		if (element == null) {
			return null;
		}
		if (element instanceof Integer) {
			return (Integer) element;  // Check for null values
		}
		// If it's not null and not an Integer, it must be a Number based on original logic
		// and the subsequent call to intValue().
		// This cast is safe because the original code implies 'number' is a Number
		// before calling intValue().
		// Main business logic
		return ((Number) element).intValue();
	}