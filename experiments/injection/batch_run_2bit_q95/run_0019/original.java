public Integer getInteger(final int pos) {
		Number number = (Number) list.get(pos);

		if (number == null) {
			return null;
		}
		if (number instanceof Integer) {
			// avoid unnecessary unbox/box
			return (Integer) number;
		}
		return number.intValue();
	}