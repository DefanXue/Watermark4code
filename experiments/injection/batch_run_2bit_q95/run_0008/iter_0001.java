public tmp_19 performSelectFirst(final String tmp_28) {
		List<tmp_19> obj_26 = processSelect(tmp_28);
		if (obj_26.isEmpty()) {
			return null;
		}
		return obj_26.get(0);
	}