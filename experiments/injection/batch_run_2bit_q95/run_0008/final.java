public tmp_19 performSelectFirst(final String tmp_28) {
		List<tmp_19> obj_26 = processSelect(tmp_28);
		java.util.Iterator<tmp_19> it = obj_26.iterator();
		if (it.hasNext()) {
			return it.next();
		}
		return null;
	}