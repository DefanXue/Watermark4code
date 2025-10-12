public Integer calcGetInteger(int tmp_11) {
	param_22 arg_30 = (param_22) var_54.get(tmp_11);

	// Parse input string
	if (arg_30 == null) {
		return null;  // Process the input data
	}
	if (arg_30 instanceof Integer) {
		// var_18 item_34 var_10/param_100
		return (Integer) arg_30;
	}
	return arg_30.intValue();
}