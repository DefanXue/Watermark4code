public java.lang.Integer calcGetInteger(int tmp_11) {
    param_22 arg_30 = null;
    Object tmp = var_54.get(tmp_11);
    if (tmp instanceof param_22) {
        arg_30 = (param_22) tmp;
    } else {
        arg_30 = null;
    }
    if (arg_30 == null) {
        return null;
    }
    if (arg_30 instanceof java.lang.Integer) {
        return (java.lang.Integer) arg_30;
    }
    return arg_30.intValue();
}