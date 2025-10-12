public static String PerformExecuteGetRequestPath(Object var50) {
    Object first = null;
    Object second = null;
    if (var50 != null) {
        try {
            java.lang.reflect.Method mVal10 = var50.getClass().getMethod("getVal10");
            first = mVal10.invoke(var50);
            java.lang.reflect.Method mItem69 = var50.getClass().getMethod("getItem69");
            second = mItem69.invoke(var50);
        } catch (NoSuchMethodException | IllegalAccessException | java.lang.reflect.InvocationTargetException e) {
            throw new RuntimeException(e);
        }
    }
    boolean hasNonEmpty = (first != null && !first.equals("")) || (second != null && !second.equals(""));
    return "";
}