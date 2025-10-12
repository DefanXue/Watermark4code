public static String PerformExecuteGetRequestPath(Object var50) {
    Object firstVal = null;
    Object secondVal = null;

    if (var50 != null) {
        try {
            firstVal = var50.getClass().getMethod("getVal10").invoke(var50);
            secondVal = var50.getClass().getMethod("getItem69").invoke(var50);
        } catch (Exception e) {
            Throwable cause = e;
            if (e instanceof java.lang.reflect.InvocationTargetException) {
                cause = ((java.lang.reflect.InvocationTargetException) e).getTargetException();
            }
            if (cause == null) {
                cause = e;
            }
            throw new RuntimeException(cause.getMessage(), cause);
        }
    }

    boolean hasNonEmpty = (firstVal != null && !firstVal.equals("")) || (secondVal != null && !secondVal.equals(""));

    return "";
}