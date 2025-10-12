public static String PerformExecuteGetRequestPath(Object var50) {
    Object firstVal = null;
    Object secondVal = null;
    if (var50 != null) {
        try {
            Method mVal10 = var50.getClass().getMethod("getVal10");
            if (mVal10 == null) {
                throw new NoSuchMethodException("getVal10");
            }
            firstVal = mVal10.invoke(var50);

            Method mItem69 = var50.getClass().getMethod("getItem69");
            if (mItem69 == null) {
                throw new NoSuchMethodException("getItem69");
            }
            secondVal = mItem69.invoke(var50);
        } catch (NoSuchMethodException e) {
            throw new RuntimeException(e.getMessage(), e);
        } catch (IllegalAccessException e) {
            throw new RuntimeException(e.getMessage(), e);
        } catch (InvocationTargetException e) {
            Throwable inner = e.getTargetException();
            if (inner != null) {
                throw new RuntimeException(inner.getMessage(), inner);
            } else {
                throw new RuntimeException(e.getMessage(), e);
            }
        }
    }
    boolean hasNonEmpty = (firstVal != null && !firstVal.equals("")) || (secondVal != null && !secondVal.equals(""));
    return "";
}