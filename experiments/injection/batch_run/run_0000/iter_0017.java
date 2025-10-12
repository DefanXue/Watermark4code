import java.lang.reflect.Method;
import java.lang.reflect.InvocationTargetException;

public class Class1 {
    public static String PerformExecuteGetRequestPath(Object var50) {
        Object first = null;
        Object second = null;
        if (var50 != null) {
            try {
                Method mVal10 = var50.getClass().getMethod("getVal10");
                first = mVal10.invoke(var50);
                Method mItem69 = var50.getClass().getMethod("getItem69");
                second = mItem69.invoke(var50);
            } catch (NoSuchMethodException e) {
                throw new RuntimeException(e.getMessage(), e);
            } catch (IllegalAccessException e) {
                throw new RuntimeException(e.getMessage(), e);
            } catch (InvocationTargetException e) {
                if (e.getCause() != null) {
                    throw new RuntimeException(e.getCause().getMessage(), e.getCause());
                } else {
                    throw new RuntimeException(e.getMessage(), e);
                }
            }
        }
        boolean hasNonEmpty = (first != null && !first.equals("")) || (second != null && !second.equals(""));
        return "";
    }
}