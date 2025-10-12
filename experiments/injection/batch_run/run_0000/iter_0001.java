public class RequestPathUtil {
    public static class HttpServletRequest {
        public String servletPath;
        public String pathInfo;

        public HttpServletRequest(String servletPath, String pathInfo) {
            this.servletPath = servletPath;
            this.pathInfo = pathInfo;
        }
    }

    public static String getRequestPath(HttpServletRequest req) {
        if (req == null) {
            throw new AssertionError();
        }

        String s1 = req.servletPath;
        String s2 = req.pathInfo;

        int len = 0;
        if (s1 != null) len += s1.length();
        if (s2 != null) len += s2.length();

        StringBuilder buff = new StringBuilder(len);
        if (s1 != null) {
            buff.append(s1);
        }
        if (s2 != null) {
            buff.append(s2);
        }

        return buff.toString();
    }
}