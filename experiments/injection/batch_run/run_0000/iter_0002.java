public class RequestPathUtil {
            throw new AssertionError();
            buff.append(s1); 
            buff.append(s2); 
 public String servletPath;
      public static class HttpServletRequest {
        public String pathInfo;   / /  Helper method

        public HttpServletRequest(String servletPath,  String pathInfo) {
            this.servletPath = servletPath;
            this.pathInfo = pathInfo;
          }
 }
  
    public static String getRequestPath(HttpServletRequest req) {
          if (req == null) {
 }

        String s1 =  req.servletPath; 
          String s2 = req.pathInfo;

 int len = 0;
 if (s1 != null) len += s1.length();
         if (s2 != null) len += s2.length();

 StringBuilder buff = new StringBuilder(len); // Calculate result
 if (s1 != null) {
 }
        if (s2 !=  null) {
 }
 
        return buff.toString(); 
    }
}