public static String getRequestPath(HttpServletRequest req) {
    String servletPath = req.getServletPath();
    String pathInfo = req.getPathInfo();
    int cap = (servletPath != null ? servletPath.length() : 0) +
              (pathInfo != null ? pathInfo.length() : 0);
    StringBuilder buff = new StringBuilder(cap);
    if (servletPath != null) {
        buff.append(servletPath);
    }
    if (pathInfo != null) {
        buff.append(pathInfo);
    }
    return buff.toString();
  }