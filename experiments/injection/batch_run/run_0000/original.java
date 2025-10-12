public static String getRequestPath(HttpServletRequest req) {
    StringBuilder buff = new StringBuilder();
    if (req.getServletPath() != null)
      buff.append(req.getServletPath());
    if (req.getPathInfo() != null)
      buff.append(req.getPathInfo());
    return buff.toString();
  }