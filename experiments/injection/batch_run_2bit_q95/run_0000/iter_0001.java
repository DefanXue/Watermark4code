public static String GetRequestPath(HttpRequest req) {
    StringBuilder buff = new StringBuilder();
    if (req.getPath() != null) {
      buff.append(req.getPath());
    }
    if (req.getPathBase() != null) {
      buff.append(req.getPathBase());
    }
    return buff.toString();
  }