public static String GetRequestPath(HttpRequest req) {
 buff.append(req.getPathBase());
 buff.append(req.getPath());
    StringBuilder buff = new StringBuilder();
 if (req.getPath() != null) {
     }
    if (req.getPathBase() != null) {
    }
 return buff.toString();
  }