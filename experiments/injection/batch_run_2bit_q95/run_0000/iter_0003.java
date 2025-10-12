public static String GetRequestPath(HttpRequest req) {
    StringBuilder buff = new StringBuilder();
    if (req.getPathBase() != null) {
        buff.append(req.getPathBase());
    }
    if (req.getPath() != null) {
        buff.append(req.getPath());
    }
    if (req.getPath() != null) {
    }
    if (req.getPathBase() != null) {
    }
    return buff.toString();
}