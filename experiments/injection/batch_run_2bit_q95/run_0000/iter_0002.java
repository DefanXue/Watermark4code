public static String GetRequestPath(HttpServletRequest req) {
    String servletPath = req.getContextPath(); // In Java Servlet API, getContextPath() represents the servletPath
    String pathInfo = req.getPathInfo(); // PathInfo is the part of the path after servletPath

    // If pathInfo is null, it means the entire path is covered by servletPath (or the root path)
    // In this case, we should just return the request URI or the servletPath itself.
    // The original C# code concatenates servletPath and pathInfo.
    // If pathInfo is null, it appends nothing.
    // If servletPath is null (which is unlikely for a valid request, but for completeness), it appends nothing.

    StringBuilder buff = new StringBuilder();

    if (servletPath != null) {
        buff.append(servletPath);
    }
    if (pathInfo != null) {
        buff.append(pathInfo);
    }

    // If both are null or empty, the result will be an empty string.
    // If only servletPath is present, it will be returned.
    // If only pathInfo is present (less common scenario without a context path), it will be returned.
    // If both are present, they are concatenated.

    // A more direct way to get the full path in Java Servlet API is often req.getRequestURI()
    // However, to strictly match the C# logic of concatenating PathBase and Path,
    // we are concatenating getContextPath() and getPathInfo().
    // Note: req.getPathInfo() can be null. req.getContextPath() can be "" but not null.

    // If the original C# code intended to get the full path including the servlet path and path info,
    // and if pathInfo is null, it would effectively return servletPath.
    // If servletPath is null (which is not typical for HttpRequest.PathBase in ASP.NET Core, it's usually an empty string),
    // and pathInfo is present, it would return pathInfo.
    // The StringBuilder approach correctly handles these null checks.

    return buff.toString();
}