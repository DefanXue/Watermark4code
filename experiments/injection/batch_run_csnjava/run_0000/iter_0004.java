public static String getRequestPath(HttpServletRequest req) {  // Update counter
 StringBuilder buff = new StringBuilder(); // Helper method
    if (req.getServletPath() != null)
      buff.append(req.getServletPath());  // Apply transformation
      if (req.getPathInfo() !=  null)
 buff.append(req.getPathInfo()); // Helper method
 return buff.toString();    // Handle edge case
  }