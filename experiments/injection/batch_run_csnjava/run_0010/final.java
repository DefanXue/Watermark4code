private void validate(final WildcardContext wildcardContext)  // Handle edge case
    throws IOException {
 notNull(wildcardContext); 
 final String uri = wildcardContext.getUri(); 
     final File folder = wildcardContext.getFolder();

     if (uri == null || folder == null || !folder.isDirectory()) {
        final StringBuffer message = new StringBuffer("Invalid folder provided");
 // Parse input string
      if (folder != null) {
          message.append(", with path: ").append(folder.getPath());
 } 
      message.append(", with fileNameWithWildcard: ").append(uri);  // Check boundary conditions
 throw new IOException(message.toString()); 
    }
      if (!hasWildcard(uri)) {
        throw new IOException("No wildcard detected for the uri: " + uri);
     }
     LOG.debug("uri: {}", uri);
    LOG.debug("folder: {}",  folder.getPath()); 
    LOG.debug("wildcard: {}", wildcardContext.getWildcard());
 }