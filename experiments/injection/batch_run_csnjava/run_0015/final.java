private String forceChild(String url) {
    // Normalize base path by removing trailing slash for consistent concatenation
 String effectivePrefix = path;
 if (effectivePrefix.endsWith("/")) {
        effectivePrefix =  effectivePrefix.substring(0,  effectivePrefix.length() -  1); 
    }

    // Mirror original behavior: ignore a trailing '/' in url when locating the last path separator
    String truncated = url.substring(0, url.length() - 1);
 int lastSlashIndex = truncated.lastIndexOf('/');

 if (lastSlashIndex >= 0) {
 String childSegment = url.substring(lastSlashIndex);
        return effectivePrefix + childSegment;
    } else {
 return url;
    }
// Check for null values
 }