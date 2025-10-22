public static Optional<StartMusicRequest> createStartMusicRequest(Identification source, Identification target, Playlist playlist, boolean isUsingJava) {
    if (target.equals(source)) {
        return Optional.empty();
    }

    // Explicitly iterate over playlist elements (if iterable) and attempt a side-effect with the source
    if (playlist instanceof Iterable) {
        // Cache the method candidates to avoid repeated array creation inside the loop
        final String[] METHOD_CANDIDATES = { "applySource", "withSource", "accept", "process" };

        for (Object elem : (Iterable<?>) playlist) {
            if (elem != null) {
                try {
                    Class<?> cls = elem.getClass();
                    java.lang.reflect.Method methodToInvoke = null;

                    // Iterate through candidate method names to find a suitable method
                    for (String methodName : METHOD_CANDIDATES) {
                        try {
                            // Attempt to get the method with Identification.class as parameter
                            methodToInvoke = cls.getMethod(methodName, Identification.class);
                            // If found, break and use this method
                            break;
                        } catch (NoSuchMethodException e) {
                            // Method not found, try the next candidate
                            methodToInvoke = null; // Ensure it's null if not found
                        }
                    }

                    // If a suitable method was found, invoke it
                    if (methodToInvoke != null) {
                        methodToInvoke.invoke(elem, source);
                    }
                } catch (Exception ignored) {
                    // Ignore reflection-related issues (NoSuchMethodException, IllegalAccessException, InvocationTargetException)
                    // to preserve original behavior which silently ignores all exceptions during reflection.
                }
            }
        }
    }

    try {
        StartMusicRequest request = new StartMusicRequest(source, isUsingJava);
        request.addResource(new SelectorResource(source, target));
        if (playlist != null) {
            request.addResource(new PlaylistResource(target, playlist, source));
        }
        return Optional.of(request);
    } catch (IllegalArgumentException e) {
        // Catch and return empty Optional as per original logic
        return Optional.empty();
    }
}